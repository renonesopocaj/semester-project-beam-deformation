"""
trainer.py
Implements a LightningModule subclass to use training
"""
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from typing import Any
from src.gnn.misc_helpers.misc import compute_mean_relative_mae
from src.gnn.preprocessing.normalizer import NormalizerFactory
from src.gnn.preprocessing.stats import StatsModule
from torch_geometric.loader import DataLoader
from src.gnn.training.losses import loss_factory, compute_thresholded_curvature_loss
from src.gnn.training.rollout import do_rollout
from src.gnn.training.scheduler import SchedulerWrapper
from src.gnn.validation.plot_rollout import (make_beam_comparison_gif, plot_roll_err_pos_vs_time,
                                             plot_err_heatmap_rollout, plot_volume_change, TARGET_FIELDS)
from src.gnn.validation.diagnostic_plot_helpers import plot_mesh_diagnostics

CHECK_NO_DEADLOCK = True
CURVATURE_LOSS_WEIGHT = 1.0

class LitModule(L.LightningModule):
    # Model, optimizer
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: SchedulerWrapper
    loss_fn: F
    # String info
    training_id: str
    eval_split_name: str
    target_names: list[str]
    # Directories
    model_dir: Path
    training_plots_dir: Path
    model_evaluation_dir: Path
    # Normalizer
    normalizer_method: str
    isotropic_normalizer: bool
    used_dataset_normalizer: NormalizerFactory
    # Features
    rest_minus_nonrest_pos: bool
    # Else
    log_test_media: bool
    node_stats: Any # Stats?
    edge_stats: Any # Stats?
    target_stats: Any # Stats?
    # Dt
    dt: float
    # Current rollout
    rollout_all_step_error: float  # current
    # Dataloader
    validation_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, model, optimizer, scheduler, loss_type, preprocess_cfg, training_id, model_root,
                 len_train_dataloader, enable_wandb_watch, train_cfg, theta_max):
        """
        Initializes the LitModule with model, optimizer, scheduler, loss function, and directory layout.

        Args:
            model: torch.nn.Module
                The GNN model.
            optimizer: torch.optim.Optimizer
                Optimizer used by Lightning automatic optimization.
            scheduler: object
                Existing custom scheduler wrapper.
            loss_type: str
                Loss name: mse, mae, or huber.
            preprocess_cfg: object
                Preprocessing configuration with attribute-style access.
            training_id: str
                Stable identifier for this training session.
            model_root: str or pathlib.Path
                Root directory that contains saved training folders.
            return: None
                Initializes the module.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_factory(loss_type, train_cfg.residual_list)
        self.residual_list = set(train_cfg.residual_list)
        self.theta_max = theta_max
        self.curvature_loss_weight = CURVATURE_LOSS_WEIGHT
        if "curvature" in self.residual_list and self.theta_max is None:
            raise ValueError("Curvature residual requested but theta_max was not provided.")

        self.training_id = training_id

        self.normalizer_method = preprocess_cfg.normalizer_method
        self.isotropic_normalizer = preprocess_cfg.isotropic_normalizer
        self.used_dataset_normalizer = NormalizerFactory.create_normalizer(self.normalizer_method,
                                                                           self.isotropic_normalizer)
        self.rest_minus_nonrest_pos = preprocess_cfg.displacement

        self.model_dir = Path(model_root) / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_plots_dir = self.model_dir / "training"
        self.training_plots_dir.mkdir(parents=True, exist_ok=True)
        self.model_evaluation_dir = self.model_dir / "model_evaluation"
        self.model_evaluation_dir.mkdir(parents=True, exist_ok=True)

        self.node_stats = None
        self.edge_stats = None
        self.target_stats = None
        self.dt = None
        self.validation_loader = None
        self.test_loader = None
        self.eval_split_name = "test"
        self.log_test_media = train_cfg.log_test_media
        self.rollout_all_step_error = 0.0
        self.enable_wandb_watch = enable_wandb_watch

        self.len_train_dataloader = len_train_dataloader
        self.enable_val = train_cfg.enable_val
        self.profile_loss_stress_vs_acceleration = train_cfg.profile_loss_stress_vs_acceleration
        self.enable_rollout_val = train_cfg.enable_rollout_val
        self.plot_diagnostics_advanced = train_cfg.plot_diagnostics_advanced

    def forward(self, batch):
        """
        Runs a forward pass through the underlying GNN model.

        Args:
            batch: Data
                PyG batch.
        return: torch.Tensor
            Model prediction.
        """
        return self.model(batch)

    def configure_optimizers(self):
        """
        Returns the optimizer to Lightning's training loop.

        Args:
            return: torch.optim.Optimizer
                Optimizer used by Lightning.
        """
        return self.optimizer

    def on_fit_start(self):
        """
        Called at the beginning of fit. Sets up W&B metric definitions and model watching.
        Only executed on rank 0 for distributed training safety.
        """
        if not self._has_wandb_logger() or not self.trainer.is_global_zero:
            return

        # Define all metrics with epoch as x-axis
        self._define_wandb_metrics()
        # Watch model gradients and parameters
        if self.enable_wandb_watch:
            self.logger.watch(self.model, log="gradients", log_freq=self.len_train_dataloader*5, log_graph=False)
        # Log initial epoch value
        # self.logger.experiment.log({"epoch": 0})

    def on_fit_end(self):
        """
        Called at the end of fit. Unwatches model to prevent memory leaks.
        """
        if self._has_wandb_logger() and self.trainer.is_global_zero and self.enable_wandb_watch:
            self.logger.experiment.unwatch(self.model)

    def on_save_checkpoint(self, checkpoint):
        """
        Saves custom scheduler state dict into the Lightning checkpoint alongside model weights.

        Args:
            checkpoint: dict
                Lightning checkpoint dictionary.
        """
        if self.scheduler is not None:
            checkpoint["custom_scheduler_state_dict"] = self.scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """
        Restores the custom scheduler state from a Lightning checkpoint when resuming training.

        Args:
            checkpoint: dict
                Lightning checkpoint dictionary.
        """
        if self.scheduler is not None and "custom_scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["custom_scheduler_state_dict"])

    def set_rollout_context(self, validation_loader, node_stats, edge_stats, target_stats, dt):
        """
        Stores in the attributes statistics and the validation loader needed to run autoregressive rollout evaluations
        during training. Called before training right after initialization.

        Stats are converted to StatsModule and registered as submodules so their tensors
        (registered as buffers) automatically move to the correct device with the parent module.

        Args:
            validation_loader: pyg.DataLoader
                Validation loader used for rollout.
            node_stats: Stats
                Node feature statistics.
            edge_stats: Stats
                Edge feature statistics.
            target_stats: Stats
                Target statistics.
            dt: float
                Time step size.
        """
        self.validation_loader = validation_loader
        # Convert Stats to StatsModule and register as submodules for automatic device handling
        self.node_stats = StatsModule.from_stats(node_stats)
        self.edge_stats = StatsModule.from_stats(edge_stats)
        self.target_stats = StatsModule.from_stats(target_stats)
        self.dt = dt

    def set_test_context(self, test_loader, split_name):
        """
        Configures the module for a final evaluation pass, specifying the loader, split label, and whether to emit plots.

        Args:
            test_loader: pyg.DataLoader
                Loader used for final evaluation rollout.
            split_name: str
                Prefix used for test logging.
            log_media: bool
                Whether to create and log media artifacts.
        """
        self.test_loader = test_loader
        self.eval_split_name = split_name

    def _collect_train_batch_metrics_noroll(self, pred, target, batch):
        """
        Collects only the training metrics that are actually needed during training.

        Args:
            pred: torch.Tensor
                Model prediction with shape `(num_nodes, 4)`.
            target: torch.Tensor
                Ground-truth tensor with shape `(num_nodes, 4)`.
            return: dict
                Dictionary with the minimal training metrics.
        """
        total_loss = self.loss_fn(pred, target)
        if "curvature" in self.residual_list:
            curvature_residual = compute_thresholded_curvature_loss(pred, batch, self.target_stats, self.dt, self.theta_max)
            total_loss = total_loss + self.curvature_loss_weight * curvature_residual
        metrics["train/loss"] = total_loss
        return metrics

    def _collect_test_batch_metrics_noroll(self, pred, target, prefix):
        """
        Collects the richer evaluation metrics used for test.

        Args:
            pred: torch.Tensor
                Model prediction with shape `(num_nodes, 4)`.
            target: torch.Tensor
                Ground-truth tensor with shape `(num_nodes, 4)`.
            prefix: str
                Metric namespace prefix, for example `val`, `test`, or `validation`.
            return: dict
                Dictionary with evaluation metrics.
        """
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()

        metrics = {f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
                   f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(pred_acc, target_acc),
                   f"{prefix}/stress_rel_mae": compute_mean_relative_mae(pred_stress, target_stress)}
        return metrics

    def _collect_val_batch_metrics_noroll(self, pred, target, prefix):
        """
        Collects the richer evaluation metrics used for validation.

        Args:
            pred: torch.Tensor
                Model prediction with shape `(num_nodes, 4)`.
            target: torch.Tensor
                Ground-truth tensor with shape `(num_nodes, 4)`.
            prefix: str
                Metric namespace prefix, for example `val`, `test`, or `validation`.
            return: dict
                Dictionary with evaluation metrics.
        """
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()
        if self.profile_loss_stress_vs_acceleration:
            metrics = {f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
                       f"{prefix}/acc_all_loss": self.loss_fn(pred_acc, target_acc),
                       f"{prefix}/stress_loss": self.loss_fn(pred_stress, target_stress)}
        else:
            metrics = {f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target)}
        return metrics

    def _log_metrics(self, metrics, prog_bar_keys, batch_size, sync_dist_keys):
        """
        Logs metrics one by one so only selected keys are synchronized across ranks.

        Args:
            metrics: dict
                Dictionary of tensor metrics.
            prog_bar_keys: list[str]
                Metrics that should appear in the progress bar.
            batch_size: int
                Batch size for Lightning aggregation.
            sync_dist_keys: list[str]
                Metric names that must be synchronized across ranks.
        """
        has_logger = self._has_wandb_logger()
        prog_bar_key_set = set(prog_bar_keys)
        sync_dist_key_set = set(sync_dist_keys)

        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True,
                     prog_bar=metric_name in prog_bar_key_set,
                     logger=has_logger, batch_size=batch_size,
                     sync_dist=False if CHECK_NO_DEADLOCK else metric_name in sync_dist_key_set)

    def training_step(self, batch, batch_idx):
        """
        Computes the training loss for one batch and logs training metrics.

        Args:
            batch: Data
                Graph batch.
            batch_idx: int
                Batch index.
            return: torch.Tensor
                Training loss.
        """
        pred = self(batch)
        target = batch.y
        metrics = self._collect_train_batch_metrics_noroll(pred, target, batch)
        self._log_metrics(metrics, prog_bar_keys=["train/loss"], batch_size=batch.num_graphs,
                          sync_dist_keys=["train/loss"])
        # on_train_epoch_end
        # 1. Steps the custom LR scheduler using the epoch loss
        # 2. Halts training early if the scheduler signals stop.
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        """
        Computes and logs validation metrics for one batch without updating model weights.

        Args:
            batch: Data
                Graph batch.
            batch_idx: int
                Batch index.
        """
        pred = self(batch)
        target = batch.y
        metrics = self._collect_val_batch_metrics_noroll(pred, target, "val")
        self._log_metrics(metrics, prog_bar_keys=["val/overall_relative_mae"], batch_size=batch.num_graphs,
                          sync_dist_keys=["val/overall_relative_mae"])

    def test_step(self, batch, batch_idx):
        """
        Computes and logs metrics for one batch during final evaluation, using the active split name as prefix.
        Syncs everything: we don't care about performance in testing phase.

        Args:
            batch: Data
                Graph batch.
            batch_idx: int
                Batch index.
        """
        pred = self(batch)
        target = batch.y
        metrics = self._collect_test_batch_metrics_noroll(pred, target, self.eval_split_name)
        self._log_metrics(metrics,
            [f"{self.eval_split_name}/loss", f"{self.eval_split_name}/overall_relative_mae"],
            batch_size=batch.num_graphs,
            sync_dist_keys=list(metrics.keys()))
        # on_test_epoch_end logs ROLLOUT metrics

    def predict_step(self, batch, batch_idx):
        """
        Returns raw model predictions for a batch during Lightning's predict loop.

        Args:
            batch: Data
                Graph batch.
            batch_idx: int
                Batch index.
        return: torch.Tensor
            Model prediction.
        """
        return self(batch)

    def on_train_epoch_end(self):
        """
        1. Logs current epoch to W&B for x-axis alignment
        2. Steps the custom LR scheduler using the epoch loss
        3. Halts training early if the scheduler signals stop.
        """
        # Log epoch to W&B for x-axis alignment (only on rank 0)
        # if self._has_wandb_logger() and self.trainer.is_global_zero:
            # self.logger.experiment.log({"epoch": self.current_epoch})

        if self.scheduler is None:
            return

        monitor_name = "val/overall_relative_mae" if self.enable_val else "train/loss"
        monitor_metric = self.trainer.callback_metrics.get(monitor_name)
        if monitor_metric is None: # additional check
            monitor_metric = self.trainer.callback_metrics["train/loss"]
        monitor_metric_value = float(monitor_metric.detach().item())
        self.scheduler.step(metric=monitor_metric_value, epoch=self.current_epoch)

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        self.log("train/lr", current_lr, prog_bar=True, logger=self._has_wandb_logger())

        if self.scheduler.stop:
            print("\n\n\n\n" + "=" * 60)
            print(f"Stopping training early because scheduler indicated it should stop at epoch {self.current_epoch}.")
            print("\n\n\n\n" + "=" * 60)
            self.trainer.should_stop = True

    def _run_rollout(self, data_loader):
        """
        FIXME: change this to take in input one trajectory, not the whole dataloader.
            This will be called by on_validation_epoch_end and be averaged over all trajectories
        Executes a full autoregressive rollout over the given loader and returns ground-truth/predicted trajectories
        plus final-step MAE, i.e. mean over all nodes POSITION ONLY of final-timestep MAE error during rollout.

        Args:
            data_loader: pyg.DataLoader
                Dataloader used for rollout evaluation.
            return: (list, list, float)
                Ground-truth rollout, predicted rollout, and final-step rollout MAE.
        """
        # Stats are already registered as submodules with buffers, so they auto-move to correct device
        true_rollout, pred_rollout = do_rollout(model=self.model, test_loader=data_loader, node_stats=self.node_stats,
                                                edge_stats=self.edge_stats, target_stats=self.target_stats,
                                                dt=self.dt, device=self.device, skip_first=0, rollout_steps=999999,
                                                normalizer=self.used_dataset_normalizer,
                                                rest_minus_nonrest_pos=self.rest_minus_nonrest_pos)
        # mean over all nodes POSITION ONLY of final-timestep MAE error during rollout
        final_roll_err_pos = torch.mean(torch.abs(true_rollout[-1].x_pos_t - pred_rollout[-1].x_pos_t)).item()
        return true_rollout, pred_rollout, final_roll_err_pos

    def _log_rollout_media(self, true_rollout, pred_rollout):
        """
        Generates and saves diagnostic plots and GIFs for a completed rollout, then uploads them to W&B if available.
        Called by on_test_epoch_end. Only runs on rank 0 for distributed training.

        Args:
            true_rollout: list
                Ground-truth rollout.
            pred_rollout: list
                Predicted rollout.
        """


        # Paths
        path_split_name = self.model_evaluation_dir / self.eval_split_name
        path_split_name.mkdir(parents=True, exist_ok=True)
        gif_path = path_split_name / f"{self.eval_split_name}_beam_comparison.gif"
        volume_png_path = path_split_name / f"{self.eval_split_name}_volume_change.png"
        rollout_error_path = path_split_name / f"{self.eval_split_name}_rollout_error_vs_timestep.png"

        # ================ Generate all rollout plots ================
        plot_volume_change(pred_rollout, true_rollout, path=volume_png_path, split_name=self.eval_split_name)
        make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout,
                                 L=1.0, W=0.1, D=0.04,
                                 gif_path=gif_path, fps=4, bbox_only_on_true=False, verbose=False,
                                 split_name=self.eval_split_name)
        plot_roll_err_pos_vs_time(true_rollout, pred_rollout, rollout_error_path, split_name=self.eval_split_name)
        plot_err_heatmap_rollout(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=path_split_name,
                                 filename_prefix=self.eval_split_name, split_name=self.eval_split_name)
        plot_err_heatmap_rollout(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=path_split_name,
                                 filename_prefix=self.eval_split_name, split_name=self.eval_split_name)

        try:
            plot_mesh_diagnostics(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=path_split_name,
                                  filename_prefix=self.eval_split_name, split_name=self.eval_split_name)
        except Exception as error:
            print(f"Error while plotting mesh diagnostics: {error}")

        if not self._has_wandb_logger():
            return

        self._log_all_wandb_media(path_split_name, gif_path, volume_png_path, rollout_error_path)

    def on_validation_epoch_end(self):
        """
        Runs a full rollout on the validation split only on rank 0 and logs the positional error.
        This metric is for reporting only and is not used by checkpoint callbacks.
        """
        if self.trainer.sanity_checking:
            return
        # This flag is for distributed training: ensure rollout is not done in parallel (would be the default ddp
        # strategy by lightning)
        if not self.trainer.is_global_zero:
            return
        if self.enable_rollout_val:
            # Returns
            _, _, final_roll_err_pos = self._run_rollout(self.validation_loader)
            self.rollout_all_step_error = final_roll_err_pos
            self.log("val/final_roll_err_pos",
                     torch.tensor(final_roll_err_pos, device=self.device),
                     prog_bar=True,
                     logger=self._has_wandb_logger(),
                     sync_dist=False,
                     rank_zero_only=True)

    def on_test_epoch_end(self):
        """
        Runs a full rollout on the active test split only on rank 0, logs the positional error,
        and optionally saves media artifacts.
        """
        if not self.trainer.is_global_zero:
            return
        # rollout_error = mean over all nodes POSITION ONLY of final-timestep MAE error during rollout
        true_rollout, pred_rollout, final_roll_err_pos = self._run_rollout(self.test_loader)
        self.rollout_all_step_error = final_roll_err_pos
        self.log(f"{self.eval_split_name}/final_roll_err_pos",
                 torch.tensor(final_roll_err_pos, device=self.device),
                 prog_bar=True,
                 logger=self._has_wandb_logger(),
                 sync_dist=False,
                 rank_zero_only=True)
        if self.log_test_media:
            self._log_rollout_media(true_rollout, pred_rollout)

    # =============================================================================================================
    #                                   BOILERPLATE WANDB CODE
    # ==============================================================================================================

    def _log_all_wandb_media(self, path_split_name, gif_path, volume_png_path, rollout_error_path):
        """
        Log all plots to W&B

        Args:
            path_split_name:
            gif_path:
            volume_png_path:
            rollout_error_path:
        """

        experiment = self.logger.experiment
        prefix = self.eval_split_name
        wandb_logs = {f"{prefix}/beam_comparison_gif": wandb.Video(str(gif_path), fps=4, format="gif"),
                      f"{prefix}/volume_change": wandb.Image(str(volume_png_path)),
                      f"{prefix}/rollout_error_vs_timestep": wandb.Image(str(rollout_error_path))}
        # wandb_logs = {"epoch": self.current_epoch}
        # Main plots
        # Per-field diagnostic plots from plot_validation_diagnostics_norollout
        for field_name, _ in TARGET_FIELDS:
            # Prediction vs target scatter
            scatter_path = path_split_name / f"{prefix}_{field_name}_prediction_vs_target.png"
            if scatter_path.exists():
                wandb_logs[f"{prefix}/{field_name}_prediction_vs_target"] = wandb.Image(str(scatter_path))
            # Edge error correlation
            corr_path = path_split_name / f"{prefix}_{field_name}_edge_error_correlation.png"
            if corr_path.exists():
                wandb_logs[f"{prefix}/{field_name}_edge_error_correlation"] = wandb.Image(str(corr_path))
            # Signed relative error GIF
            error_gif_path = path_split_name / f"{prefix}_{field_name}_signed_relative_error.gif"
            if error_gif_path.exists():
                wandb_logs[f"{prefix}/{field_name}_signed_relative_error"] = wandb.Video(
                    str(error_gif_path), fps=4, format="gif")
        if self.plot_diagnostics_advanced:
            # Mesh diagnostics (in diagnostics/ subdirectory)
            diag_dir = path_split_name / "diagnostics"
            if diag_dir.exists():
                for field_name, _ in TARGET_FIELDS:
                    # Neighbor absolute error
                    diag_path = diag_dir / f"{prefix}_{field_name}_neighbor_abs_error.png"
                    if diag_path.exists():
                        wandb_logs[f"{prefix}/diag/{field_name}_neighbor_abs_error"] = wandb.Image(str(diag_path))
                    # Error vs distance from worst node
                    diag_path = diag_dir / f"{prefix}_{field_name}_error_vs_dist_worst.png"
                    if diag_path.exists():
                        wandb_logs[f"{prefix}/diag/{field_name}_error_vs_dist_worst"] = wandb.Image(str(diag_path))
                    # Error vs curvature
                    diag_path = diag_dir / f"{prefix}_{field_name}_error_vs_curvature.png"
                    if diag_path.exists():
                        wandb_logs[f"{prefix}/diag/{field_name}_error_vs_curvature"] = wandb.Image(str(diag_path))
                    # Error vs boundary distance
                    diag_path = diag_dir / f"{prefix}_{field_name}_error_vs_boundary_dist.png"
                    if diag_path.exists():
                        wandb_logs[f"{prefix}/diag/{field_name}_error_vs_boundary_dist"] = wandb.Image(str(diag_path))
                # Combined diagnostics
                dirichlet_path = diag_dir / f"{prefix}_dirichlet_energy.png"
                if dirichlet_path.exists():
                    wandb_logs[f"{prefix}/diag/dirichlet_energy"] = wandb.Image(str(dirichlet_path))
                spectral_path = diag_dir / f"{prefix}_spectral_comparison.png"
                if spectral_path.exists():
                    wandb_logs[f"{prefix}/diag/spectral_comparison"] = wandb.Image(str(spectral_path))
        # Log all at once
        experiment.log(wandb_logs)

    def _has_wandb_logger(self):
        """
        Check if we have a valid WandbLogger attached.
        """
        return isinstance(self.logger, WandbLogger)

    def _define_wandb_metrics(self):
        """
        Define all W&B metrics with epoch as the x-axis and appropriate summaries.
        Should be called once at the start of training, only on rank 0.
        """
        if not self._has_wandb_logger() or not self.trainer.is_global_zero:
            return

        experiment = self.logger.experiment

        # Define epoch as the global step metric for x-axis experiment.define_metric("epoch")
        # All metrics use epoch as x-axis by default experiment.define_metric("*", step_metric="epoch")

        # Training metrics - track last value (default)
        experiment.define_metric("train/loss", summary="min")
        experiment.define_metric("train/supervised_loss", summary="min")
        experiment.define_metric("train/curvature_loss", summary="min")
        experiment.define_metric("train/acc_all_loss", summary="min")
        experiment.define_metric("train/stress_loss", summary="min")
        experiment.define_metric("train/acc_all_rel_mae", summary="min")
        experiment.define_metric("train/acc_x_rel_mae", summary="min")
        experiment.define_metric("train/acc_y_rel_mae", summary="min")
        experiment.define_metric("train/acc_z_rel_mae", summary="min")
        experiment.define_metric("train/stress_rel_mae", summary="min")
        experiment.define_metric("train/lr", summary="last")

        # Validation metrics
        experiment.define_metric("val/loss", summary="min") # FIXME: currently not implemented, might renitroduce
        experiment.define_metric("val/overall_relative_mae", summary="min")
        experiment.define_metric("val/acc_all_loss", summary="min")
        experiment.define_metric("val/stress_loss", summary="min")
        experiment.define_metric("val/acc_all_rel_mae", summary="min")
        experiment.define_metric("val/acc_x_rel_mae", summary="min") # FIXME: currently not implemented, might renitroduce
        experiment.define_metric("val/acc_y_rel_mae", summary="min") # FIXME: currently not implemented, might renitroduce
        experiment.define_metric("val/acc_z_rel_mae", summary="min") # FIXME: currently not implemented, might renitroduce
        experiment.define_metric("val/stress_rel_mae", summary="min")
        experiment.define_metric("val/final_roll_err_pos", summary="min")

        # Test/validation final metrics (dynamic prefix)
        for prefix in ["test", "validation"]:
            experiment.define_metric(f"{prefix}/loss", summary="min")
            experiment.define_metric(f"{prefix}/overall_relative_mae", summary="min")
            experiment.define_metric(f"{prefix}/final_roll_err_pos", summary="min")