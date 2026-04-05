"""
trainer.py
Implements a LightningModule subclass to train models.
"""

from pathlib import Path

import lightning as L
import math
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Subset
from typing import Any
from src.gnn.mesh_numerical_helpers.time_discretizer import create_time_discretizer
from src.gnn.misc_helpers.misc import compute_mean_relative_mae
from src.gnn.preprocessing.normalizer import NormalizerFactory
from src.gnn.preprocessing.stats import StatsModule
from torch_geometric.loader import DataLoader
from src.gnn.mesh_numerical_helpers.compute_volume import compute_final_volumes
from src.gnn.training.losses import loss_factory, angle_residual
from src.gnn.training.rollout import do_rollout
from src.gnn.training.scheduler import SchedulerWrapper
from src.gnn.validation.plot_rollout import in_depth_plots
from src.gnn.validation.plot_overall import overall_plot

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
    # Else
    log_test_media: bool
    node_stats: Any # Stats?
    edge_stats: Any # Stats?
    target_stats: Any # Stats?
    # Dt
    dt: float
    # Current rollout
    rollout_all_step_error: float  # current
    # Datasets (for per-trajectory rollout access)
    validation_dataset: Any
    test_dataset: Any
    # Which trajectories get in-depth plots
    selected_validation_trajectories: list[int]
    selected_test_trajectories: list[int]

    def __init__(self, model, optimizer, scheduler, loss_type, preprocess_cfg, training_id, model_root,
                 len_train_dataloader, enable_wandb_watch, train_cfg, theta_max, wandb_train_cfg_yaml):
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
            wandb_train_cfg_yaml: str
                Resolved train config as YAML (no hydra: block); empty when W&B is disabled.
            return: None
                Initializes the module.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_factory(loss_type)
        self.residual_list = set(train_cfg.residual_list)
        self.theta_max = theta_max
        self.curvature_loss_weight = train_cfg.curvature_loss_weight
        if "curvature" in self.residual_list and self.theta_max is None:
            raise ValueError("Curvature residual requested but theta_max was not provided.")

        self.training_id = training_id

        self.normalizer_method = preprocess_cfg.normalizer_method
        self.isotropic_normalizer = preprocess_cfg.isotropic_normalizer
        self.used_dataset_normalizer = NormalizerFactory.create_normalizer(self.normalizer_method,
                                                                           self.isotropic_normalizer)
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
        self.validation_dataset = None
        self.test_dataset = None
        self.eval_split_name = "test"
        self.log_test_media = train_cfg.log_test_media
        self.rollout_all_step_error = 0.0
        self.enable_wandb_watch = enable_wandb_watch

        self.len_train_dataloader = len_train_dataloader
        self.enable_val = train_cfg.enable_val
        self.profile_loss_stress_vs_acceleration = train_cfg.profile_loss_stress_vs_acceleration
        self.enable_rollout_val = train_cfg.enable_rollout_val
        self.plot_diagnostics_advanced = train_cfg.plot_diagnostics_advanced
        self.selected_validation_trajectories = train_cfg.selected_validation_trajectories
        self.selected_test_trajectories = train_cfg.selected_test_trajectories
        self.wandb_train_cfg_yaml = wandb_train_cfg_yaml

        # performance stuff
        self.rollout_num_workers = train_cfg.num_workers
        self.pin_memory = train_cfg.pin_memory

        # Features
        self.enable_edge_velocity = preprocess_cfg.enable_edge_velocity
        self.enable_node_norms = preprocess_cfg.enable_node_norms
        self.enable_stretch_node = preprocess_cfg.enable_stretch_node
        self.enable_stretch_edge = preprocess_cfg.enable_stretch_edge
        self.enable_force_edge = preprocess_cfg.enable_force_edge

        # Time discretizer from preprocess
        self.time_discretizer = create_time_discretizer(preprocess_cfg.time_discretization_method)


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

        if self.wandb_train_cfg_yaml:
            self.logger.experiment.summary["train_cfg"] = self.wandb_train_cfg_yaml

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

    def set_rollout_context(self, validation_dataset, node_stats, edge_stats, target_stats, dt):
        """
        Stores in the attributes statistics and the validation dataset needed to run autoregressive rollout evaluations
        during training. Called before training right after initialization.

        Stats are converted to StatsModule and registered as submodules so their tensors
        (registered as buffers) automatically move to the correct device with the parent module.

        Args:
            validation_dataset: InMemoryTimeStepDataset
                Validation dataset with trajectory boundary information for per-trajectory rollout.
            node_stats: Stats
                Node feature statistics.
            edge_stats: Stats
                Edge feature statistics.
            target_stats: Stats
                Target statistics.
            dt: float
                Time step size.
        """
        self.validation_dataset = validation_dataset
        # Convert Stats to StatsModule and register as submodules for automatic device handling
        self.node_stats = StatsModule.from_stats(node_stats)
        self.edge_stats = StatsModule.from_stats(edge_stats)
        self.target_stats = StatsModule.from_stats(target_stats)
        self.dt = dt

    def set_test_context(self, test_dataset, split_name):
        """
        Configures the module for a final evaluation pass, specifying the dataset, split label.

        Args:
            test_dataset: InMemoryTimeStepDataset
                Dataset with trajectory boundary information for per-trajectory rollout.
            split_name: str
                Prefix used for test logging.
        """
        self.test_dataset = test_dataset
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
        metrics = {}
        total_loss = self.loss_fn(pred, target)
        if "curvature" in self.residual_list:
            curvature_residual = angle_residual(pred, batch, self.target_stats, self.dt, self.theta_max)
            total_loss = total_loss + self.curvature_loss_weight * curvature_residual
        if "volume" in self.residual_list:
            pass
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
                     sync_dist=metric_name in sync_dist_key_set)

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
                          sync_dist_keys=list(metrics.keys()))

    def test_step(self, batch, batch_idx):
        """
        Computes and logs metrics for one batch during final evaluation, using the W&B final-eval prefix.
        Syncs everything: we don't care about performance in testing phase.

        Args:
            batch: Data
                Graph batch.
            batch_idx: int
                Batch index.
        """
        pred = self(batch)
        target = batch.y
        wb_prefix = self._wandb_final_eval_metric_prefix()
        metrics = self._collect_test_batch_metrics_noroll(pred, target, wb_prefix)
        self._log_metrics(metrics,
            [f"{wb_prefix}/loss", f"{wb_prefix}/overall_relative_mae"],
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
        if self._has_wandb_logger() and self.trainer.is_global_zero:
            self.logger.experiment.log({"epoch": self.current_epoch + 1})

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

    def _get_trajectory_loader(self, dataset, traj_idx):
        """
        Creates a DataLoader for a single trajectory by slicing the dataset.

        Args:
            dataset: InMemoryTimeStepDataset
                Dataset with trajectory_boundaries.
            traj_idx: int
                Index of the trajectory to extract.
            return: DataLoader
                DataLoader yielding graphs for the single trajectory in order.
        """
        start, end = dataset.trajectory_boundaries[traj_idx]
        subset = Subset(dataset, range(start, end))
        return DataLoader(subset, batch_size=1, shuffle=False,
                          num_workers=self.rollout_num_workers,
                          pin_memory=self.pin_memory)

    def _run_rollout(self, traj_loader):
        """
        Executes a full autoregressive rollout over a single-trajectory loader and returns ground-truth/predicted
        trajectories plus final-step MAE, i.e. mean over all nodes POSITION ONLY of final-timestep MAE error.

        Args:
            traj_loader: DataLoader
                Single-trajectory dataloader used for rollout evaluation.
            return: (list, list, float)
                Ground-truth rollout, predicted rollout, and final-step rollout MAE.
        """
        # Stats are already registered as submodules with buffers, so they auto-move to correct device
        true_rollout, pred_rollout = do_rollout(model=self.model, test_loader=traj_loader, node_stats=self.node_stats,
                                                edge_stats=self.edge_stats, target_stats=self.target_stats,
                                                dt=self.dt, device=self.device, skip_first=0, rollout_steps=999999,
                                                normalizer=self.used_dataset_normalizer,
                                                enable_edge_velocity=self.enable_edge_velocity,
                                                enable_node_norms=self.enable_node_norms,
                                                time_discretizer=self.time_discretizer,
                                                enable_stretch_node=self.enable_stretch_node,
                                                enable_force_edge=self.enable_force_edge,
                                                enable_stretch_edge=self.enable_stretch_edge)
        # mean over all nodes POSITION ONLY of final-timestep MAE error during rollout
        final_roll_err_pos = torch.mean(torch.abs(true_rollout[-1].x_pos_t - pred_rollout[-1].x_pos_t)).item()
        return true_rollout, pred_rollout, final_roll_err_pos

    def _compute_per_trajectory_noroll_metrics(self, dataset):
        """
        Runs one-step (non-autoregressive) inference on every trajectory and returns
        per-trajectory relative MAE for acceleration and stress.

        Args:
            dataset: InMemoryTimeStepDataset
                Dataset with trajectory_boundaries.
            return: (list[float], list[float])
                Per-trajectory acceleration relative MAE and stress relative MAE.
        """
        per_traj_acc_rel_mae = []
        per_traj_stress_rel_mae = []
        self.eval()
        with torch.no_grad():
            for traj_idx in range(dataset.num_trajectories):
                traj_loader = self._get_trajectory_loader(dataset, traj_idx)
                acc_maes = []
                stress_maes = []
                for batch in traj_loader:
                    batch = batch.to(self.device)
                    pred = self(batch)
                    target = batch.y
                    pred_acc = pred[:, 0:3].contiguous()
                    target_acc = target[:, 0:3].contiguous()
                    pred_stress = pred[:, 3].contiguous()
                    target_stress = target[:, 3].contiguous()
                    acc_maes.append(compute_mean_relative_mae(pred_acc, target_acc).item())
                    stress_maes.append(compute_mean_relative_mae(pred_stress, target_stress).item())
                per_traj_acc_rel_mae.append(sum(acc_maes) / len(acc_maes) if acc_maes else 0.0)
                per_traj_stress_rel_mae.append(sum(stress_maes) / len(stress_maes) if stress_maes else 0.0)
        return per_traj_acc_rel_mae, per_traj_stress_rel_mae

    def _log_plot_roll_single_traj(self, true_rollout, pred_rollout, traj_idx):
        """
        Sets up per-trajectory output directory and naming, then delegates to in_depth to generate
        all diagnostic plots for one trajectory. Does not log to W&B.

        Args:
            true_rollout: list
                Ground-truth rollout graphs.
            pred_rollout: list
                Predicted rollout graphs.
            traj_idx: int
                Index of the trajectory.
            return: dict
                Nested dict with keys "images" and "videos" from in_depth.
        """
        traj_split_name = f"traj{traj_idx}, {self.eval_split_name}"
        filename_prefix = f"traj{traj_idx}_{self.eval_split_name}"
        output_dir = self.model_evaluation_dir / self.eval_split_name / f"traj{traj_idx}"
        paths = in_depth_plots(true_rollout, pred_rollout, output_dir, traj_split_name, filename_prefix,
                               self.plot_diagnostics_advanced)
        return paths

    def on_validation_epoch_end(self):
        """
        Runs a full rollout on every trajectory of the validation split (rank 0 only),
        averages the positional error across trajectories, and logs it.
        This metric is for reporting only and is not used by checkpoint callbacks.
        """
        if self.trainer.sanity_checking:
            return
        # This flag is for distributed training: ensure rollout is not done in parallel (would be the default ddp
        # strategy by lightning)
        if not self.trainer.is_global_zero:
            return
        if not self.enable_rollout_val or self.validation_dataset is None:
            return
        all_final_roll_err_pos = []
        for traj_idx in range(self.validation_dataset.num_trajectories):
            traj_loader = self._get_trajectory_loader(self.validation_dataset, traj_idx)
            _, _, final_roll_err_pos = self._run_rollout(traj_loader)
            all_final_roll_err_pos.append(final_roll_err_pos)
        avg_final_roll_err_pos = sum(all_final_roll_err_pos) / len(all_final_roll_err_pos)
        self.rollout_all_step_error = avg_final_roll_err_pos
        self.log("val/final_roll_err_pos",
                 torch.tensor(avg_final_roll_err_pos, device=self.device),
                 prog_bar=True,
                 logger=self._has_wandb_logger(),
                 sync_dist=False,
                 rank_zero_only=True)

    def on_test_epoch_end(self):
        """
        Runs a full rollout on every trajectory of the active test/validation split (rank 0 only),
        averages the positional error, and optionally generates in-depth plots for selected
        trajectories plus overall summary plots, all logged via a single _log_all_wandb_media call.
        """
        if not self.trainer.is_global_zero:
            return

        # 1. Run rollout for every trajectory, collect results
        all_true_rollouts = {}
        all_pred_rollouts = {}
        all_final_roll_err_pos = []
        num_trajectories = self.test_dataset.num_trajectories

        # FIXME: add pbar
        for traj_idx in range(num_trajectories):
            traj_loader = self._get_trajectory_loader(self.test_dataset, traj_idx)
            true_rollout, pred_rollout, final_roll_err_pos = self._run_rollout(traj_loader)
            all_true_rollouts[traj_idx] = true_rollout
            all_pred_rollouts[traj_idx] = pred_rollout
            all_final_roll_err_pos.append(final_roll_err_pos)

        # 2. Average and log rollout error across all trajectories
        avg_final_roll_err_pos = sum(all_final_roll_err_pos) / len(all_final_roll_err_pos)
        self.rollout_all_step_error = avg_final_roll_err_pos
        wb_prefix = self._wandb_final_eval_metric_prefix()
        self.log(f"{wb_prefix}/final_roll_err_pos",
                 torch.tensor(avg_final_roll_err_pos, device=self.device),
                 prog_bar=True, logger=self._has_wandb_logger(), sync_dist=False, rank_zero_only=True)

        if not self.log_test_media:
            return

        # 3. Compute per-trajectory no-rollout metrics (one-step inference)
        # print(f"\t[{self.eval_split_name}] Computing per-trajectory no-rollout metrics...")
        per_traj_acc_rel_mae, per_traj_stress_rel_mae = self._compute_per_trajectory_noroll_metrics(self.test_dataset)
        # 4. Compute final volumes from rollout
        final_volumes_true, final_volumes_pred = compute_final_volumes(all_true_rollouts, all_pred_rollouts)

        selected = (self.selected_test_trajectories if self.eval_split_name == "test"
                    else self.selected_validation_trajectories)

        # 5. Generate overall/summary plots (across all trajectories)
        overall_output_dir = self.model_evaluation_dir / self.eval_split_name / "overall"
        overall_output_dir.mkdir(parents=True, exist_ok=True)
        overall_paths = overall_plot(all_final_roll_err_pos, per_traj_acc_rel_mae, per_traj_stress_rel_mae,
                                     final_volumes_true, final_volumes_pred, overall_output_dir,
                                     self.eval_split_name, selected)

        # 6. Generate in-depth plots for selected trajectories only
        per_traj_paths = {}
        # FIXME: add pbar
        for traj_idx in selected:
            if traj_idx >= num_trajectories:
                print(f"\tWarning: selected traj{traj_idx} out of range (only {num_trajectories} trajectories).")
                continue
            per_traj_paths[traj_idx] = self._log_plot_roll_single_traj(all_true_rollouts[traj_idx],
                                                                       all_pred_rollouts[traj_idx], traj_idx)

        # 7. Single wandb logging call
        self._log_all_wandb_media(overall_paths, per_traj_paths)

    # =============================================================================================================
    #                                   BOILERPLATE WANDB CODE
    # ==============================================================================================================

    def _wandb_final_eval_metric_prefix(self):
        """
        W&B key prefix for final evaluation scalars (trainer.test on val or test split).

        Args:
            return: str
                final_validation, final_test, or eval_split_name if unknown.
        """
        if self.eval_split_name == "validation":
            return "final_validation"
        if self.eval_split_name == "test":
            return "final_test"
        return self.eval_split_name

    def _wandb_final_eval_media_roots(self):
        """
        W&B folder roots for overall summary media vs per-trajectory in-depth media.

        Args:
            return: (str, str)
                overall_root and traj_root, e.g. final_validation and final_validation_traj.
        """


    def _log_all_wandb_media(self, overall_plot_paths, per_traj_in_depth_paths):
        """
        Single function that logs all media to W&B. Logs overall/summary plots first,
        then per-trajectory in-depth plots for selected trajectories.

        Args:
            overall_plot_paths: dict
                Nested dict {"images": {name: Path}, "videos": {name: Path}} from overall_plot.
            per_traj_in_depth_paths: dict[int, dict]
                Mapping of traj_idx to nested dicts {"images": {...}, "videos": {...}} from in_depth.
        """
        if not self._has_wandb_logger():
            return
        experiment = self.logger.experiment

        # Naming
        if self.eval_split_name == "validation":
            overall_root, traj_root = "final_validation", "final_validation_traj"
        elif self.eval_split_name == "test":
            overall_root, traj_root = "final_test", "final_test_traj"
        else:
            overall_root, traj_root = self.eval_split_name, f"{self.eval_split_name}_traj"

        wandb_logs = {}
        # 1. Log overall/summary plots
        for name, path in overall_plot_paths.get("images", {}).items():
            if Path(path).exists():
                wandb_logs[f"{overall_root}/overall/{name}"] = wandb.Image(str(path))
        for name, path in overall_plot_paths.get("videos", {}).items():
            if Path(path).exists():
                wandb_logs[f"{overall_root}/overall/{name}"] = wandb.Video(str(path), fps=4, format="gif")

        # 2. Log per-trajectory in-depth plots (one W&B folder per trajectory index)
        for traj_idx, traj_paths in per_traj_in_depth_paths.items():
            traj_prefix = f"{traj_root}/{traj_idx}"
            for name, path in traj_paths.get("images", {}).items():
                if Path(path).exists():
                    wandb_logs[f"{traj_prefix}/{name}"] = wandb.Image(str(path))
            for name, path in traj_paths.get("videos", {}).items():
                if Path(path).exists():
                    wandb_logs[f"{traj_prefix}/{name}"] = wandb.Video(str(path), fps=4, format="gif")

        experiment.log(wandb_logs, commit=False)

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
        experiment.define_metric("epoch")
        experiment.define_metric("*", step_metric="epoch")
        # Training metrics
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

        # Final evaluation metrics (trainer.test on val or test)
        for prefix in ["final_test", "final_validation"]:
            experiment.define_metric(f"{prefix}/loss", summary="min")
            experiment.define_metric(f"{prefix}/overall_relative_mae", summary="min")
            experiment.define_metric(f"{prefix}/acc_all_rel_mae", summary="min")
            experiment.define_metric(f"{prefix}/stress_rel_mae", summary="min")
            experiment.define_metric(f"{prefix}/final_roll_err_pos", summary="min")