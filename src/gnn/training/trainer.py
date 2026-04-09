"""
trainer.py

Implements a LightningModule subclass to train models.
Does the following:
"""

from pathlib import Path

import lightning as L
import math
import torch

import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Subset
from typing import Any
from src.gnn.mesh_numerical_helpers.time_discretizer import create_time_discretizer
from src.gnn.misc_helpers.misc import compute_mean_relative_mae
from src.gnn.preprocessing.normalizer import create_normalizer
from src.gnn.preprocessing.stats import StatsModule
from src.gnn.training.wandb_loggers import wandb_final_eval_metric_prefix, log_all_wandb_media, define_wandb_metrics
from torch_geometric.loader import DataLoader
from src.gnn.mesh_numerical_helpers.compute_volume import compute_final_volumes
from src.gnn.training.losses import loss_factory, angle_residual, compute_total_loss_exp_mises
from src.gnn.training.rollout import do_rollout
from src.gnn.training.scheduler import SchedulerWrapper
from src.gnn.validation.plot_rollout import in_depth_plots
from src.gnn.validation.plot_overall import overall_plot

class LitModule(L.LightningModule):
    # Model, optimizer
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: SchedulerWrapper
    loss_fn: callable
    # String info
    training_id: str
    eval_split_name: str
    # Directories
    model_dir: Path
    training_plots_dir: Path
    model_evaluation_dir: Path
    # Normalizer
    normalizer_method: str
    isotropic_normalizer: bool
    used_dataset_normalizer: NormalizerFactory
    # Exp mises
    exp_mises: bool
    stress_weight: float
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
    noise_feature_update_mode: str

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
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_factory(loss_type)
        self.exp_mises = train_cfg.exp_mises
        self.stress_weight = train_cfg.stress_weight
        self.residual_list = set(train_cfg.residual_list)
        self.theta_max = theta_max
        self.curvature_loss_weight = train_cfg.curvature_loss_weight
        if "curvature" in self.residual_list and self.theta_max is None:
            raise ValueError("Curvature residual requested but theta_max was not provided.")
        # training id
        self.training_id = training_id
        # used normalizers
        self.normalizer_method = preprocess_cfg.normalizer_method
        self.isotropic_normalizer = preprocess_cfg.isotropic_normalizer
        self.used_dataset_normalizer = create_normalizer(self.normalizer_method,
                                                                           self.isotropic_normalizer)
        self.model_dir = Path(model_root) / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_plots_dir = self.model_dir / "training"
        self.training_plots_dir.mkdir(parents=True, exist_ok=True)
        self.model_evaluation_dir = self.model_dir / "model_evaluation"
        self.model_evaluation_dir.mkdir(parents=True, exist_ok=True)
        # Init stuff then allocated in set_rollout_context
        self.node_stats = None
        self.edge_stats = None
        self.target_stats = None
        self.dt = None
        self.validation_dataset = None
        self.test_dataset = None
        # default
        self.eval_split_name = "test"
        self.log_test_media = train_cfg.log_test_media
        self.rollout_all_step_error = 0.0
        # Train
        self.wandb_train_cfg_yaml = wandb_train_cfg_yaml
        self.len_train_dataloader = len_train_dataloader
        # Others validation
        self.enable_wandb_watch = enable_wandb_watch
        self.enable_val = train_cfg.enable_val
        self.profile_loss_stress_vs_acceleration = train_cfg.profile_loss_stress_vs_acceleration
        self.enable_rollout_val = train_cfg.enable_rollout_val
        self.plot_diagnostics_advanced = train_cfg.plot_diagnostics_advanced
        self.selected_validation_trajectories = train_cfg.selected_validation_trajectories
        self.selected_test_trajectories = train_cfg.selected_test_trajectories
        self.rollout_batch_size = train_cfg.rollout_batch_size
        # performance stuff
        self.rollout_num_workers = train_cfg.num_workers
        self.pin_memory = train_cfg.pin_memory
        # Features
        self.preprocess_cfg = preprocess_cfg
        # Time discretizer from preprocess
        self.time_discretizer = create_time_discretizer(preprocess_cfg.time_discretization_method)
        self.noise_feature_update_mode = str(train_cfg.noise_feature_update_mode).lower()
        if self.noise_feature_update_mode not in {"exact", "taylor"}:
            raise ValueError(
                f"Unknown noise_feature_update_mode '{train_cfg.noise_feature_update_mode}'. "
                f"Expected 'exact' or 'taylor'."
            )


    ##############################################################################################################
    #                                            SETUP FUNCTIONS
    ##############################################################################################################

    def _has_wandb_logger(self):
        """
        Check if we have a valid WandbLogger attached.
        """
        return isinstance(self.logger, WandbLogger)

    def _current_wandb_epoch(self):
        """
        Returns the 1-based epoch number used as W&B step.
        """
        return int(self.current_epoch) + 1

    def _log_wandb_epoch_marker(self, commit):
        """
        Writes an explicit epoch marker to W&B with step=epoch.

        Args:
            commit: bool
                W&B commit flag for this epoch marker log.
        """
        if not self._has_wandb_logger() or not self.trainer.is_global_zero:
            return
        epoch_number = self._current_wandb_epoch()
        self.logger.experiment.log({"epoch": epoch_number}, step=epoch_number, commit=commit)

    def configure_optimizers(self):
        """
        Returns the optimizer to Lightning's training loop.

        Args:
            return: torch.optim.Optimizer
                Optimizer used by Lightning.
        """
        return self.optimizer

    def _unwrap_train_dataset_root(self, dataset):
        """
        Unwrap torch.utils.data.Subset (and similar) to reach the underlying map-style dataset.

        Args:
            dataset: object
                Dataset instance from the train DataLoader.
        return: object
            Innermost dataset before repeated Subset wrapping stops.
        """
        ds = dataset
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    def on_train_epoch_start(self):
        """
        Syncs InMemoryNoisedDataset noise RNG epoch so each training epoch draws different noise at the same scales.
        Uses a multiprocessing-safe counter on the dataset so forked DataLoader workers see the same epoch.
        """
        if self.trainer is None or self.trainer.train_dataloader is None:
            return
        root_ds = self._unwrap_train_dataset_root(self.trainer.train_dataloader.dataset)
        if hasattr(root_ds, "set_noise_epoch"):
            root_ds.set_noise_epoch(self.current_epoch)

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
        define_wandb_metrics(self._has_wandb_logger(), self.trainer.is_global_zero, self.logger.experiment)
        # Watch model gradients and parameters
        if self.enable_wandb_watch:
            self.logger.watch(self.model, log="gradients", log_freq=self.len_train_dataloader*5, log_graph=False)
        # Log initial epoch value
        self.logger.experiment.log({"epoch": 0}, step=0, commit=True)

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

    ##############################################################################################################
    #                                            TRAIN, VAL
    ##############################################################################################################

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

    def _collect_train_batch_metrics_noroll(self, pred, target, batch):
        """
        Collects only the training metrics that are actually needed during training.
        This is only the whole loss for now.

        Args:
            pred: torch.Tensor
                Model prediction with shape `(num_nodes, 4)`.
            target: torch.Tensor
                Ground-truth tensor with shape `(num_nodes, 4)`.
        return: dict
            Dictionary with the minimal training metrics.
        """
        metrics = {}
        if self.exp_mises:
            total_loss = compute_total_loss_exp_mises(pred, target, self.target_stats, self.loss_fn, self.stress_weight)
        else:
            total_loss = self.loss_fn(pred, target)
        if "curvature" in self.residual_list:
            curvature_residual = angle_residual(pred, batch, self.target_stats, self.dt, self.theta_max,
                                                self.time_discretizer)
            total_loss = total_loss + self.curvature_loss_weight * curvature_residual
        if "volume" in self.residual_list:
            # TODO: implement volume loss
            raise NotImplementedError("Volume loss is not implemented.")
        metrics["train/loss"] = total_loss
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
        Used by all steps: validation_step, train_step, test_step to log the final metrics collected for that epoch.

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
        if has_logger:
            self.log("epoch", float(self._current_wandb_epoch()), on_step=False, on_epoch=True,
                     prog_bar=False, logger=True, batch_size=batch_size, sync_dist=False, rank_zero_only=True)

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
        # Metrics collected: only train loss (for now)
        metrics = self._collect_train_batch_metrics_noroll(pred, target, batch)
        # Log metrics collected
        self._log_metrics(metrics, prog_bar_keys=["train/loss"], batch_size=batch.num_graphs,
                          sync_dist_keys=["train/loss"])
        # on_train_epoch_end
        # 1. Steps the custom LR scheduler using the epoch loss
        # 2. Halts training early if the scheduler signals stop.
        return metrics["train/loss"]

    def on_train_epoch_end(self):
        """
        1. Logs current epoch to W&B for x-axis alignment
        2. Steps the custom LR scheduler using the epoch loss
        3. Halts training early if the scheduler signals stop.
        """
        # Logs epoch
        self._log_wandb_epoch_marker(commit=False)
        # If there is no scheduler don't do nothing
        if self.scheduler is None:
            return
        # Collects metrics for scheduler monitoring
        monitor_name = "val/overall_relative_mae" if self.enable_val else "train/loss"
        monitor_metric = self.trainer.callback_metrics.get(monitor_name)
        if monitor_metric is None: # additional check
            monitor_metric = self.trainer.callback_metrics["train/loss"]
        monitor_metric_value = float(monitor_metric.detach().item())
        # Scheduler step
        self.scheduler.step(metric=monitor_metric_value, epoch=self.current_epoch)
        # Logs current learning rate value
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        self.log("train/lr", current_lr, on_step=False, on_epoch=True, prog_bar=True,
                 logger=self._has_wandb_logger(), sync_dist=False, rank_zero_only=True)
        # Checks if it has to stop
        if self.scheduler.stop:
            print("\n\n\n\n" + "=" * 60)
            print(f"Stopping training early because scheduler indicated it should stop at epoch {self.current_epoch}.")
            print("\n\n\n\n" + "=" * 60)
            self.trainer.should_stop = True


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

    def on_validation_epoch_end(self):
        """
        IF AND ONLY IF self.enable_rollout_val == True NOTE: right now it's not ran since it's False
        Runs a full rollout on every trajectory of the validation split (rank 0 only), averages the positional error
        across trajectories, and logs it.
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
        self._log_wandb_epoch_marker(commit=False)
        # 1. Collect all (for each traj) final rollout MAE error on position (averaged among all nodes)
        all_final_roll_err_pos = []
        for traj_idx in range(self.validation_dataset.num_trajectories):
            traj_loader = self._get_trajectory_loader(self.validation_dataset, traj_idx)
            _, _, final_roll_err_pos = self._run_rollout(traj_loader)
            all_final_roll_err_pos.append(final_roll_err_pos)
        # 2. Compute average among all trajectories
        avg_final_roll_err_pos = sum(all_final_roll_err_pos) / len(all_final_roll_err_pos)
        self.rollout_all_step_error = avg_final_roll_err_pos
        avg_final_roll_err_pos_tensor = torch.tensor(avg_final_roll_err_pos, device=self.device)
        # Rank zero only
        self.log("val/final_roll_err_pos", avg_final_roll_err_pos_tensor, prog_bar=True,
                 logger=self._has_wandb_logger(), sync_dist=False, rank_zero_only=True)

    def _get_trajectory_loader(self, dataset, traj_idx):
        """
        Creates a DataLoader for a single trajectory by slicing the dataset. Used for all rollouts operations,
        since I cannot use "batches" to execute evaluation in parallel due to the sequential nature of autoregressive
        rollout.
        Used for on_validation_epoch_end IFF self.enable_rollout_val == True, and by on_test_epoch_end by default

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
        dataloader_traj = DataLoader(subset, batch_size=self.rollout_batch_size, shuffle=False,
                                     num_workers=self.rollout_num_workers, pin_memory=self.pin_memory)
        return dataloader_traj


    #################################################################################################################
    #                                    TEST = FINAL EVALUATION (TEST AND VAL SPLITS)
    #################################################################################################################

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
        # Helper to get prefix
        wb_prefix = wandb_final_eval_metric_prefix(self.eval_split_name)
        # collects not rollout metrics overall_relative_mae, acc_all_rel_mae, stress_rel_mae
        metrics = self._collect_test_batch_metrics_noroll(pred, target, wb_prefix)
        # logs
        self._log_metrics(metrics, prog_bar_keys=[f"{wb_prefix}/overall_relative_mae"], batch_size=batch.num_graphs,
                          sync_dist_keys=list(metrics.keys()))
        # on_test_epoch_end logs ROLLOUT metrics

    def on_test_epoch_end(self):
        """
        Runs a full rollout on every trajectory of the active test/validation split (rank 0 only),
        averages the positional error, and optionally generates in-depth plots for selected
        trajectories plus overall summary plots, all logged via a single _log_all_wandb_media call.
        """
        if not self.trainer.is_global_zero:
            return
        self._log_wandb_epoch_marker(commit=False)

        # 1. Compute per-trajectory no-rollout metrics (one-step inference)
        # print(f"\t[{self.eval_split_name}] Computing per-trajectory no-rollout metrics...")
        per_traj_acc_rel_mae, per_traj_stress_rel_mae = self._compute_per_trajectory_noroll_metrics(self.test_dataset)
        # 2. Run rollout for every trajectory, collect results, and then log
        all_true_rollouts, all_pred_rollouts, all_final_roll_err_pos = {}, {}, []
        num_trajectories = self.test_dataset.num_trajectories
        # FIXME: add pbar
        for traj_idx in range(num_trajectories):
            traj_loader = self._get_trajectory_loader(self.test_dataset, traj_idx)
            true_rollout, pred_rollout, final_roll_err_pos = self._run_rollout(traj_loader)
            all_true_rollouts[traj_idx] = true_rollout
            all_pred_rollouts[traj_idx] = pred_rollout
            all_final_roll_err_pos.append(final_roll_err_pos)
        # --- Average rollout error across all trajectories
        avg_final_roll_err_pos = sum(all_final_roll_err_pos) / len(all_final_roll_err_pos)
        avg_final_roll_err_pos_tensor = torch.tensor(avg_final_roll_err_pos, device=self.device)
        self.rollout_all_step_error = avg_final_roll_err_pos
        # --- Log to wandb rollout info (position only) (media will be logged later)
        # FIXME: log also volume
        wb_prefix = wandb_final_eval_metric_prefix(self.eval_split_name)
        self.log(name=f"{wb_prefix}/final_roll_err_pos", value=avg_final_roll_err_pos_tensor,
                 prog_bar=True, logger=self._has_wandb_logger(), sync_dist=False, rank_zero_only=True)
        if not self.log_test_media: # If not logging, skip the rest
            return
        # 3. Compute final volumes from rollout
        final_volumes_true, final_volumes_pred = compute_final_volumes(all_true_rollouts, all_pred_rollouts)
        selected = (self.selected_test_trajectories if self.eval_split_name == "test"
                    else self.selected_validation_trajectories)
        # 4. Generate overall/summary plots (across all trajectories)
        overall_output_dir = self.model_evaluation_dir / self.eval_split_name / "overall"
        overall_output_dir.mkdir(parents=True, exist_ok=True)
        overall_plot_paths = overall_plot(all_final_roll_err_pos, per_traj_acc_rel_mae, per_traj_stress_rel_mae,
                                          final_volumes_true, final_volumes_pred, overall_output_dir,
                                          self.eval_split_name, selected)
        # 5. Generate in-depth plots for selected trajectories only
        per_traj_paths = {}
        for traj_idx in selected:
            if traj_idx >= num_trajectories:
                print(f"\tWarning: selected traj{traj_idx} out of range (only {num_trajectories} trajectories).")
                continue
            per_traj_paths[traj_idx] = self._log_plot_roll_single_traj(all_true_rollouts[traj_idx],
                                                                       all_pred_rollouts[traj_idx], traj_idx)
        # 6. Single wandb logging call
        log_all_wandb_media(overall_plot_paths, per_traj_paths, self.eval_split_name, self.logger.experiment,
                            self._has_wandb_logger(), self._current_wandb_epoch())

    def _collect_test_batch_metrics_noroll(self, pred, target, prefix):
        """
        Collects the evaluation metrics not rollout (input, target) during test.

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
        # FIXME: hardcoded pred
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()
        metrics = {f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
                   f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(pred_acc, target_acc),
                   f"{prefix}/stress_rel_mae": compute_mean_relative_mae(pred_stress, target_stress)}
        return metrics

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
                                                edge_stats=self.edge_stats, target_stats=self.target_stats, dt=self.dt,
                                                device=self.device, skip_first_num=0, rollout_steps=999999,
                                                normalizer=self.used_dataset_normalizer,
                                                preprocess_cfg=self.preprocess_cfg,
                                                time_discretizer=self.time_discretizer,
                                                exp_mises=self.exp_mises)
        # mean over all nodes POSITION ONLY of final-timestep MAE error during rollout
        final_roll_err_pos = torch.mean(torch.abs(true_rollout[-1].x_pos_t - pred_rollout[-1].x_pos_t)).item()
        return true_rollout, pred_rollout, final_roll_err_pos

    def _compute_per_trajectory_noroll_metrics(self, dataset):
        """
        Runs one-step (non-autoregressive) inference on every trajectory and returns per-trajectory relative MAE
        for acceleration and stress.

        Args:
            dataset: InMemoryTimeStepDataset
                Dataset with trajectory_boundaries.
        return: (list[float], list[float])
            Per-trajectory acceleration relative MAE and stress relative MAE.
        """
        per_traj_acc_rel_mae, per_traj_stress_rel_mae = [], []
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