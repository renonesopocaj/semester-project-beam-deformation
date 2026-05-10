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
from src.gnn.misc_helpers.misc import (
    absolute_mae_sum_and_count,
    compute_mean_absolute_mae,
    compute_mean_relative_mae,
    human_readable_storage,
    module_param_and_buffer_storage_bytes,
    relative_mae_sum_and_count,
)
from src.gnn.preprocessing.normalizer import create_normalizer
from src.gnn.preprocessing.stats import StatsModule
from src.gnn.preprocessing.build_features import make_scripted_feature_builders
from src.gnn.training.online_edge_augmentation import (
    OnlineRandomEdgeAugmenter,
    fold_online_seed,
    num_graphs_in_pyg_batch,
    SPLIT_TRAIN_ONLINE,
    SPLIT_VAL_DURING_FIT,
    SPLIT_VAL_FINAL,
    SPLIT_TEST_FINAL,
    SPLIT_PREDICT,
    SPLIT_NOROLL_PER_TRAJ,
    SPLIT_ROLLOUT_VAL,
    SPLIT_ROLLOUT_TEST,
)
from src.gnn.training.wandb_loggers import wandb_final_eval_metric_prefix, log_all_wandb_media, define_wandb_metrics
from torch_geometric.loader import DataLoader
from src.gnn.mesh_numerical_helpers.compute_volume import rollout_volume_mae
from src.gnn.training.losses import loss_factory, angle_residual, compute_total_loss_exp_mises, volume_residual
from src.gnn.training.rollout import do_rollout
from src.gnn.training.scheduler import SchedulerWrapper
from src.gnn.plots.plot_rollout import in_depth_plots
from src.gnn.plots.plot_overall import overall_plot
from src.gnn.plots.plot_train import plot_final_parameter_distributions

MAX_SELECTED_TIMESTEPS = 400

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
    selected_eval_rollout_limit: int
    test_traj_metadata: dict
    noise_feature_update_mode: str
    random_online: float
    train_seed: int
    val_monitor_metric: str
    _online_random_edge_augmenter: Any
    _last_fit_epoch: int

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
        supported_residuals = {"curvature", "volume"}
        unknown_residuals = sorted(self.residual_list.difference(supported_residuals))
        if len(unknown_residuals) > 0:
            raise ValueError(f"Unsupported residual(s): {unknown_residuals}. Supported residuals: {sorted(supported_residuals)}")
        self.theta_max = theta_max
        self.curvature_loss_weight = train_cfg.curvature_loss_weight
        self.volume_loss_weight = train_cfg.volume_loss_weight
        if "curvature" in self.residual_list and self.theta_max is None:
            raise ValueError("Curvature residual requested but theta_max was not provided.")
        if "volume" in self.residual_list and float(self.volume_loss_weight) < 0.0:
            raise ValueError("volume_loss_weight must be >= 0 when volume residual is enabled.")
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
        self.test_traj_metadata = {}
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
        self.val_monitor_metric = train_cfg.val_monitor_metric
        self.selected_eval_rollout_limit = min(
            int(getattr(train_cfg, "selected_eval_rollout_limit", MAX_SELECTED_TIMESTEPS)),
            MAX_SELECTED_TIMESTEPS
        )
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
        self.random_online = float(train_cfg.random_online)
        self.train_seed = int(train_cfg.seed)
        self._online_random_edge_augmenter = None
        self._last_fit_epoch = 0

    def on_fit_start(self):
        """
        Called at the beginning of fit. Sets up W&B metric definitions and model watching.
        Only executed on rank 0 for distributed training safety.

        Old one:
        if not self.trainer.is_global_zero:
            return
        nbytes = module_param_and_buffer_storage_bytes(self.model)
        print(f"LitModule.model at fit start (params+buffers): {human_readable_storage(nbytes)} ({nbytes} bytes)")
        """
        if not self._has_wandb_logger() or not self.trainer.is_global_zero:
            return

        if self.wandb_train_cfg_yaml:
            self.logger.experiment.summary["train_cfg"] = self.wandb_train_cfg_yaml

        # Define all metrics with epoch as x-axis
        define_wandb_metrics(self._has_wandb_logger(), self.trainer.is_global_zero, self.logger.experiment)
        # Keep strict epoch-based step semantics: do not emit per-batch watch logs.
        if self.enable_wandb_watch:
            print("W&B watch disabled: enforcing one W&B step per epoch.")
        # Log initial epoch value
        epoch_number = self._current_wandb_epoch()
        self.logger.experiment.log({"epoch": epoch_number}, step=epoch_number, commit=True)

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
        Returns the epoch number used as W&B step.
        During fit: the current training epoch (0-based).
        During test (final evaluation): _last_fit_epoch + split_offset, where split_offset is 0
        for "validation" and 1 for "test". This places final-eval metrics at consecutive steps
        strictly after the last training step, avoiding the monotonic-step constraint violation
        even when both final-validation and final-test evals log into the same run.
        """
        if self.trainer is not None and self.trainer.testing:
            split_offset = 1 if self.eval_split_name == "test" else 0
            return self._last_fit_epoch + split_offset
        return int(self.current_epoch)

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

    def _to_wandb_scalar(self, value):
        """
        Converts a metric value to a Python float if it is scalar-like.

        Args:
            value: object
                Metric value from trainer.callback_metrics.
        return: float or None
            Float value when conversion is possible, otherwise None.
        """
        if hasattr(value, "detach"):
            detached = value.detach()
            if detached.numel() != 1:
                return None
            return float(detached.item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _collect_callback_metrics_for_wandb(self, include_prefixes, include_exact):
        """
        Collects callback metrics to be logged to W&B at the current epoch step.

        Args:
            include_prefixes: list[str]
                Metric prefixes to include.
            include_exact: list[str]
                Exact metric names to include.
        return: dict
            Metric payload ready for experiment.log.
        """
        payload = {}
        include_prefixes_set = set(include_prefixes)
        include_exact_set = set(include_exact)
        for metric_name, metric_value in self.trainer.callback_metrics.items():
            metric_name_str = str(metric_name)
            if metric_name_str in include_exact_set:
                pass
            else:
                has_prefix = False
                for prefix in include_prefixes_set:
                    if metric_name_str.startswith(prefix):
                        has_prefix = True
                        break
                if not has_prefix:
                    continue
            scalar_value = self._to_wandb_scalar(metric_value)
            if scalar_value is None:
                continue
            payload[metric_name_str] = scalar_value
        return payload

    def _log_callback_metrics_to_wandb(self, include_prefixes, include_exact, commit):
        """
        Logs selected callback metrics to W&B at step=current_epoch.

        Args:
            include_prefixes: list[str]
                Metric prefixes to include.
            include_exact: list[str]
                Exact metric names to include.
            commit: bool
                W&B commit flag for this call.
        """
        if not self._has_wandb_logger() or not self.trainer.is_global_zero:
            return
        payload = self._collect_callback_metrics_for_wandb(include_prefixes, include_exact)
        if len(payload) == 0:
            return
        epoch_number = self._current_wandb_epoch()
        payload["epoch"] = epoch_number
        self.logger.experiment.log(payload, step=epoch_number, commit=commit)

    def _write_final_eval_to_summary(self, include_prefixes, include_exact):
        """
        Writes selected callback metrics to W&B run.summary as run-level scalars (no step / time series).

        Args:
            include_prefixes: list[str]
                Metric prefixes to include.
            include_exact: list[str]
                Exact metric names to include.
        """
        if not self._has_wandb_logger() or not self.trainer.is_global_zero:
            return
        payload = self._collect_callback_metrics_for_wandb(include_prefixes, include_exact)
        exp = self.logger.experiment
        for key, value in payload.items():
            exp.summary[key] = value

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

    def on_fit_end(self):
        """
        Called at the end of fit. Writes final parameter distribution plot, logs it to W&B when enabled,
        and unwatches the model to prevent memory leaks.
        Sets _last_fit_epoch so that test-phase W&B logging (final evaluation) uses epoch+1 as the step,
        placing final-eval artifacts cleanly after the last training point.
        """
        if self.trainer is not None and self.trainer.is_global_zero:
            out_path = self.training_plots_dir / "final_parameter_distributions.png"
            plot_final_parameter_distributions(self.model, out_path)
            if self._has_wandb_logger():
                epoch_step = self._current_wandb_epoch()
                self.logger.experiment.log(
                    {
                        "epoch": epoch_step,
                        "train/final_parameter_distributions": wandb.Image(str(out_path)),
                    },
                    step=epoch_step,
                    commit=True,
                )
        # Store last fit epoch + 1 so that on_test_epoch_end logs at a fresh step
        # strictly greater than any step committed during training.
        self._last_fit_epoch = int(self.current_epoch) + 1
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
        if self.random_online > 0.0:
            _, edge_builder = make_scripted_feature_builders(self.preprocess_cfg)
            self._online_random_edge_augmenter = OnlineRandomEdgeAugmenter(
                edge_builder, self.used_dataset_normalizer, self.edge_stats)

    def set_test_context(self, test_dataset, split_name, traj_metadata):
        """
        Configures the module for a final evaluation pass, specifying the dataset, split label.

        Args:
            test_dataset: InMemoryTimeStepDataset
                Dataset with trajectory boundary information for per-trajectory rollout.
            split_name: str
                Prefix used for test logging.
            traj_metadata: dict
                Map from trajectory index to metadata dict (fx, fy, fz, L, W, D, ood_features) for plot subtitles.
        """
        self.test_dataset = test_dataset
        self.eval_split_name = split_name
        self.test_traj_metadata = traj_metadata if traj_metadata is not None else {}

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

    def _with_online_random_edges_train(self, batch, batch_idx, split_code, train_epoch):
        """
        When random_online > 0, returns a batch clone with merged random edges and recomputed edge_attr.

        Args:
            batch: Data
                PyG batch on the training device.
            batch_idx: int
                Dataloader batch index.
            split_code: int
                Namespace for deterministic seed folding.
            train_epoch: int
                Current training epoch folded into per-graph seeds.
        return: Data
            Possibly augmented batch for the forward pass.
        """
        if self.random_online <= 0.0:
            return batch
        if self._online_random_edge_augmenter is None:
            _, edge_builder = make_scripted_feature_builders(self.preprocess_cfg)
            self._online_random_edge_augmenter = OnlineRandomEdgeAugmenter(
                edge_builder, self.used_dataset_normalizer, self.edge_stats)
        ngraphs = num_graphs_in_pyg_batch(batch)
        seeds = []
        for g in range(ngraphs):
            seeds.append(fold_online_seed(self.train_seed, split_code, train_epoch, batch_idx, g))
        return self._online_random_edge_augmenter.apply(batch, self.random_online, seeds)

    def _with_online_random_edges_eval_step(self, batch, batch_idx, split_code):
        """
        Same as training-time online augmentation but without epoch in the seed fold (deterministic eval batches).

        Args:
            batch: Data
                PyG batch on the training device.
            batch_idx: int
                Dataloader batch index.
            split_code: int
                Namespace for deterministic seed folding.
        return: Data
            Possibly augmented batch for the forward pass.
        """
        if self.random_online <= 0.0:
            return batch
        if self._online_random_edge_augmenter is None:
            _, edge_builder = make_scripted_feature_builders(self.preprocess_cfg)
            self._online_random_edge_augmenter = OnlineRandomEdgeAugmenter(
                edge_builder, self.used_dataset_normalizer, self.edge_stats)
        ngraphs = num_graphs_in_pyg_batch(batch)
        seeds = []
        for g in range(ngraphs):
            seeds.append(fold_online_seed(self.train_seed, split_code, batch_idx, g))
        return self._online_random_edge_augmenter.apply(batch, self.random_online, seeds)

    def _with_online_random_edges_noroll(self, batch, traj_idx, inner_batch_idx):
        """
        Deterministic online augmentation for per-trajectory one-step metrics during final evaluation.

        Args:
            batch: Data
                Single trajectory minibatch on device.
            traj_idx: int
                Trajectory index in the dataset.
            inner_batch_idx: int
                Index within the trajectory DataLoader.
        return: Data
            Possibly augmented batch for the forward pass.
        """
        if self.random_online <= 0.0:
            return batch
        if self._online_random_edge_augmenter is None:
            _, edge_builder = make_scripted_feature_builders(self.preprocess_cfg)
            self._online_random_edge_augmenter = OnlineRandomEdgeAugmenter(
                edge_builder, self.used_dataset_normalizer, self.edge_stats)
        ngraphs = num_graphs_in_pyg_batch(batch)
        seeds = []
        for g in range(ngraphs):
            seeds.append(fold_online_seed(self.train_seed, SPLIT_NOROLL_PER_TRAJ, traj_idx, inner_batch_idx, g))
        return self._online_random_edge_augmenter.apply(batch, self.random_online, seeds)

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
            metrics["train/curvature_residual"] = curvature_residual
            total_loss = total_loss + self.curvature_loss_weight * curvature_residual
        if "volume" in self.residual_list:
            volume_loss_value = volume_residual(pred, target, batch, self.target_stats, self.dt,
                                                self.time_discretizer, self.loss_fn)
            metrics["train/volume_residual"] = volume_loss_value
            total_loss = total_loss + self.volume_loss_weight * volume_loss_value
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
                       f"{prefix}/overall_abs_mae": compute_mean_absolute_mae(pred, target),
                       f"{prefix}/acc_all_abs_mae": compute_mean_absolute_mae(pred_acc, target_acc),
                       f"{prefix}/stress_abs_mae": compute_mean_absolute_mae(pred_stress, target_stress),
                       f"{prefix}/acc_all_loss": self.loss_fn(pred_acc, target_acc),
                       f"{prefix}/stress_loss": self.loss_fn(pred_stress, target_stress)}
        else:
            metrics = {f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
                       f"{prefix}/overall_abs_mae": compute_mean_absolute_mae(pred, target),
                       f"{prefix}/acc_all_abs_mae": compute_mean_absolute_mae(pred_acc, target_acc),
                       f"{prefix}/stress_abs_mae": compute_mean_absolute_mae(pred_stress, target_stress)}
        return metrics

    def _noroll_eval_metric_weights(self, pred, metric_names):
        """
        Lightning uses logged value times weight summed over steps, divided by sum of weights,
        so each weight must be the tensor numel the corresponding batch scalar is a mean over.

        Args:
            pred: torch.Tensor
                Model output `(num_nodes, 4)` for the batch.
            metric_names: list[str]
                Keys present in the metrics dict being logged.
        return: dict
            metric_name -> positive int weight (element count).
        """
        weights = {}
        for name in metric_names:
            if name.endswith("overall_relative_mae") or name.endswith("overall_abs_mae"):
                weights[name] = int(pred.numel())
            elif name.endswith("acc_all_rel_mae") or name.endswith("acc_all_abs_mae") or name.endswith("acc_all_loss"):
                weights[name] = int(pred[:, 0:3].numel())
            elif name.endswith("stress_rel_mae") or name.endswith("stress_abs_mae") or name.endswith("stress_loss"):
                weights[name] = int(pred[:, 3].numel())
            else:
                raise ValueError("Unknown metric for node-weighted logging: %s" % name)
        return weights

    def _train_loss_log_weight(self, batch, pred):
        """
        Weight for epoch aggregation of train/loss (scalar mean over tensor elements per batch when not exp_mises).

        Args:
            batch: Data
                PyG batch.
            pred: torch.Tensor
                Model predictions.
        return: int
            Element count for mean losses, or num_graphs when exp_mises mixes non-comparable means.
        """
        if self.exp_mises:
            return int(batch.num_graphs)
        return int(pred.numel())

    def _log_metrics(self, metrics, prog_bar_keys, metric_weights, sync_dist_keys):
        """
        Logs metrics one by one so only selected keys are synchronized across ranks.
        Used by all steps: validation_step, train_step, test_step to log the final metrics collected for that epoch.

        Args:
            metrics: dict
                Dictionary of tensor metrics.
            prog_bar_keys: list[str]
                Metrics that should appear in the progress bar.
            metric_weights: dict
                Maps each key in metrics to a positive int (denominator of that batch's logged mean).
            sync_dist_keys: list[str]
                Metric names that must be synchronized across ranks.
        """
        prog_bar_key_set = set(prog_bar_keys)
        sync_dist_key_set = set(sync_dist_keys)

        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True,
                     prog_bar=metric_name in prog_bar_key_set,
                     logger=False, batch_size=metric_weights[metric_name],
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
        batch = self._with_online_random_edges_train(batch, batch_idx, SPLIT_TRAIN_ONLINE, int(self.current_epoch))
        pred = self(batch)
        target = batch.y
        # Metrics collected: only train loss (for now)
        metrics = self._collect_train_batch_metrics_noroll(pred, target, batch)
        lw = self._train_loss_log_weight(batch, pred)
        metric_weights = {}
        for metric_name in metrics.keys():
            if metric_name == "train/volume_residual":
                metric_weights[metric_name] = int(batch.num_graphs)
            else:
                metric_weights[metric_name] = lw
        self._log_metrics(metrics, prog_bar_keys=["train/loss"], metric_weights=metric_weights,
                          sync_dist_keys=list(metrics.keys()))
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
        if self.enable_val:
            if self.val_monitor_metric == "overall_abs_mae":
                monitor_name = "val/overall_abs_mae"
            else:
                monitor_name = "val/overall_relative_mae"
        else:
            monitor_name = "train/loss"
        monitor_metric = self.trainer.callback_metrics.get(monitor_name)
        if monitor_metric is None: # additional check
            monitor_metric = self.trainer.callback_metrics["train/loss"]
        monitor_metric_value = float(monitor_metric.detach().item())
        # Scheduler step
        self.scheduler.step(metric=monitor_metric_value, epoch=self.current_epoch)
        # Logs current learning rate value
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        self.log("train/lr", current_lr, on_step=False, on_epoch=True, prog_bar=True,
                 logger=False, sync_dist=False, rank_zero_only=True)
        should_run_validation = False
        if self.enable_val and self.trainer is not None:
            val_every_n = int(getattr(self.trainer, "check_val_every_n_epoch", 1))
            if val_every_n <= 0:
                val_every_n = 1
            should_run_validation = ((int(self.current_epoch) + 1) % val_every_n) == 0
        self._log_callback_metrics_to_wandb(
            include_prefixes=["train/"], include_exact=[], commit=(not should_run_validation)
        )
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
        batch = self._with_online_random_edges_eval_step(batch, batch_idx, SPLIT_VAL_DURING_FIT)
        pred = self(batch)
        target = batch.y
        metrics = self._collect_val_batch_metrics_noroll(pred, target, "val")
        mw = self._noroll_eval_metric_weights(pred, list(metrics.keys()))
        prog_bar_keys = ["val/overall_relative_mae", "val/overall_abs_mae"]
        if self.val_monitor_metric == "overall_abs_mae":
            prog_bar_keys = ["val/overall_abs_mae", "val/overall_relative_mae"]
        self._log_metrics(metrics, prog_bar_keys=prog_bar_keys, metric_weights=mw,
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
        batch = self._with_online_random_edges_eval_step(batch, batch_idx, SPLIT_PREDICT)
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
        if self.enable_rollout_val and self.validation_dataset is not None:
            self._log_wandb_epoch_marker(commit=False)
            # 1. Collect all-step rollout position MAE for each trajectory, then pool
            # across trajectories weighted by element count so the scalar matches one global mean.
            all_rollout_pos_mae = []
            weighted_sum, weight_total = 0.0, 0
            for traj_idx in range(self.validation_dataset.num_trajectories):
                traj_loader = self._get_trajectory_loader(self.validation_dataset, traj_idx, max_timesteps=None)
                _, _, rollout_pos_mae, n_pos = self._run_rollout(traj_loader, rollout_steps=999999,
                                                                 rollout_online_tag=SPLIT_ROLLOUT_VAL, traj_idx=traj_idx)
                all_rollout_pos_mae.append(rollout_pos_mae)
                weighted_sum += rollout_pos_mae * n_pos
                weight_total += n_pos
            # 2. Element-weighted mean across trajectories
            avg_rollout_pos_mae = weighted_sum / weight_total if weight_total else 0.0
            self.rollout_all_step_error = avg_rollout_pos_mae
            avg_rollout_pos_mae_tensor = torch.tensor(avg_rollout_pos_mae, device=self.device)
            # Rank zero only
            self.log("val/rollout_pos_mae", avg_rollout_pos_mae_tensor, prog_bar=True,
                     logger=False, sync_dist=False, rank_zero_only=True)
        self._log_callback_metrics_to_wandb(include_prefixes=["train/", "val/"], include_exact=[], commit=True)

    def _get_trajectory_loader(self, dataset, traj_idx, max_timesteps):
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
            max_timesteps: int or None
                Optional maximum number of timesteps kept from the trajectory.
        return: DataLoader
            DataLoader yielding graphs for the single trajectory in order.
        """
        start, end = dataset.trajectory_boundaries[traj_idx]
        if max_timesteps is not None:
            capped_end = min(end, start + int(max_timesteps))
        else:
            capped_end = end
        subset = Subset(dataset, range(start, capped_end))
        dataloader_traj = DataLoader(subset, batch_size=self.rollout_batch_size, shuffle=False,
                                     num_workers=self.rollout_num_workers, pin_memory=self.pin_memory)
        return dataloader_traj

    def _rollout_steps_for_trajectory(self, traj_idx):
        """
        Return rollout horizon for one trajectory in final evaluation.

        Args:
            traj_idx: int
                Index of the trajectory.
        return: int
            Maximum rollout length for the trajectory.
        """
        selected = (self.selected_test_trajectories if self.eval_split_name == "test"
                    else self.selected_validation_trajectories)
        if traj_idx in selected:
            return int(self.selected_eval_rollout_limit)
        return 999999


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
        split_code = SPLIT_VAL_FINAL if self.eval_split_name == "validation" else SPLIT_TEST_FINAL
        batch = self._with_online_random_edges_eval_step(batch, batch_idx, split_code)
        pred = self(batch)
        target = batch.y
        # Helper to get prefix
        wb_prefix = wandb_final_eval_metric_prefix(self.eval_split_name)
        # collects not rollout metrics overall_relative_mae, acc_all_rel_mae, stress_rel_mae
        metrics = self._collect_test_batch_metrics_noroll(pred, target, wb_prefix)
        mw = self._noroll_eval_metric_weights(pred, list(metrics.keys()))
        prog_bar_keys_test = [f"{wb_prefix}/overall_relative_mae", f"{wb_prefix}/overall_abs_mae"]
        if self.val_monitor_metric == "overall_abs_mae":
            prog_bar_keys_test = [f"{wb_prefix}/overall_abs_mae", f"{wb_prefix}/overall_relative_mae"]
        self._log_metrics(metrics, prog_bar_keys=prog_bar_keys_test, metric_weights=mw,
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

        # 1. Compute per-trajectory no-rollout metrics (one-step inference)
        # print(f"\t[{self.eval_split_name}] Computing per-trajectory no-rollout metrics...")
        per_traj_metrics = self._compute_per_trajectory_noroll_metrics(self.test_dataset)
        per_traj_acc_rel_mae = per_traj_metrics["acc_rel_mae"]
        per_traj_stress_rel_mae = per_traj_metrics["stress_rel_mae"]
        per_traj_acc_abs_mae = per_traj_metrics["acc_abs_mae"]
        per_traj_stress_abs_mae = per_traj_metrics["stress_abs_mae"]
        per_traj_overall_abs_mae = per_traj_metrics["overall_abs_mae"]
        # 2. Run rollout for every trajectory, collect results, and then log
        all_true_rollouts, all_pred_rollouts, all_rollout_pos_mae, all_rollout_volume_mae = {}, {}, [], []
        num_trajectories = self.test_dataset.num_trajectories
        # FIXME: add pbar
        weighted_sum, weight_total = 0.0, 0
        volume_weighted_sum, volume_weight_total = 0.0, 0
        for traj_idx in range(num_trajectories):
            if self.eval_split_name == "validation":
                rollout_steps = 999999
            else:
                rollout_steps = self._rollout_steps_for_trajectory(traj_idx)
            max_timesteps = rollout_steps if rollout_steps != 999999 else None
            traj_loader = self._get_trajectory_loader(self.test_dataset, traj_idx, max_timesteps=max_timesteps)
            rollout_tag = SPLIT_ROLLOUT_VAL if self.eval_split_name == "validation" else SPLIT_ROLLOUT_TEST
            true_rollout, pred_rollout, rollout_pos_mae, n_pos = self._run_rollout(
                traj_loader, rollout_steps=rollout_steps, rollout_online_tag=rollout_tag, traj_idx=traj_idx
            )
            volume_mae, n_volume = rollout_volume_mae(true_rollout, pred_rollout)
            all_true_rollouts[traj_idx] = true_rollout
            all_pred_rollouts[traj_idx] = pred_rollout
            all_rollout_pos_mae.append(rollout_pos_mae)
            all_rollout_volume_mae.append(volume_mae)
            weighted_sum += rollout_pos_mae * n_pos
            weight_total += n_pos
            volume_weighted_sum += volume_mae * n_volume
            volume_weight_total += n_volume
        # --- Element-weighted mean of all-step position MAE across all trajectories
        avg_rollout_pos_mae = weighted_sum / weight_total if weight_total else 0.0
        avg_rollout_pos_mae_tensor = torch.tensor(avg_rollout_pos_mae, device=self.device)
        avg_rollout_volume_mae = volume_weighted_sum / volume_weight_total if volume_weight_total else 0.0
        avg_rollout_volume_mae_tensor = torch.tensor(avg_rollout_volume_mae, device=self.device)
        self.rollout_all_step_error = avg_rollout_pos_mae
        # --- Log rollout metrics (media will be logged later)
        wb_prefix = wandb_final_eval_metric_prefix(self.eval_split_name)
        self.log(name=f"{wb_prefix}/rollout_pos_mae", value=avg_rollout_pos_mae_tensor,
                 prog_bar=True, logger=False, sync_dist=False, rank_zero_only=True)
        self.log(name=f"{wb_prefix}/rollout_volume_mae", value=avg_rollout_volume_mae_tensor,
                 prog_bar=True, logger=False, sync_dist=False, rank_zero_only=True)
        if self.eval_split_name == "validation":
            self.log(name="final_validation/rollout_pos_mae_all_trajectories", value=avg_rollout_pos_mae_tensor,
                     prog_bar=False, logger=False, sync_dist=False, rank_zero_only=True)
        include_exact = []
        if self.eval_split_name == "validation":
            include_exact.append("final_validation/rollout_pos_mae_all_trajectories")
        self._write_final_eval_to_summary(
            include_prefixes=[f"{wb_prefix}/"],
            include_exact=include_exact,
        )
        if not self.log_test_media: # If not logging, skip the rest
            return
        selected = (self.selected_test_trajectories if self.eval_split_name == "test"
                    else self.selected_validation_trajectories)
        # 4. Generate overall/summary plots (across all trajectories)
        overall_output_dir = self.model_evaluation_dir / self.eval_split_name / "overall"
        overall_output_dir.mkdir(parents=True, exist_ok=True)
        overall_plot_paths = overall_plot(all_rollout_pos_mae, per_traj_acc_rel_mae, per_traj_stress_rel_mae,
                                          per_traj_acc_abs_mae, per_traj_stress_abs_mae, per_traj_overall_abs_mae,
                                          all_rollout_volume_mae, overall_output_dir,
                                          self.eval_split_name, selected)
        # 5. Generate in-depth plots for selected trajectories only
        per_traj_paths = {}
        for traj_idx in selected:
            if traj_idx >= num_trajectories:
                print(f"\tWarning: selected traj{traj_idx} out of range (only {num_trajectories} trajectories).")
                continue
            true_rollout_for_plot = all_true_rollouts[traj_idx][:self.selected_eval_rollout_limit]
            pred_rollout_for_plot = all_pred_rollouts[traj_idx][:self.selected_eval_rollout_limit]
            per_traj_paths[traj_idx] = self._log_plot_roll_single_traj(true_rollout_for_plot,
                                                                       pred_rollout_for_plot, traj_idx)
        # 6. Single wandb logging call (guard: logger may be False when use_wandb=false; do not touch .experiment)
        if self._has_wandb_logger():
            self._log_wandb_epoch_marker(commit=False)
            log_all_wandb_media(overall_plot_paths, per_traj_paths, self.eval_split_name, self.logger.experiment,
                                True, self._current_wandb_epoch())

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
                   f"{prefix}/overall_abs_mae": compute_mean_absolute_mae(pred, target),
                   f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(pred_acc, target_acc),
                   f"{prefix}/acc_all_abs_mae": compute_mean_absolute_mae(pred_acc, target_acc),
                   f"{prefix}/stress_rel_mae": compute_mean_relative_mae(pred_stress, target_stress),
                   f"{prefix}/stress_abs_mae": compute_mean_absolute_mae(pred_stress, target_stress)}
        return metrics

    def _run_rollout(self, traj_loader, rollout_steps, rollout_online_tag, traj_idx):
        """
        Executes a full autoregressive rollout over a single-trajectory loader and returns ground-truth/predicted
        trajectories plus position MAE over all rollout timesteps and nodes.

        Args:
            traj_loader: DataLoader
                Single-trajectory dataloader used for rollout evaluation.
            rollout_steps: int
                Maximum number of rollout steps to run.
            rollout_online_tag: int
                Seed namespace for deterministic online edge augmentation during rollout.
            traj_idx: int
                Trajectory index folded into rollout augmentation seeds.
            return: (list, list, float, int)
                Ground-truth rollout, predicted rollout, all-step rollout position MAE,
                and number of position elements used for cross-trajectory weighting.
        """
        # Stats are already registered as submodules with buffers, so they auto-move to correct device
        true_rollout, pred_rollout = do_rollout(model=self.model, test_loader=traj_loader, node_stats=self.node_stats,
                                                edge_stats=self.edge_stats, target_stats=self.target_stats, dt=self.dt,
                                                device=self.device, skip_first_num=0, rollout_steps=rollout_steps,
                                                normalizer=self.used_dataset_normalizer,
                                                preprocess_cfg=self.preprocess_cfg,
                                                time_discretizer=self.time_discretizer,
                                                exp_mises=self.exp_mises,
                                                random_online=self.random_online,
                                                rollout_seed_base=self.train_seed,
                                                rollout_online_tag=rollout_online_tag,
                                                traj_idx=traj_idx)
        rollout_pos_mae, n_pos = self._rollout_position_mae(true_rollout, pred_rollout)
        return true_rollout, pred_rollout, rollout_pos_mae, n_pos

    def _rollout_position_mae(self, true_rollout, pred_rollout):
        """
        Computes absolute position MAE over all timesteps, nodes, and coordinates for one trajectory.

        Args:
            true_rollout: list
                Ground-truth rollout graphs.
            pred_rollout: list
                Predicted rollout graphs.
        return: (float, int)
            Mean absolute position error and number of finite position elements used.
        """
        err_sum = 0.0
        err_count = 0
        for true_graph, pred_graph in zip(true_rollout, pred_rollout):
            diff = torch.abs(true_graph.x_pos_t - pred_graph.x_pos_t)
            finite = diff[torch.isfinite(diff)]
            err_count += int(finite.numel())
            if finite.numel() > 0:
                err_sum += float(torch.sum(finite).detach().item())
        if err_count == 0:
            return 0.0, 0
        return err_sum / err_count, err_count

    def _compute_per_trajectory_noroll_metrics(self, dataset):
        """
        Runs one-step (non-autoregressive) inference on every trajectory and returns per-trajectory
        relative and absolute MAE for acceleration, stress, and overall predictions.

        Args:
            dataset: InMemoryTimeStepDataset
                Dataset with trajectory_boundaries.
        return: dict
            Keys: acc_rel_mae, stress_rel_mae, acc_abs_mae, stress_abs_mae, overall_abs_mae.
            Each value is a list[float] with one entry per trajectory.
        """
        per_traj_acc_rel_mae, per_traj_stress_rel_mae = [], []
        per_traj_acc_abs_mae, per_traj_stress_abs_mae, per_traj_overall_abs_mae = [], [], []
        self.eval()
        with torch.no_grad():
            for traj_idx in range(dataset.num_trajectories):
                rollout_steps = self._rollout_steps_for_trajectory(traj_idx)
                max_timesteps = rollout_steps if rollout_steps != 999999 else None
                traj_loader = self._get_trajectory_loader(dataset, traj_idx, max_timesteps=max_timesteps)
                acc_rel_sum, acc_rel_cnt = 0.0, 0.0
                stress_rel_sum, stress_rel_cnt = 0.0, 0.0
                acc_abs_sum, acc_abs_cnt = 0.0, 0.0
                stress_abs_sum, stress_abs_cnt = 0.0, 0.0
                overall_abs_sum, overall_abs_cnt = 0.0, 0.0
                for inner_batch_idx, batch in enumerate(traj_loader):
                    batch = batch.to(self.device)
                    batch = self._with_online_random_edges_noroll(batch, traj_idx, inner_batch_idx)
                    pred = self(batch)
                    target = batch.y
                    pred_acc = pred[:, 0:3].contiguous()
                    target_acc = target[:, 0:3].contiguous()
                    pred_stress = pred[:, 3].contiguous()
                    target_stress = target[:, 3].contiguous()
                    s_a, c_a = relative_mae_sum_and_count(pred_acc, target_acc)
                    s_s, c_s = relative_mae_sum_and_count(pred_stress, target_stress)
                    sa_a, ca_a = absolute_mae_sum_and_count(pred_acc, target_acc)
                    sa_s, ca_s = absolute_mae_sum_and_count(pred_stress, target_stress)
                    sa_o, ca_o = absolute_mae_sum_and_count(pred, target)
                    acc_rel_sum += float(s_a.detach().item())
                    acc_rel_cnt += float(c_a.detach().item())
                    stress_rel_sum += float(s_s.detach().item())
                    stress_rel_cnt += float(c_s.detach().item())
                    acc_abs_sum += float(sa_a.detach().item())
                    acc_abs_cnt += float(ca_a.detach().item())
                    stress_abs_sum += float(sa_s.detach().item())
                    stress_abs_cnt += float(ca_s.detach().item())
                    overall_abs_sum += float(sa_o.detach().item())
                    overall_abs_cnt += float(ca_o.detach().item())
                per_traj_acc_rel_mae.append(acc_rel_sum / acc_rel_cnt if acc_rel_cnt > 0 else 0.0)
                per_traj_stress_rel_mae.append(stress_rel_sum / stress_rel_cnt if stress_rel_cnt > 0 else 0.0)
                per_traj_acc_abs_mae.append(acc_abs_sum / acc_abs_cnt if acc_abs_cnt > 0 else 0.0)
                per_traj_stress_abs_mae.append(stress_abs_sum / stress_abs_cnt if stress_abs_cnt > 0 else 0.0)
                per_traj_overall_abs_mae.append(overall_abs_sum / overall_abs_cnt if overall_abs_cnt > 0 else 0.0)
        return {
            "acc_rel_mae": per_traj_acc_rel_mae,
            "stress_rel_mae": per_traj_stress_rel_mae,
            "acc_abs_mae": per_traj_acc_abs_mae,
            "stress_abs_mae": per_traj_stress_abs_mae,
            "overall_abs_mae": per_traj_overall_abs_mae,
        }

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
        traj_meta = self.test_traj_metadata.get(traj_idx)
        paths = in_depth_plots(true_rollout, pred_rollout, output_dir, traj_split_name, filename_prefix,
                               self.plot_diagnostics_advanced, traj_meta)
        return paths