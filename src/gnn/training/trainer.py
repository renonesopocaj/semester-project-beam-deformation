from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from typing import Any
from src.gnn.misc_helpers.misc import plot_validation_diagnostics_norollout, compute_mean_relative_mae
from src.gnn.preprocessing.normalizer import NormalizerFactory, Normalizer
from src.gnn.preprocessing.stats import Stats
from torch_geometric.loader import DataLoader
from src.gnn.training.rollout import do_rollout
from src.gnn.training.scheduler import SchedulerWrapper
from src.gnn.validation.plot_rollout import (make_beam_comparison_gif, plot_mesh_diagnostics,
                                             plot_rollout_error_vs_timestep, plot_validation_diagnostics_rollout,
                                             plot_volume_change)

def loss_factory(loss_type):
    """
    Maps a loss type string to the corresponding PyTorch loss function.

    Args:
        loss_type: str
            Name of the loss function.
        return: callable
            PyTorch loss function.
    """
    if loss_type == "mse":
        return F.mse_loss
    if loss_type == "mae":
        return F.l1_loss
    if loss_type == "huber":
        return F.huber_loss
    raise NotImplementedError("Unsupported loss function")


class LitModule(L.LightningModule):
    """
    LightningModule that handles:
    - train/validation/test batch metrics
    - custom rollout evaluation at epoch end
    - W&B media logging for final evaluation
    - manual stepping of the existing custom scheduler wrapper
    """
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
    #
    normalizer_method: str
    isotropic_normalizer: bool
    used_dataset_normalizer: NormalizerFactory
    # Features
    rest_minus_nonrest_pos: bool
    # Else
    should_log_test_media: bool
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

    def __init__(self, model, optimizer, scheduler, loss_type, preprocess_cfg, training_id, model_root):
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
        self.loss_fn = loss_factory(loss_type)

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
        self.should_log_test_media = False
        self.rollout_all_step_error = 0.0

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
        self.node_stats = node_stats
        self.edge_stats = edge_stats
        self.target_stats = target_stats
        self.dt = dt

    def set_test_context(self, test_loader, split_name, log_media):
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
        self.should_log_test_media = log_media

    def _collect_batch_metrics_norollout(self, pred, target, prefix):
        """
        Used in 1) training step 2) validation step 3) test step.
        Computes per-component and aggregate loss/MAE metrics for a single batch.
        - loss: overall loss
        - acceleration: loss & relative mae
        - stress: loss & relative mae
        - acc_x_loss: & relative mae
        - acc_y_loss: & relative mae
        - acc_z_loss: & relative mae
        - von_mises_loss: & relative mae

        Args:
            pred: torch.Tensor
                Model prediction with shape `(num_nodes, 4)`.
            target: torch.Tensor
                Ground-truth tensor with shape `(num_nodes, 4)`.
            prefix: str
                Metric namespace prefix, for example `train`, `val`, or `test`.
        """
        metrics = {f"{prefix}/loss": self.loss_fn(pred, target),
                   f"{prefix}/acc_all_loss": self.loss_fn(pred[:, 0:3].contiguous(), target[:, 0:3].contiguous()),
                   f"{prefix}/stress_loss": self.loss_fn(pred[:, 3].contiguous(), target[:, 3].contiguous()),
                   f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
                   f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(
                       pred[:, 0:3].contiguous(), target[:, 0:3].contiguous()
                   ),
                   f"{prefix}/acc_x_rel_mae": compute_mean_relative_mae(
                       pred[:, 0].contiguous(), target[:, 0].contiguous()
                   ),
                   f"{prefix}/acc_y_rel_mae": compute_mean_relative_mae(
                       pred[:, 1].contiguous(), target[:, 1].contiguous()
                   ),
                   f"{prefix}/acc_z_rel_mae": compute_mean_relative_mae(
                       pred[:, 2].contiguous(), target[:, 2].contiguous()
                   ),
                   f"{prefix}/stress_rel_mae": compute_mean_relative_mae(
                       pred[:, 3].contiguous(), target[:, 3].contiguous()
                   )}
        return metrics

    def _log_metrics(self, metrics, prog_bar_keys):
        """
        Logs a metrics dict to the lightning log, and a subset of keys to the progress bar.
        Called by training_step, validation_step, and test_step.

        Args:
            metrics: dict
                Dictionary of tensor metrics.
            prog_bar_keys: list[str]
                Metrics that should also appear in Lightning's progress bar.
        """
        prog_bar_key_set = set(prog_bar_keys)
        plain_metrics = {}
        for metric_name, metric_value in metrics.items():
            # distinction on the wanted
            if metric_name in prog_bar_key_set:
                self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=1)
            else:
                plain_metrics[metric_name] = metric_value

        if len(plain_metrics) > 0:
            self.log_dict(plain_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=1)

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
        metrics = self._collect_batch_metrics_norollout(pred, target, "train")
        self._log_metrics(metrics, ["train/loss", "train/overall_relative_mae"])
        # on_train_epoch_end 1. Steps the custom LR scheduler using the epoch loss 2. Halts training early if the
        # scheduler signals stop.
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
        # collects non rollout metrics
        metrics = self._collect_batch_metrics_norollout(pred, target, "val")
        self._log_metrics(metrics, ["val/loss", "val/overall_relative_mae"])
        # on_validation_epoch_end runs a full rollout and logs rollout

    def test_step(self, batch, batch_idx):
        """
        Computes and logs metrics for one batch during final evaluation, using the active split name as prefix.

        Args:
            batch: Data
                Graph batch.
            batch_idx: int
                Batch index.
        """
        pred = self(batch)
        target = batch.y
        # non rollout metrics
        metrics = self._collect_batch_metrics_norollout(pred, target, self.eval_split_name)
        self._log_metrics(metrics,[f"{self.eval_split_name}/loss", f"{self.eval_split_name}/relative_mae"])
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
        1. Steps the custom LR scheduler using the epoch loss
        2. Halts training early if the scheduler signals stop.
        """
        if self.scheduler is None:
            return

        epoch_loss = self.trainer.callback_metrics["train/loss"]
        epoch_loss_value = float(epoch_loss.detach().cpu())
        self.scheduler.step(metric=epoch_loss_value, epoch=self.current_epoch)

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        self.log("train/lr", current_lr, prog_bar=True, logger=True)

        if self.scheduler.stop:
            self.trainer.should_stop = True

    def _run_rollout(self, data_loader):
        """
        Executes a full autoregressive rollout over the given loader and returns ground-truth/predicted trajectories plus final-step MAE.

        Args:
            data_loader: pyg.DataLoader
                Dataloader used for rollout evaluation.
            return: (list, list, float)
                Ground-truth rollout, predicted rollout, and final-step rollout MAE.
        """
        self.target_stats.to(self.device)
        true_rollout, pred_rollout = do_rollout(model=self.model, test_loader=data_loader, node_stats=self.node_stats,
                                                edge_stats=self.edge_stats, target_stats=self.target_stats,
                                                dt=self.dt, device=self.device, skip_first=0, rollout_steps=999999,
                                                normalizer=self.used_dataset_normalizer,
                                                rest_minus_nonrest_pos=self.rest_minus_nonrest_pos)
        rollout_error = torch.mean(torch.abs(true_rollout[-1].x_pos_t - pred_rollout[-1].x_pos_t)).item()
        return true_rollout, pred_rollout, rollout_error

    def _log_rollout_media(self, true_rollout, pred_rollout):
        """
        Generates and saves diagnostic plots and GIFs for a completed rollout, then uploads them to W&B if available.
        Called by on_test_epoch_end.

        Args:
            split_name: str
                Prefix used for files and W&B keys.
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

        # ================ All rollout plots ================
        plot_volume_change(pred_rollout, true_rollout, path=volume_png_path)
        make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout,
                                 L=1.0, W=0.1, D=0.04,
                                 gif_path=gif_path, fps=4, bbox_only_on_true=False, verbose=False)
        plot_rollout_error_vs_timestep(true_rollout, pred_rollout, rollout_error_path)
        plot_validation_diagnostics_norollout(true_rollout=true_rollout, pred_rollout=pred_rollout,
                                              output_dir=path_split_name, filename_prefix=self.eval_split_name)
        plot_validation_diagnostics_rollout(true_rollout=true_rollout, pred_rollout=pred_rollout,
                                            output_dir=path_split_name, filename_prefix=self.eval_split_name)

        try:
            plot_mesh_diagnostics(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=path_split_name,
                                  filename_prefix=self.eval_split_name)
        except Exception as error:
            print(f"Error while plotting mesh diagnostics: {error}")

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {f"{self.eval_split_name}/beam_comparison_gif": wandb.Video(str(gif_path), fps=4, format="gif"),
                 f"{self.eval_split_name}/volume_change_png": wandb.Image(str(volume_png_path)),
                 # global_step is epoch
                 "trainer/global_step": self.global_step}
            )

    def on_validation_epoch_end(self):
        """
        Runs a full rollout on the validation split at the end of each validation epoch and logs the positional error.
        """
        if self.trainer.sanity_checking:
            return
        _, _, rollout_error = self._run_rollout(self.validation_loader)
        self.rollout_all_step_error = rollout_error
        self.log("val/gen_rollout_error", torch.tensor(rollout_error, device=self.device),
                 prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        """
        Runs a full rollout on the active test split, logs the positional error, and optionally saves media artifacts.
        """
        true_rollout, pred_rollout, rollout_error = self._run_rollout(self.test_loader)
        self.rollout_all_step_error = rollout_error
        self.log(f"{self.eval_split_name}/gen_rollout_error",
                 torch.tensor(rollout_error, device=self.device),
                 prog_bar=True,
                 logger=True)

        # log iff it's rank 0 (main process)
        if self.should_log_test_media and self.trainer.is_global_zero:
            self._log_rollout_media(true_rollout, pred_rollout)