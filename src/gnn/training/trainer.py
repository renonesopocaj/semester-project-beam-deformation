from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from src.gnn.training.rollout import do_rollout
from src.gnn.validation.plot_rollout import (make_beam_comparison_gif, plot_volume_change,
                                             plot_validation_diagnostics_rollout, plot_rollout_error_vs_timestep,
                                             plot_mesh_diagnostics)
from src.gnn.preprocessing.normalizer import NormalizerFactory
from src.gnn.misc_helpers.misc import plot_validation_diagnostics_norollout

def loss_factory(loss_type):
    """
    Factory pattern, returns the proper function
    """
    # Set loss function
    if loss_type == "mse":
        return F.mse_loss
    elif loss_type == "mae":
        return F.l1_loss
    elif loss_type == "huber":
        return F.huber_loss
    else:
        raise NotImplementedError("Unsupported loss function")

class Trainer:
    """
    Trainer class for training and evaluating the MeshGraphNet model.

    Manages:
    - Training process using graph-based data
    - Model evaluation through test rollouts
    - Model checkpointing and saving
    - Experiment tracking with wandb
    """
    training_id: str
    run: wandb.sdk.wandb_run.Run
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    loss_fn: callable
    train_history: list[float]
    train_acc_history: list[float]
    train_stress_history: list[float]
    model_dir: Path
    loss: float
    rollout_all_step_error: float
    epoch_losses: list[float]
    train_acc_loss: list[float]
    train_stress_loss: list[float]

    def __init__(self, model, optimizer, device, loss_type, use_wandb, preprocess_cfg, scheduler, training_id, run,
                 model_root):
        """
        Pure constructor: receives all identity and external resources from the
        caller.  Does NOT generate training_id, call wandb.init(), or decide
        its own directory.

        Args:
            model: torch.nn.Module
                The GNN model to be trained.
            optimizer: torch.optim.Optimizer
                Optimizer for training.
            device: torch.device
                Target computing device (CPU/GPU).
            loss_type: str
                Loss function type ('mse', 'mae', or 'huber').
            use_wandb: bool
                Whether W&B logging is enabled.
            preprocess_cfg: object
                Preprocessing configuration with attribute-style access.
            scheduler: SchedulerWrapper or None
                LR scheduler wrapper; None disables scheduling entirely.
            training_id: str
                Stable identifier for this training session.
            run: wandb.sdk.wandb_run.Run
                Already-initialised W&B run (or disabled run).
            model_root: str or pathlib.Path
                Root directory that contains all saved-training folders.
        """
        self.training_id = training_id
        self.device = device
        self.use_wandb = use_wandb
        self.run = run
        self.model = model
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.loss_fn = loss_factory(loss_type)
        self.scheduler = scheduler
        self.normalizer_method = preprocess_cfg.normalizer_method
        self.isotropic_normalizer = preprocess_cfg.isotropic_normalizer
        self.used_dataset_normalizer = NormalizerFactory.create_normalizer(self.normalizer_method,
                                                                           self.isotropic_normalizer)
        self.rest_minus_nonrest_pos = preprocess_cfg.displacement
        self.target_names = ["acc_x", "acc_y", "acc_z", "von_mises"]

        # Allocate all directories
        self.model_dir = Path(model_root) / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_plots_dir = self.model_dir / "training"
        self.training_plots_dir.mkdir(parents=True, exist_ok=True)
        self.model_evaluation_dir = self.model_dir / "model_evaluation"
        self.model_evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated training history (across all epochs)
        self.train_history, self.train_acc_history, self.train_stress_history = [], [], []
        (self.train_epoch_loss_history, self.val_epoch_loss_history, self.val_epochs_history,
         self.val_rollout_error_history, self.train_epoch_relative_mae_history, self.val_epoch_relative_mae_history,
         self.lr_history) = [], [], [], [], [], [], []
        self.train_component_loss_history = {target_name: [] for target_name in self.target_names}
        self.train_component_mae_history = {target_name: [] for target_name in self.target_names}
        self.val_component_loss_history = {target_name: [] for target_name in self.target_names}
        self.val_component_mae_history = {target_name: [] for target_name in self.target_names}
        # Scalar metrics
        self.loss = 0.0
        self.rollout_all_step_error = 0.0
        self.should_stop = False
        # Per-epoch buffers (reset at end of each epoch)
        self.epoch_losses, self.train_relative_mae_losses, self.train_acc_loss, self.train_stress_loss = [], [],[], []
        self.epoch_component_loss = {target_name: [] for target_name in self.target_names}
        self.epoch_component_mae = {target_name: [] for target_name in self.target_names}

    def _collect_batch_metrics(self, pred, target):
        """
        Collects metrics on a train/validation set, about pred vs target, so it's actually agnostic and is not as
        validation necessarily but in general as pred vs target.
        No rollout.

        Args:
            pred: torch.Tensor
                Model prediction with shape `(num_nodes, 4)`.
            target: torch.Tensor
                Ground-truth tensor with shape `(num_nodes, 4)`.
            return: dict
                Loss and MAE metrics for the full target and each scalar component.
        """
        denominator = torch.clamp(torch.abs(target), min=1e-12)
        relative_mae = torch.mean(torch.abs(pred - target) / denominator).item()
        metrics = {
            "loss": self.loss_fn(pred, target).item(),
            "relative_mae": relative_mae,
            "acc_loss": self.loss_fn(pred[:, 0:3].contiguous(), target[:, 0:3].contiguous()).item(),
            "stress_loss": self.loss_fn(pred[:, 3].contiguous(), target[:, 3].contiguous()).item(),
        }
        for target_idx, target_name in enumerate(self.target_names):
            metrics[f"{target_name}_loss"] = self.loss_fn(pred[:, target_idx], target[:, target_idx]).item()
            metrics[f"{target_name}_mae"] = F.l1_loss(pred[:, target_idx], target[:, target_idx]).item()
        return metrics

    def train(self, batch):
        """
        Perform one training step.
        Args:
            batch: Data
                graph object
        """
        self.model.train()

        # Forward pass: the input is the whole graph
        pred = self.model(batch)
        target = batch.y.to(self.device)
        # Compute loss
        loss = self.loss_fn(pred, target)
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute alternative metrics (also different from loss) on training set, i.e., pred vs targets, such as loss
        # and relative mae (no rollout). NOTE: it's not validation yet
        batch_metrics = self._collect_batch_metrics(pred, target)
        # Store histories of no-rollout loss and metrics (eg relative mae)
        self.train_acc_loss.append(batch_metrics["acc_loss"])
        self.train_stress_loss.append(batch_metrics["stress_loss"])
        self.loss = batch_metrics["loss"]
        self.train_history.append(self.loss)
        self.epoch_losses.append(self.loss)
        self.train_relative_mae_losses.append(batch_metrics["relative_mae"])
        for target_name in self.target_names:
            self.epoch_component_loss[target_name].append(batch_metrics[f"{target_name}_loss"])
            self.epoch_component_mae[target_name].append(batch_metrics[f"{target_name}_mae"])

    def epoch_end_log_metrics_scheduler(self, numepoch):
        """
        Log non-rollout metrics such as realtive mae and loss at the end of each epoch.

        Args:
            numepoch: int
        """
        # Compute mean from loss of batches inside the given epoch, to return as final value the "mean of that metric
        # in that epoch"
        epoch_loss = float(np.mean(self.epoch_losses))
        epoch_relative_mae = float(np.mean(self.train_relative_mae_losses))
        self.train_epoch_loss_history.append(epoch_loss)
        self.train_epoch_relative_mae_history.append(epoch_relative_mae)
        self.train_acc_history.append(float(np.mean(self.train_acc_loss)))
        self.train_stress_history.append(float(np.mean(self.train_stress_loss)))

        data = {"train/loss": epoch_loss,
                "train/relative_mae": epoch_relative_mae,
                "train/acc_loss": self.train_acc_history[-1],
                "train/stress_loss": self.train_stress_history[-1]}

        for target_name in self.target_names:
            component_loss = float(np.mean(self.epoch_component_loss[target_name]))
            component_mae = float(np.mean(self.epoch_component_mae[target_name]))
            self.train_component_loss_history[target_name].append(component_loss)
            self.train_component_mae_history[target_name].append(component_mae)
            data[f"train/{target_name}_loss"] = component_loss
            data[f"train/{target_name}_mae"] = component_mae

        data = self._update_scheduler(epoch_loss, numepoch, data)

        # Log everything in wandb
        self.run.log(data, step=numepoch)

        # Reset epoch losses
        self.epoch_losses, self.train_relative_mae_losses, self.train_acc_loss, self.train_stress_loss = [], [], [], []
        self.epoch_component_loss = {target_name: [] for target_name in self.target_names}
        self.epoch_component_mae = {target_name: [] for target_name in self.target_names}

    def _update_scheduler(self, epoch_loss, numepoch, data):
        # Scheduler updates
        self.scheduler.step(metric=epoch_loss, epoch=numepoch)
        current_lr = self.scheduler.get_last_lr()[0]
        self.should_stop = self.scheduler.stop
        self.lr_history.append(current_lr)
        data["train/lr"] = current_lr
        return data

    def _validation_metrics_norollout(self, validation_loader):
        """
        Compute all non-rollout validation matrics (loss, relative mae, etc)

        Args:
            validation_loader: pyg.DataLoader
                Validation dataloader.
            return: dict
                Averaged loss and MAE metrics on the validation set.
        """
        aggregated_metrics = {"loss": [], "relative_mae": [], "acc_loss": [], "stress_loss": []}
        for target_name in self.target_names: # example of self.target_names: ["acc_x", "acc_y", "acc_z", "von_mises"]
            aggregated_metrics[f"{target_name}_loss"] = []
            aggregated_metrics[f"{target_name}_mae"] = []

        with torch.no_grad():
            for graph in validation_loader:
                graph = graph.to(self.device)
                pred = self.model(graph)
                target = graph.y.to(self.device)
                batch_metrics = self._collect_batch_metrics(pred, target)
                for metric_name, metric_value in batch_metrics.items():
                    aggregated_metrics[metric_name].append(metric_value)

        return {metric_name: float(np.mean(metric_values)) for metric_name, metric_values in aggregated_metrics.items()}

    def _run_rollout(self, data_loader, node_stats, edge_stats, target_stats, dt):
        """
        Args:
            data_loader: pyg.DataLoader
                Dataloader used for rollout evaluation.
            node_stats: Stats
                stats object of node features
            edge_stats: Stats
                stats object of edge features
            target_stats: Stats
                stats object of target
            dt: float
                time interval
            return: tuple
                Ground-truth rollout, predicted rollout, and final-step rollout MAE.
        """
        true_rollout, pred_rollout = do_rollout(model=self.model, test_loader=data_loader, device=self.device,
                                                node_stats=node_stats, edge_stats=edge_stats, target_stats=target_stats,
                                                dt=dt, skip_first=0, rollout_steps=999999,
                                                normalizer=self.used_dataset_normalizer,
                                                rest_minus_nonrest_pos=self.rest_minus_nonrest_pos)
        mae = lambda target, pred: torch.mean(torch.abs(target - pred))
        rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t).item()
        return true_rollout, pred_rollout, rollout_error

    def validation(self, validation_loader, node_stats, edge_stats, target_stats, dt, epoch):
        """
        Perform validation metrics tracking during training. Perform also rollout.

        Args:
            validation_loader: pyg.DataLoader
                dataloader of validation dataset
            node_stats: Stats
                stats object of node features
            edge_stats: Stats
                stats object of edge features
            target_stats: Stats
                stats object of target
            dt: float
                time interval
            epoch: int
                number of peoch
        """
        # Set model
        self.model.eval()
        target_stats.to(self.device)
        # Compute non rollout validation metrics on validation set
        validation_metrics = self._validation_metrics_norollout(validation_loader)
        self.val_epochs_history.append(epoch)
        self.val_epoch_loss_history.append(validation_metrics["loss"])
        self.val_epoch_relative_mae_history.append(validation_metrics["relative_mae"])
        for target_name in self.target_names:
            self.val_component_loss_history[target_name].append(validation_metrics[f"{target_name}_loss"])
            self.val_component_mae_history[target_name].append(validation_metrics[f"{target_name}_mae"])
        # Compute rollout validation metrics
        _, _, rollout_error = self._run_rollout(validation_loader, node_stats, edge_stats, target_stats, dt)
        self.val_rollout_error_history.append(rollout_error)
        # Log to wandb
        data = {"val/loss": validation_metrics["loss"],
                "val/relative_mae": validation_metrics["relative_mae"],
                "val/gen_rollout_error": rollout_error,
                "val/acc_loss": validation_metrics["acc_loss"],
                "val/stress_loss": validation_metrics["stress_loss"]}
        for target_name in self.target_names:
            data[f"val/{target_name}_loss"] = validation_metrics[f"{target_name}_loss"]
            data[f"val/{target_name}_mae"] = validation_metrics[f"{target_name}_mae"]
        self.run.log(data, step=epoch)
        # Store rollout error
        self.rollout_all_step_error = rollout_error

    def final_test(self, test_dataloader, node_stats, edge_stats, target_stats, dt, split_name):
        """
        Args:
            test_dataloader: pyg.DataLoader
                Dataloader used for the final evaluation split.
            node_stats: Stats
                stats object of node features
            edge_stats: Stats
                stats object of edge features
            target_stats: Stats
                stats object of target
            dt: float
                time interval
            split_name: str
                Name used for files and logging.
            return: float
                Final rollout MAE on the evaluation split.
        """
        # Evaluate model (no plots, only computing rollout)
        self.model.eval()
        target_stats.to(self.device)
        # Compute pred vs target (no rollout) metrics (loss, relative mae)
        test_metrics = self._validation_metrics_norollout(test_dataloader)
        # Compute rollout on given dataloader
        true_rollout, pred_rollout, rollout_error = self._run_rollout(test_dataloader, node_stats, edge_stats,
                                                                      target_stats, dt)
        # make sure exists
        path_split_name = self.model_evaluation_dir / split_name
        path_split_name.mkdir(parents=True, exist_ok = True)
        # Create plots of rollout
        gif_path = path_split_name / f"{split_name}_beam_comparison.gif"
        volume_png_path = path_split_name / f"{split_name}_volume_change.png"
        print("\tPlotting volume change...")
        plot_volume_change(pred_rollout, true_rollout, path=volume_png_path)
        print("\tPlotting comparison gif...")
        make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout,
                                 L=1.0, W=0.1, D=0.04, # FIXME: hardcoded
                                 gif_path=gif_path,
                                 fps=4, bbox_only_on_true=False, verbose=False)
        print("\tPlotting rollout error vs timestep...")
        rollout_error_path = path_split_name / f"{split_name}_rollout_error_vs_timestep.png"
        plot_rollout_error_vs_timestep(true_rollout, pred_rollout, rollout_error_path)
        # Plot non rollout values
        print("\tPlotting non rollout values...")
        plot_validation_diagnostics_norollout(true_rollout=true_rollout, pred_rollout=pred_rollout,
                                              output_dir= path_split_name,
                                              filename_prefix=split_name)
        print("\tPlotting rollout values...")
        plot_validation_diagnostics_rollout(true_rollout=true_rollout, pred_rollout=pred_rollout,
                                              output_dir= path_split_name,
                                              filename_prefix=split_name)
        print("\tPlotting mesh diagnostics...")
        try:
            plot_mesh_diagnostics(true_rollout=true_rollout, pred_rollout=pred_rollout,
                                  output_dir=path_split_name, filename_prefix=split_name)
        except Exception as e:
            print(f"Error while plotting mesh diagnostics: {e}")

        # Save
        data = {f"{split_name}/loss": test_metrics["loss"],
                f"{split_name}/relative_mae": test_metrics["relative_mae"],
                f"{split_name}/gen_rollout_error": rollout_error,
                f"{split_name}/acc_loss": test_metrics["acc_loss"],
                f"{split_name}/stress_loss": test_metrics["stress_loss"],
                f"{split_name}/beam_comparison_gif": wandb.Video(str(gif_path), fps=4, format="gif"),
                f"{split_name}/volume_change_png": wandb.Image(str(volume_png_path))}
        for target_name in self.target_names:
            data[f"{split_name}/{target_name}_loss"] = test_metrics[f"{target_name}_loss"]
            data[f"{split_name}/{target_name}_mae"] = test_metrics[f"{target_name}_mae"]
        self.run.log(data)
        self.rollout_all_step_error = rollout_error
        return rollout_error
