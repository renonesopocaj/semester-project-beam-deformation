from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from src.gnn.helpers.eval_utilities import do_rollout
from src.gnn.helpers.plots.visualize import make_beam_comparison_gif, plot_volume_change
from src.gnn.preprocessing.normalizer import NormalizerFactory
from src.gnn.training.train_plots import plot_validation_diagnostics

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

    def __init__(self, model, optimizer, device, loss_type, use_wandb, preprocess_cfg, scheduler, training_id, run, model_root):
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
        self.model_dir = Path(model_root) / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Accumulated training history (across all epochs)
        self.train_history, self.train_acc_history, self.train_stress_history = [], [], []
        self.train_epoch_loss_history = []
        self.val_epoch_loss_history = []
        self.val_epochs_history = []
        self.val_rollout_error_history = []
        self.train_epoch_relative_mae_history, self.val_epoch_relative_mae_history = [], []
        self.lr_history = []
        self.train_component_loss_history = {target_name: [] for target_name in self.target_names}
        self.train_component_mae_history = {target_name: [] for target_name in self.target_names}
        self.val_component_loss_history = {target_name: [] for target_name in self.target_names}
        self.val_component_mae_history = {target_name: [] for target_name in self.target_names}
        # Scalar metrics
        self.loss = 0.0
        self.rollout_all_step_error = 0.0
        self.should_stop = False
        # Per-epoch buffers (reset at end of each epoch)
        self.epoch_losses, self.train_relative_mae_losses = [], []
        self.train_acc_loss, self.train_stress_loss = [], []
        self.epoch_component_loss = {target_name: [] for target_name in self.target_names}
        self.epoch_component_mae = {target_name: [] for target_name in self.target_names}

    def _collect_batch_metrics(self, pred, target):
        """
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
            "acc_loss": self.loss_fn(pred[:, :3].contiguous(), target[:, :3].contiguous()).item(),
            "stress_loss": self.loss_fn(pred[:, 3].contiguous(), target[:, 3].contiguous()).item(),
        }
        for target_idx, target_name in enumerate(self.target_names):
            metrics[f"{target_name}_loss"] = self.loss_fn(pred[:, target_idx], target[:, target_idx]).item()
            metrics[f"{target_name}_mae"] = F.l1_loss(pred[:, target_idx], target[:, target_idx]).item()
        return metrics

    def train(self, graph):
        """
        Perform one training step.
        Args:
            graph: Data
                graph object
        """
        self.model.train()

        # Forward pass: the input is the whole graph
        pred = self.model(graph)
        target = graph.y.to(self.device)
        # Compute loss
        loss = self.loss_fn(pred, target)
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = self._collect_batch_metrics(pred, target)
        # Store loss history
        self.train_acc_loss.append(metrics["acc_loss"])
        self.train_stress_loss.append(metrics["stress_loss"])
        self.loss = metrics["loss"]
        self.train_history.append(self.loss)
        self.epoch_losses.append(self.loss)
        self.train_relative_mae_losses.append(metrics["relative_mae"])
        for target_name in self.target_names:
            self.epoch_component_loss[target_name].append(metrics[f"{target_name}_loss"])
            self.epoch_component_mae[target_name].append(metrics[f"{target_name}_mae"])

    def epoch_end(self, numepoch):
        """
        Log metrics at the end of each epoch.

        Args:
            numepoch: int
        """
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
        if self.scheduler is not None:
            self.scheduler.step(metric=epoch_loss, epoch=numepoch)
            current_lr = self.scheduler.get_last_lr()[0]
            self.should_stop = self.scheduler.stop
        else:
            current_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_history.append(current_lr)
        data["train/lr"] = current_lr

        self.run.log(data, step=numepoch)

        # Reset epoch losses
        self.epoch_losses, self.train_relative_mae_losses = [], []
        self.train_acc_loss, self.train_stress_loss = [], []
        self.epoch_component_loss = {target_name: [] for target_name in self.target_names}
        self.epoch_component_mae = {target_name: [] for target_name in self.target_names}

    def _compute_validation_epoch_metrics(self, validation_loader):
        """
        Args:
            validation_loader: pyg.DataLoader
                Validation dataloader.
            return: dict
                Averaged loss and MAE metrics on the validation set.
        """
        aggregated_metrics = {"loss": [], "relative_mae": [], "acc_loss": [], "stress_loss": []}
        for target_name in self.target_names:
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
        Perform validation metrics tracking during training.

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
        print(f"Validation laoder\n\t{validation_loader}\n\t{type(validation_loader)}\n\t{len(validation_loader)}")
        self.model.eval()
        target_stats.to(self.device)
        validation_metrics = self._compute_validation_epoch_metrics(validation_loader)
        self.val_epochs_history.append(epoch)
        self.val_epoch_loss_history.append(validation_metrics["loss"])
        self.val_epoch_relative_mae_history.append(validation_metrics["relative_mae"])
        for target_name in self.target_names:
            self.val_component_loss_history[target_name].append(validation_metrics[f"{target_name}_loss"])
            self.val_component_mae_history[target_name].append(validation_metrics[f"{target_name}_mae"])
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
        self.model.eval()
        target_stats.to(self.device)
        test_metrics = self._compute_validation_epoch_metrics(test_dataloader)
        true_rollout, pred_rollout, rollout_error = self._run_rollout(test_dataloader, node_stats, edge_stats,
                                                                      target_stats, dt)

        gif_path = self.model_dir / f"{split_name}_beam_comparison.gif"
        volume_png_path = self.model_dir / f"{split_name}_volume_change.png"
        plot_volume_change(pred_rollout, true_rollout, path=volume_png_path)
        make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout, L=1.0, W=0.1, D=0.04,
                                 out_gif=gif_path, fps=4, bbox_only_on_true=False, verbose=False)
        plot_validation_diagnostics(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=self.model_dir,
                                    output_prefix=split_name)

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


