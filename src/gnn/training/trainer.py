import os
import time
import torch
from src.gnn.helpers.plots.visualize import make_beam_comparison_gif, plot_volume_change
from src.gnn.helpers.eval_utilities import do_rollout
import numpy as np
from pathlib import Path
import wandb
import torch.nn.functional as F
from src.gnn.preprocessing.normalizer import NormalizerFactory

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
    cur_dir: str
    model_dir: Path
    path: Path
    loss: float
    rollout_all_step_error: float
    epoch_losses: list[float]
    train_acc_loss: list[float]
    train_stress_loss: list[float]

    def __init__(self, model, optimizer, device, loss_type, use_wandb, preprocess_cfg):
        """
        Args:
            model:
                The GNN model to be trained
            optimizer:
                optimizer for training
            device:
                Target computing device (CPU/GPU)
            loss_type: Literal["mse", "mae"]
                Loss function type ('mse' or 'mae')
            use_wandb: bool
                Whether to use Weights & Biases for logging
        """
        self.training_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = device
        # wandb
        self.run = wandb.init(mode="disabled") if not use_wandb else wandb.init()
        # model, optimizer, loss
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_factory(loss_type)
        # dataset-dependent stuff
        self.used_dataset_normalizer = NormalizerFactory.create_normalizer(preprocess_cfg.normalizer_method,
                                                                           preprocess_cfg.isotropic_normalizer)
        self.rest_minus_nonrest_pos = preprocess_cfg.rest_minus_nonrest_pos
        # Training history
        self.train_history, self.train_acc_history, self.train_stress_history = [], [], []
        # Create directory for saving models
        self.cur_dir = os.getcwd()
        self.model_dir = Path(self.cur_dir) / "saved_models" / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Initialize metrics
        self.loss = 0.0
        self.rollout_all_step_error = 0.0
        # Training history for 1 epoch
        self.epoch_losses, self.train_acc_loss, self.train_stress_loss = [], [], []

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

        # Compute component losses for logging
        pred_acc = pred[:, :3].contiguous()
        target_acc = target[:, :3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()
        stress_loss = self.loss_fn(pred_stress, target_stress)
        acc_loss = self.loss_fn(pred_acc, target_acc)
        # Store loss history
        self.train_acc_loss.append(acc_loss.cpu().detach().numpy())
        self.train_stress_loss.append(stress_loss.cpu().detach().numpy())
        self.loss = loss.cpu().detach().numpy()
        self.train_history.append(self.loss)
        self.epoch_losses.append(self.loss)

    def epoch_end(self, numepoch):
        """
        Log metrics at the end of each epoch.

        Args:
            numepoch: int
        """
        data = {"train/loss": self.loss,
                "train/acc_loss": np.mean(self.train_acc_loss),
                "train/stress_loss": np.mean(self.train_stress_loss)}
        self.run.log(data, step=numepoch)

        # Reset epoch losses
        self.epoch_losses, self.train_acc_loss, self.train_stress_loss = [], [], []

    def validation(self, validation_loader, node_stats, edge_stats, target_stats, dt, epoch):
        """
        Perform validation rollout and create visualizations.

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
        self.model.eval()
        target_stats.to(self.device)
        # Perform model rollout testing
        true_rollout, pred_rollout = do_rollout(model=self.model, test_loader=validation_loader, device=self.device,
                                                node_stats=node_stats, edge_stats=edge_stats, target_stats=target_stats,
                                                dt=dt, skip_first=0, rollout_steps=999999,
                                                normalizer=self.used_dataset_normalizer,
                                                rest_minus_nonrest_pos=self.rest_minus_nonrest_pos)
        # Compute rollout error with MAE
        mae = lambda target, pred: torch.mean(torch.abs(target - pred))
        rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t)
        # Create visualizations
        gif_path = self.model_dir / f"Epoch{epoch}_beam_comparison.gif"
        volume_png_path = self.model_dir / f"Epoch{epoch}_volume_change.png"
        plot_volume_change(pred_rollout, true_rollout, path=volume_png_path)
        make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout, L=1.0, W=0.1, D=0.04,
                                 out_gif=gif_path, fps=4)
        # Log to wandb
        data = {"val/gen_rollout_error": rollout_error,
                "val/beam_comparison_gif": wandb.Video(str(gif_path), fps=4, format="gif"),
                "val/volume_change_png": wandb.Image(str(volume_png_path))}
        self.run.log(data, step=epoch)
        # Store rollout error
        self.rollout_all_step_error = rollout_error

    def save_model(self, epochnum):
        """
        Save model checkpoint.

        Args:
            epochnum: int
                number of the epoch
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Create filename with epoch and rollout error
        if epochnum is not None:
            filename = f"Epoch_{epochnum}_GenLoss_{self.rollout_all_step_error:.10f}.pth"
        else:
            filename = f"GenLoss_{self.rollout_all_step_error:.10f}.pth"
        self.path = self.model_dir / filename
        # Save complete checkpoint
        checkpoint = {'epoch': epochnum, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.loss,
                      'rollout_all_step_error': self.rollout_all_step_error, 'training_id': self.training_id}
        torch.save(checkpoint, self.path)
        print(f"Model saved to {self.path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint to resume training.

        Args:
            checkpoint_path: str
                Path to the checkpoint file
        Returns: int: The epoch number to resume from (checkpoint epoch + 1)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore other states
        resume_epoch = checkpoint['epoch']
        self.loss = checkpoint.get('loss', 0.0)
        self.rollout_all_step_error = checkpoint.get('rollout_all_step_error', 0.0)

        # Save a notes.txt file with the original training ID
        original_training_id = checkpoint.get('training_id', 'unknown')
        notes_path = self.model_dir / "notes.txt"
        with open(notes_path, 'w') as f:
            f.write(f"Resumed from checkpoint: {checkpoint_path}\n\tOriginal training ID: {original_training_id}\n"
                    f"\tResumed at epoch: {resume_epoch}\n\tNew training ID: {self.training_id}\n")

        print(f"Checkpoint loaded successfully!\nResuming from epoch: {resume_epoch}\n\tLast loss: {self.loss:.6f}"
              f"\n\tLast rollout error: {self.rollout_all_step_error:.6f}\n\tNotes saved to: {notes_path}")

        return resume_epoch

