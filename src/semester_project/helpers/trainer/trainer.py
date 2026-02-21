class Trainer:
    """
    Trainer class for training and evaluating the MeshGraphNet model.

    Manages:
    - Training process using graph-based data
    - Model evaluation through test rollouts
    - Model checkpointing and saving
    - Experiment tracking with wandb

    Parameters:
    - model: The GNN model to be trained
    - optimizer: Optimizer for training
    - device: Target computing device (CPU/GPU)
    - loss_type: Loss function type ('mse' or 'mae')
    - use_wandb: Whether to use Weights & Biases for logging
    """

    def __init__(
            self,
            model,
            optimizer,
            device,
            loss_type: Literal["mse", "mae"] = "mse",
            use_wandb: bool = False,
    ):
        self.training_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run = wandb.init(mode="disabled") if not use_wandb else wandb.init()

        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Set loss function
        if loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "mae":
            self.loss_fn = F.l1_loss
        else:
            print(f"Unsupported loss function: {loss_type}!")
            print("Falling back to MSE loss.")
            self.loss_fn = F.mse_loss

        # Training history
        self.train_history = []
        self.train_acc_history = []
        self.train_stress_history = []

        # Create directory for saving models
        self.cur_dir = os.getcwd()
        self.model_dir = Path(self.cur_dir) / "saved_models" / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.loss = 0.0
        self.rollout_all_step_error = 0.0
        self.epoch_losses = []
        self.train_acc_loss = []
        self.train_stress_loss = []

    def train(
            self,
            graph: Data,
            node_stats: Stats,
            edge_stats: Stats,
            target_stats: Stats,
    ):
        """Perform one training step."""
        self.model.train()

        # Forward pass
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

    def epoch_end(self, epoch: int):
        """Log metrics at the end of each epoch."""
        data = {
            "train/loss": self.loss,
            "train/acc_loss": np.mean(self.train_acc_loss),
            "train/stress_loss": np.mean(self.train_stress_loss),
        }
        self.run.log(data, step=epoch)

        # Reset epoch losses
        self.epoch_losses = []
        self.train_acc_loss = []
        self.train_stress_loss = []

    def test(
            self,
            test_loader,
            node_stats: Stats,
            edge_stats: Stats,
            target_stats: Stats,
            dt: float,
            epoch: int,
    ):
        """Perform validation rollout and create visualizations."""
        self.model.eval()
        target_stats.to("cpu")

        # Perform model rollout testing
        true_rollout, pred_rollout = do_rollout(
            model=self.model,
            test_loader=test_loader,
            device=self.device,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            dt=dt,
        )

        # Compute rollout error
        rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t)

        # Create visualizations
        gif_path = self.model_dir / f"Epoch_{epoch}_beam_comparison.gif"
        volume_png_path = self.model_dir / f"Epoch_{epoch}_volume_change.png"

        plot_volume_change(pred_rollout, true_rollout, path=volume_png_path)
        make_beam_comparison_gif(
            pred_rollout=pred_rollout,
            true_rollout=true_rollout,
            L=1.0,
            W=0.1,
            D=0.04,
            out_gif=gif_path,
            fps=4,
        )

        # Log to wandb
        data = {
            "val/gen_rollout_error": rollout_error,
            "val/beam_comparison_gif": wandb.Video(str(gif_path), fps=4, format="gif"),
            "val/volume_change_png": wandb.Image(str(volume_png_path)),
        }
        self.run.log(data, step=epoch)

        # Store rollout error
        self.rollout_all_step_error = rollout_error

    def save_model(self, epoch: int = None):
        """Save model checkpoint."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with epoch and rollout error
        if epoch is not None:
            filename = f"Epoch_{epoch}_GenLoss_{self.rollout_all_step_error:.10f}.pth"
        else:
            filename = f"GenLoss_{self.rollout_all_step_error:.10f}.pth"
        self.path = self.model_dir / filename

        # Save complete checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'rollout_all_step_error': self.rollout_all_step_error,
            'training_id': self.training_id,
        }
        torch.save(checkpoint, self.path)
        print(f"Model saved to {self.path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint to resume training.

        Parameters:
        - checkpoint_path: Path to the checkpoint file

        Returns:
        - int: The epoch number to resume from (checkpoint epoch + 1)
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
            f.write(f"Resumed from checkpoint: {checkpoint_path}\n")
            f.write(f"Original training ID: {original_training_id}\n")
            f.write(f"Resumed at epoch: {resume_epoch}\n")
            f.write(f"New training ID: {self.training_id}\n")

        print(f"Checkpoint loaded successfully!")
        print(f"Resuming from epoch: {resume_epoch}")
        print(f"Last loss: {self.loss:.6f}")
        print(f"Last rollout error: {self.rollout_all_step_error:.6f}")
        print(f"Notes saved to: {notes_path}")

        return resume_epoch


print("Trainer class defined!")