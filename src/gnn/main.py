from IPython.display import Image, display
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.gnn.helpers.eval_utilities import do_rollout, mae
from src.gnn.helpers.data_utils.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.helpers.trainer import Trainer
from src.gnn.helpers.data_utils.stats import Stats
from src.gnn.helpers.models.meshgraphnet.meshgraphnet import MeshGraphNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set matplotlib to non-interactive mode for faster plotting
matplotlib.use("Agg")
plt.ioff()

# Load datasets from processed folder
DATASET_SIZE = "small"  # Use same size as preprocessing
base_dir = Path("./elastic-beam-3d")
processed_dir = base_dir / "processed" / DATASET_SIZE

train_dataset = InMemoryTimeStepDataset(sample_dir=processed_dir / "train" / "graphs")
test_dataset = InMemoryTimeStepDataset(sample_dir=processed_dir / "val" / "graphs")

# Create dataloaders
batch_size = 16
num_workers = 4
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Initialize model
node_channels = 13  # 1 (boundary) + 3 (position) + 3 (velocity) + 3 (rest position) + 3 (force)
edge_channels = 8   # 3 (relative position) + 1 (norm) + 3 (rest relative position) + 1 (norm)
num_messages = 8
latent_dim = 128

torch.manual_seed(42)
model = MeshGraphNet(
    node_channels=node_channels,
    edge_channels=edge_channels,
    latent_size=latent_dim,
    num_msgs=num_messages,
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


# Load normalization statistics from processed data
DATASET_SIZE = "small"  # Use same size as before
base_dir = Path("./elastic-beam-3d")
stats_path = base_dir / "processed" / DATASET_SIZE / "train" / "stats" / "stats.json"

print(f"Loading stats from {stats_path}...")
with open(stats_path, "r") as f:
    stats = json.load(f)
node_stats = Stats.from_dict(stats["node"])
edge_stats = Stats.from_dict(stats["edge"])
target_stats = Stats.from_dict(stats["target"])
dt = 0.08

print(f"Node features: mean shape = {node_stats.mean.shape}, std shape = {node_stats.std.shape}")
print(f"Edge features: mean shape = {edge_stats.mean.shape}, std shape = {edge_stats.std.shape}")
print(f"Target features: mean shape = {target_stats.mean.shape}, std shape = {target_stats.std.shape}")


# Training parameters
lr = 5e-5
total_epochs = 30
validation_interval = 2
loss_type = "mse"  # or "mae"
use_wandb = False  # Set to False if you don't want to use wandb

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    loss_type=loss_type,
    use_wandb=use_wandb,
)

print(f"Training ID: {trainer.training_id}")
print(f"Learning rate: {lr}")
print(f"Total epochs: {total_epochs}")
print(f"Validation interval: {validation_interval}")
print(f"Loss type: {loss_type}")
print(f"Using wandb: {use_wandb}")
print(f"Model directory: {trainer.model_dir}")


# Training loop using the Trainer class
epoch_bar = tqdm(range(total_epochs), desc="Epochs")
for epoch in epoch_bar:
    # Training phase
    for batch in tqdm(train_dataloader, desc="Batches", leave=False):
        batch = batch.to(device)

        # Train step
        trainer.train(
            graph=batch,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
        )

    # End of epoch logging
    trainer.epoch_end(epoch=epoch)

    # Validation phase
    if epoch % validation_interval == 0:
        trainer.test(
            test_loader=test_dataloader,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            dt=dt,
            epoch=epoch,
        )
        trainer.save_model(epoch=epoch)

        epoch_bar.set_postfix({
            "train_loss": f"{trainer.loss:.6f}",
            "rollout_error": f"{trainer.rollout_all_step_error:.6f}"
        })

        print(f"\nEpoch {epoch}: Train Loss = {trainer.loss:.6f}, Rollout Error = {trainer.rollout_all_step_error:.6f}")
    else:
        epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}"})

print("\nTraining complete!")
print(f"Models saved to: {trainer.model_dir}")

# Plot training history from the trainer
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Training loss
ax.plot(trainer.train_history)
ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss (MSE)")
ax.set_title("Training Loss Over Time")
ax.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print(f"Total training steps: {len(trainer.train_history)}")
print(f"Final training loss: {trainer.train_history[-1]:.6f}")
if hasattr(trainer, 'rollout_all_step_error'):
    print(f"Final rollout error: {trainer.rollout_all_step_error:.6f}")


# Display the training history plot
display(Image(filename='training_history.png'))

# ========================================================================================
#                                   Final evaluation
# ========================================================================================

model.eval()
target_stats.to("cpu")

true_rollout, pred_rollout = do_rollout(
    model=model,
    test_loader=test_dataloader,
    device=device,
    node_stats=node_stats,
    edge_stats=edge_stats,
    target_stats=target_stats,
    dt=dt,
)

# Compute final metrics
final_rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t)
print(f"\nFinal rollout MAE: {final_rollout_error:.6f}")

# Compute per-timestep errors
timestep_errors = []
for true_graph, pred_graph in zip(true_rollout, pred_rollout):
    error = mae(true_graph.x_pos_t, pred_graph.x_pos_t)
    timestep_errors.append(error.item())

plt.figure(figsize=(10, 6))
plt.plot(timestep_errors)
plt.xlabel("Timestep")
plt.ylabel("Position Error (MAE)")
plt.title("Prediction Error Over Rollout")
plt.grid(True)
plt.savefig("rollout_error_over_time.png")
plt.show()

print(f"Average per-timestep error: {np.mean(timestep_errors):.6f}")

# Display the training history plot
display(Image(filename='training_history.png'))

# Plot training history from the trainer
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Training loss
ax.plot(trainer.train_history)
ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss (MSE)")
ax.set_title("Training Loss Over Time")
ax.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print(f"Total training steps: {len(trainer.train_history)}")
print(f"Final training loss: {trainer.train_history[-1]:.6f}")
if hasattr(trainer, 'rollout_all_step_error'):
    print(f"Final rollout error: {trainer.rollout_all_step_error:.6f}")