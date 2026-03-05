from IPython.display import Image, display
import json
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.gnn.preprocessing.data_utils.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.helpers.trainer import Trainer
from src.gnn.preprocessing.data_utils.stats import Stats
from src.gnn.helpers.models.meshgraphnet.meshgraphnet import MeshGraphNet
import yaml
from src.gnn.helpers.misc.yamlobject import TrainYaml

def train(cfg):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Allocating dataloader and datasets
    print("Train dataset:")
    train_dataset = InMemoryTimeStepDataset(sample_dir=cfg.train_dir, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    if cfg.enable_val:
        print("Test dataset:")
        test_dataset = InMemoryTimeStepDataset(sample_dir=cfg.val_dir, device=device)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Test samples: {len(test_dataset)}

    # Initialize model FIXME: questo spaccherebbe prenderli dalla cartella di generazione del dataset in automatico.
    # se rest_minus_nonrest true, cambia
    node_channels = 13  # 1 (boundary) + 3 (position) + 3 (velocity) + 3 (rest position) + 3 (force)
    edge_channels = 8 # 3 (relative position) + 1 (norm) + 3 (rest relative position) + 1 (norm)

    torch.manual_seed(cfg.seed)
    model = MeshGraphNet(node_channels=node_channels, edge_channels=edge_channels, latent_size=cfg.latent_dim,
                         num_msgs=cfg.num_messages)
    model = model.to(device)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    stats_path = cfg.stats_path

    print(f"Loading stats from {stats_path}...")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    node_stats = Stats.from_dict(stats["node"])
    edge_stats = Stats.from_dict(stats["edge"])
    target_stats = Stats.from_dict(stats["target"])
    # FIXME: dt should probably be loaded from dataset
    dt = 0.08

    print(f"\tNode features: mean shape = {node_stats.mean.shape}, std shape = {node_stats.std.shape}")
    print(f"\tEdge features: mean shape = {edge_stats.mean.shape}, std shape = {edge_stats.std.shape}")
    print(f"\tTarget features: mean shape = {target_stats.mean.shape}, std shape = {target_stats.std.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, loss_type=cfg.loss_type, use_wandb=use_wandb)

    print(f"Training ID: {trainer.training_id}\n\tLearning rate: {cfg.lr}\n\tTotal epochs: {cfg.total_epochs}\n\t"
          f"Validation interval: {cfg.validation_interval}\n\tLoss type: {cfg.loss_type}\n\tUsing wandb: "
          f"{cfg.use_wandb}\n\tModel directory: {trainer.model_dir}")

    # Training loop using the Trainer class
    idx = 0
    epoch_bar = tqdm(range(cfg.total_epochs), desc="Epochs")
    for epoch in epoch_bar:
        # Training phase
        for batch in tqdm(train_dataloader, desc="Batches", leave=False):
            batch = batch.to(device)
            # Train step
            trainer.train(graph=batch)

        # End of epoch logging
        trainer.epoch_end(numepoch=epoch)

        # Validation phase
        if epoch % cfg.validation_interval == 0 and cfg.enable_val:
            trainer.test(test_loader=test_dataloader, node_stats=node_stats, edge_stats=edge_stats,
                         target_stats=target_stats, dt=dt, epoch=epoch)
            # trainer.save_model(epochnum=epoch)
            epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}",
                                   "rollout_error": f"{trainer.rollout_all_step_error:.6f}"})
            print(f"\nEpoch {epoch}: Train Loss = {trainer.loss:.6f}, "
                  f"Rollout Error = {trainer.rollout_all_step_error:.6f}")
        else:
            epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}"})
            if idx == cfg.total_epochs - 1:
                trainer.save_model(epochnum=epoch)
        idx += 1

    plot_training_history(trainer)

def plot_training_history(trainer):
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

    print(f"Total training steps: {len(trainer.train_history)}")
    print(f"Final training loss: {trainer.train_history[-1]:.6f}")
    if hasattr(trainer, 'rollout_all_step_error'):
        print(f"Final rollout error: {trainer.rollout_all_step_error:.6f}")

    # Display the training history plot
    display(Image(filename='training_history.png'))

if __name__ == "__main__":
    YAML_PATH = "conf/train.yaml"
    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg = TrainYaml(cfg)
    train(cfg)