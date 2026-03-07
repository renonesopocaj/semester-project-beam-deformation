import os
import yaml
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.training.trainer import Trainer
from src.gnn.preprocessing.stats import load_stats
from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
import hydra
from src.gnn.helpers.misc.yamlobject import TrainYaml, PreprocessYaml
from src.gnn.training.train_plots import plot_training_history
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
plt.ioff()

def train(cfg, preprocess_cfg, device):
    # Allocating dataloader and datasets
    print("Train dataset:")
    train_dataset = InMemoryTimeStepDataset(sample_dir=cfg.train_dir, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    if cfg.enable_val:
        print("Validation dataset:")
        validation_dataset = InMemoryTimeStepDataset(sample_dir=cfg.val_dir, device=device)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    # se rest_minus_nonrest true, cambia
    node_channels = preprocess_cfg.node_channels
    edge_channels = preprocess_cfg.edge_channels

    torch.manual_seed(cfg.seed)
    model = MeshGraphNet(node_channels=node_channels, edge_channels=edge_channels, latent_size=cfg.latent_dim,
                         num_msgs=cfg.num_messages)
    model = model.to(device)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print(f"Loading stats from {cfg.train_dir}...")
    node_stats, edge_stats, target_stats = load_stats(cfg.train_dir)
    dt = preprocess_cfg.dt

    print(f"\tNode features: mean shape = {node_stats.mean.shape}, std shape = {node_stats.std.shape}")
    print(f"\tEdge features: mean shape = {edge_stats.mean.shape}, std shape = {edge_stats.std.shape}")
    print(f"\tTarget features: mean shape = {target_stats.mean.shape}, std shape = {target_stats.std.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, loss_type=cfg.loss_type, use_wandb=cfg.use_wandb,
                      preprocess_cfg=preprocess_cfg)

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
            trainer.validation(validation_loader=validation_dataloader, node_stats=node_stats, edge_stats=edge_stats,
                               target_stats=target_stats, dt=dt, epoch=epoch)
            epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}",
                                   "rollout_error": f"{trainer.rollout_all_step_error:.6f}"})
            print(f"\nEpoch {epoch}: Train Loss = {trainer.loss:.6f}, "
                  f"Validation Rollout Error = {trainer.rollout_all_step_error:.6f}")
        else:
            epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}"})
        idx += 1

    return trainer, node_stats, edge_stats, target_stats, dt


@hydra.main(version_base=None, config_path="../../conf/train", config_name="train")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration. Run with --multirun to sweep over multiple values,
            e.g.: python -m src.gnn.train --multirun training.lr=1e-3,5e-4,1e-4
    """
    train_cfg = TrainYaml(cfg)
    with open(Path(train_cfg.dataset_dir) / "config.yaml", "r") as file:
        data = yaml.safe_load(file)
    preprocess_cfg = PreprocessYaml(data)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    trainer, node_stats, edge_stats, target_stats, dt = train(train_cfg, preprocess_cfg, device)
    print("Test dataset:")
    test_dataset = InMemoryTimeStepDataset(sample_dir=cfg.test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    plot_training_history(trainer)
    final_rollout_error = trainer.final_test(test_dataloader=test_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="test")
    print(f"\nFinal evaluation rollout error: {final_rollout_error:.6f}")
    print("Train-sample dataset (seen trajectory):")
    train_files = sorted([f for f in os.listdir(cfg.train_dir) if f.startswith("graph_") and f.endswith(".pt")])
    train_sample_data = torch.load(os.path.join(cfg.train_dir, train_files[0]), map_location=device, weights_only=False)
    train_sample_dataloader = DataLoader(train_sample_data, batch_size=1, shuffle=False)
    train_rollout_error = trainer.final_test(test_dataloader=train_sample_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="train_sample")
    print(f"\nTrain-sample rollout error: {train_rollout_error:.6f}")
    trainer.save_model(epochnum=cfg.total_epochs - 1)


if __name__ == "__main__":
    main()
