from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.preprocessing.stats import load_stats
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
from src.gnn.training.checkpointing import build_trainer_from_checkpoint


def _model_builder(model_cfg):
    return MeshGraphNet(node_channels=model_cfg["node_channels"], edge_channels=model_cfg["edge_channels"],
                        latent_size=model_cfg["latent_size"], num_msgs=model_cfg["num_msgs"])


def _optimizer_builder(model, optimizer_cfg):
    return torch.optim.Adam(model.parameters(), lr=optimizer_cfg["lr"])


def _scheduler_builder(optimizer, scheduler_cfg):
    return scheduler_factory(optimizer, scheduler_cfg)


def main():
    checkpoint_dir = Path("saved_models") / "2026-03-07_02-36-43" / "checkpoint_epoch_49"
    model_root = Path("saved_models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer, _ = build_trainer_from_checkpoint(
        checkpoint_dir=checkpoint_dir, model_builder=_model_builder,
        optimizer_builder=_optimizer_builder, scheduler_builder=_scheduler_builder,
        device=device, model_root=model_root,
    )

    # TODO: load these from the checkpoint's trainer_state.json or config
    train_dir = "dataset/processed/z_anisotropic_miseslog/small/train"
    test_dir = "dataset/processed/z_anisotropic_miseslog/small/test"
    node_stats, edge_stats, target_stats = load_stats(train_dir)
    dt = 0.01  # TODO: read from preprocess config

    print("Test dataset:")
    test_dataset = InMemoryTimeStepDataset(sample_dir=test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    final_rollout_error = trainer.final_test(test_dataloader=test_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="test")
    print(f"\nFinal evaluation rollout error: {final_rollout_error:.6f}")


if __name__ == "__main__":
    main()
