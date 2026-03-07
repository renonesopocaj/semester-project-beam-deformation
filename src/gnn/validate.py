from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.preprocessing.stats import load_stats
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.training.checkpointing import build_trainer_from_checkpoint, model_builder, optimizer_builder


def main(CHECKPOINT_DIR):
    model_root = Path("saved_models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer, _ = build_trainer_from_checkpoint(
        checkpoint_dir=CHECKPOINT_DIR, model_builder=model_builder,
        optimizer_builder=optimizer_builder, scheduler_builder=scheduler_factory,
        device=device, model_root=model_root,
    )

    dataset_train_dir = "dataset/processed/z_anisotr_log_displace/small/train" # TODO: load these from the checkpoint's trainer_state.json or config
    dataset_test_dir = "dataset/processed/z_anisotr_log_displace/small/test" # TODO: load these from the checkpoint's trainer_state.json or config
    dt = 0.08  # TODO: read from preprocess config

    node_stats, edge_stats, target_stats = load_stats(dataset_train_dir)
    print("Test dataset:")
    test_dataset = InMemoryTimeStepDataset(sample_dir=dataset_test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    final_rollout_error = trainer.final_test(test_dataloader=test_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="test")
    print(f"\nFinal evaluation rollout error: {final_rollout_error:.6f}")


if __name__ == "__main__":
    CHECKPOINT_DIR = Path("saved_models") / "2026-03-07_02-36-43" / "Epoch_49_GenLoss_0.0137837054.pth"
    main(CHECKPOINT_DIR)
