from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

from src.gnn.helpers.misc.MGN_Beam_DEMO import MeshGraphNet
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.preprocessing.stats import load_stats
from src.gnn.training.checkpointing import load_trainer_for_eval


def main(CHECKPOINT_DIR):
    model_root = Path("saved_models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loss_type = "mse"
    MeshGraphNet(node_channels=preprocess_cfg.node_channels,
                 edge_channels=preprocess_cfg.edge_channels,
                 latent_size=128,
                 num_msgs=8)

    trainer = load_trainer_for_eval(checkpoint_path=CHECKPOINT_DIR, device=device, model_root=model_root, model=model)

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
