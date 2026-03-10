from pathlib import Path
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml
from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.preprocessing.stats import load_stats
from src.gnn.training.checkpointing import load_trainer_for_eval


def main():
    training_id = "2026-03-08_02-22-47"
    model_root = Path("saved_models")
    checkpoint_path = model_root / training_id / "checkpoint_epoch_59" / "torch_state.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(f"saved_models/{training_id}/preprocess.yaml", "r") as file:
        preprocess_cfg = yaml.safe_load(file)
    preprocess_cfg = PreprocessYaml(preprocess_cfg)
    with open(f"saved_models/{training_id}/train.yaml", "r") as file:
        train_cfg = yaml.safe_load(file)
    train_cfg = TrainYaml(train_cfg)
    loss_type = "mae"
    model = MeshGraphNet(node_channels=preprocess_cfg.node_channels, edge_channels=preprocess_cfg.edge_channels,
                         latent_size=train_cfg.latent_dim, num_msgs=train_cfg.num_messages)
    trainer = load_trainer_for_eval(checkpoint_path=checkpoint_path, preprocess_cfg=preprocess_cfg,
                                    device=device, model_root=model_root, model=model,
                                    loss_type=loss_type, training_id=training_id)
    # self.model_dir = Path(model_root) / self.training_id
    # self.model_dir.mkdir(parents=True, exist_ok=True)
    # self.training_plots_dir = self.model_dir / "training"
    # self.training_plots_dir.mkdir(parents=True, exist_ok=True)
    # self.model_evaluation_dir = self.model_dir / "model_evaluation"
    # self.model_evaluation_dir.mkdir(parents=True, exist_ok=True)
    trainer.model_dir = trainer.model_dir / "evaluation_uncoupled"
    trainer.model_dir.mkdir(parents=True, exist_ok=True)
    trainer.training_plots_dir = trainer.model_dir / "training"
    trainer.training_plots_dir.mkdir(parents=True, exist_ok=True)
    trainer.model_evaluation_dir = trainer.model_dir / "model_evaluation"
    trainer.model_evaluation_dir.mkdir(parents=True, exist_ok=True)

    dataset_train_dir = "dataset/processed/z_isotr_log_displace/medium/train" # TODO: load these from the checkpoint's trainer_state.json or config
    dataset_test_dir = "dataset/processed/z_isotr_log_displace/medium/test" # TODO: load these from the checkpoint's trainer_state.json or config
    dataset_val_dir = "dataset/processed/z_isotr_log_displace/medium/val"
    dt = 0.08  # TODO: read from preprocess config

    node_stats, edge_stats, target_stats = load_stats(dataset_train_dir)
    print("Test dataset:")
    test_dataset = InMemoryTimeStepDataset(sample_dir=dataset_test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    validation_dataset = InMemoryTimeStepDataset(sample_dir=dataset_val_dir, device=device)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    print("\nStarting validation\n")
    print("\nTest set:")
    final_rollout_error = trainer.final_test(test_dataloader=test_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="test")
    print(f"Final evaluation rollout error: {final_rollout_error:.6f}")

    # Test model on a new trajectory (validation set)
    print("\nValidation dataset:\n")
    final_rollout_error = trainer.final_test(test_dataloader=validation_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="validation")
    print(f"\nFinal evaluation rollout error: {final_rollout_error:.6f}")


if __name__ == "__main__":
    main()
