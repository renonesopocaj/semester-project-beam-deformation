from pathlib import Path
import yaml
from src.gnn.helpers.misc.yamlobject import TrainYaml, PreprocessYaml
import torch
from src.gnn.training.trainer import Trainer
from torch_geometric.loader import DataLoader
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset

# def rollout_error_vs_timestep_plot():
#     # Final evaluation
#     model.eval()
#     target_stats.to("cpu")
#
#     true_rollout, pred_rollout = do_rollout(
#         model=model,
#         test_loader=test_dataloader,
#         device=device,
#         node_stats=node_stats,
#         edge_stats=edge_stats,
#         target_stats=target_stats,
#         dt=dt,
#     )
#
#     # Compute final metrics
#     final_rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t)
#     print(f"\nFinal rollout MAE: {final_rollout_error:.6f}")
#
#     # Compute per-timestep errors
#     timestep_errors = []
#     for true_graph, pred_graph in zip(true_rollout, pred_rollout):
#         error = mae(true_graph.x_pos_t, pred_graph.x_pos_t)
#         timestep_errors.append(error.item())
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(timestep_errors)
#     plt.xlabel("Timestep")
#     plt.ylabel("Position Error (MAE)")
#     plt.title("Prediction Error Over Rollout")
#     plt.grid(True)
#     plt.savefig("rollout_error_over_time.png")
#     plt.show()
#
#     print(f"Average per-timestep error: {np.mean(timestep_errors):.6f}")

def main():
    model_path = Path("saved_models") / "2026-03-07_02-36-43"
    with open(model_path / "train.yaml", 'r') as file:
        train_cfg = yaml.safe_load(file)
    train_cfg = TrainYaml(train_cfg)
    with open(Path(train_cfg.dataset_dir) / "preprocess.yaml", "r") as file:
        preprocess_cfg = yaml.safe_load(file)
    preprocess_cfg = PreprocessYaml(preprocess_cfg)
    trainer = Trainer()
    trainer.load_model()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    final_rollout_error = trainer.final_test(test_dataloader=test_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="test")
    print(f"\nFinal evaluation rollout error: {final_rollout_error:.6f}")