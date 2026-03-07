import torch
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Stats:
    """
    Statistics for data normalization.
    """
    mean: torch.Tensor # or median
    std: torch.Tensor # or IQR

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=torch.tensor(d["mean"]), std=torch.tensor(d["std"]))

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

def load_stats(train_save_dir):
    """
    Args:
        train_save_dir: str or Path
            Path to a stats file or to the directory where training data was saved.
        return: (Stats, Stats, Stats)
            node_stats, edge_stats, target_stats loaded from stats.json.
    """
    stats_path = Path(train_save_dir) / "stats" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found at {stats_path}. Preprocess the training data first.")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    node_stats = Stats.from_dict(stats["node"])
    edge_stats = Stats.from_dict(stats["edge"])
    target_stats = Stats.from_dict(stats["target"])
    return node_stats, edge_stats, target_stats