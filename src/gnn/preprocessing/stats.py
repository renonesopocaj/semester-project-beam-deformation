"""
stats.py
Contains Stats dataclass and StatsModule to compute and retrieve statistics for normalization.
"""

import torch
import torch.nn as nn
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


class StatsModule(nn.Module):
    """
    A torch.nn.Module wrapper for Stats that registers mean and std as buffers.
    This ensures the tensors automatically move to the correct device when the
    parent module is moved (e.g., to GPU).
    """

    def __init__(self, stats: Stats):
        """
        Args:
            stats: Stats
                Stats dataclass containing mean and std tensors.
        """
        super().__init__()
        self.register_buffer("mean", stats.mean.clone())
        self.register_buffer("std", stats.std.clone())

    @classmethod
    def from_stats(cls, stats: Stats) -> "StatsModule":
        """
        Create a StatsModule from a Stats dataclass.

        Args:
            stats: Stats
                Stats dataclass to convert.
            return: StatsModule
                Module with registered buffers.
        """
        return cls(stats)

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


def load_theta_max(train_save_dir):
    """
    Args:
        train_save_dir: str or Path
            Path to the directory where training data was saved.
        return: float
            Maximum allowed boundary dihedral angle measured on the train set.
    """
    theta_max_path = Path(train_save_dir) / "stats" / "theta_max.json"
    if not theta_max_path.exists():
        raise FileNotFoundError(f"Theta max file not found at {theta_max_path}. Preprocess the training data first.")
    with open(theta_max_path, "r") as f:
        theta_max_payload = json.load(f)
    return float(theta_max_payload["theta_max"])