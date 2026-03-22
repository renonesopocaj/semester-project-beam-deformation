"""
inmemorytimestepdataset.py
In-memory dataset for graph data.
"""

from torch_geometric.data import Dataset
import os
import torch

class InMemoryTimeStepDataset(Dataset):
    """
    In-memory dataset for graph data.
    Loads all preprocessed graphs into memory for fast training.
    Each graph_*.pt file corresponds to one trajectory (a sequence of timesteps).
    """
    data: list
    trajectory_boundaries: list[tuple[int, int]]

    def __init__(self, sample_dir, device):
        self.data = []
        self.trajectory_boundaries = []
        sample_files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.startswith("graph_")
                               and f.endswith(".pt")])
        print(f"\tFound {len(sample_files)} sample files")
        for sample_file in sample_files:
            sample_data = torch.load(sample_file, map_location=device, weights_only=False)
            start = len(self.data)
            self.data.extend(sample_data)  # Flatten all time steps into one list
            self.trajectory_boundaries.append((start, len(self.data)))
        print(f"\tLoaded the dataset with {len(self.data)} samples "
              f"({self.num_trajectories} trajectories)")

    @property
    def num_trajectories(self):
        return len(self.trajectory_boundaries)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)