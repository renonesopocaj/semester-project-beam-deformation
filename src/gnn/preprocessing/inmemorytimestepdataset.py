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
    """

    def __init__(self, sample_dir, device):
        self.data = []
        sample_files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.startswith("graph_")
                               and f.endswith(".pt")])
        print(f"\tFound {len(sample_files)} sample files")
        for sample_file in sample_files:
            sample_data = torch.load(sample_file, map_location=device, weights_only=False)
            self.data.extend(sample_data)  # Flatten all time steps into one list
        print(f"\tLoaded the dataset with {len(self.data)} samples")

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)