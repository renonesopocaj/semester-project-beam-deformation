import torch
from dataclasses import dataclass

@dataclass
class Stats:
    """Statistics for data normalization."""
    mean: torch.Tensor
    std: torch.Tensor

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=torch.tensor(d["mean"]), std=torch.tensor(d["std"]))

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def compute_stats(tensor_list) -> Stats:
    """Compute mean and std statistics from a list of tensors."""
    sum_val = 0
    sum_sq = 0
    count = 0
    for tensor in tensor_list:
        sum_val += tensor.sum(dim=0)
        sum_sq += (tensor**2).sum(dim=0)
        count += tensor.shape[0]
    mean = sum_val / count
    var = (sum_sq / count) - (mean**2)
    std = torch.sqrt(var)
    return Stats(mean=mean, std=std)


def normalize(tensor: torch.Tensor, stats: Stats, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor using statistics."""
    return (tensor - stats.mean) / (stats.std + eps)


def denormalize(tensor: torch.Tensor, stats: Stats, eps: float = 1e-8) -> torch.Tensor:
    """Denormalize tensor using statistics."""
    return tensor * (stats.std + eps) + stats.mean