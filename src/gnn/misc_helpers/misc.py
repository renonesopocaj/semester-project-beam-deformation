"""
misc.py
Contains miscellaneous helper functions.
"""

import torch
import numpy as np

def compute_mean_relative_mae(pred, target):
    return torch.mean(compute_relative_mae(pred, target))

def compute_relative_mae(pred, target):
    return torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-12)

def to_numpy(value):
    """
    Args:
        value: object
            Tensor-like or array-like object.
        return: np.ndarray
            Detached numpy array.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)