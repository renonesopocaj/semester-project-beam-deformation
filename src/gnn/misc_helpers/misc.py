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


def relative_mae_sum_and_count(pred, target):
    """
    Sum of elementwise relative MAE and element count for pooling across batches or timesteps.

    Args:
        pred: torch.Tensor
            Predictions (same shape as target).
        target: torch.Tensor
            Targets (same shape as pred).
    return: (torch.Tensor, torch.Tensor)
        Scalar sum of |pred-target|/max(|target|,1e-12), and scalar count = numel for weighting.
    """
    rel = compute_relative_mae(pred, target)
    n = rel.numel()
    return rel.sum(), torch.as_tensor(n, device=rel.device, dtype=rel.dtype)


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


def human_readable_storage(num_bytes):
    """
    Format a byte count as MB or GB (1024-based) for logs.

    Args:
        num_bytes: int
            Non-negative size in bytes.
    return: str
        Human-readable string, e.g. "12.345 MB" or "1.234 GB".
    """
    nb = int(num_bytes)
    if nb >= 1024 ** 3:
        return f"{nb / (1024 ** 3):.3f} GB"
    return f"{nb / (1024 ** 2):.3f} MB"


def pyg_data_tensor_storage_bytes(data):
    """
    Sum element storage for all tensor attributes on a PyG Data or Batch.

    Args:
        data: object
            torch_geometric.data.Data or Batch (must expose .keys and __getitem__).
    return: int
        Total bytes: sum of element_size * numel for each tensor attribute.
    """
    keys_attr = data.keys
    try:
        if callable(keys_attr):
            key_list = list(keys_attr())
        else:
            key_list = list(keys_attr)
    except Exception:
        return 0
    total = 0
    for key in key_list:
        try:
            value = data[key]
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.numel()
    return total


def module_param_and_buffer_storage_bytes(module):
    """
    Sum storage for all parameters and registered buffers of a module.

    Args:
        module: torch.nn.Module
            Module to measure.
    return: int
        Total bytes for parameters and buffers (theoretical weight footprint).
    """
    total = 0
    for param in module.parameters():
        total += param.element_size() * param.numel()
    for buf in module.buffers():
        total += buf.element_size() * buf.numel()
    return total