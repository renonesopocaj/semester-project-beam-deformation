"""
normalizer.py

Contains Normalizer class and its subclasses. Implements 4 normalizers:
- Robust (Median, IQR) both isotropic and not
- Z (Gaussian) both isotropic and not
- None: doesn't change anything, only computes statistics via Z processing FIXME: probably can cancel the compute_statas
        since it has only to not do nothing

`compute_stats(tensor_list, isotropic_groups)` consumes a list of `(start, length)` tuples that describe
contiguous channel groups that must share a single statistic (mean/median, std/IQR). Length-1 groups behave
as the anisotropic per-column branch. The helper `make_isotropic_groups_from_split(split_idx, total_dim)`
reconstructs the legacy `split_idx` semantics (first split_idx channels grouped as xyz triplets, rest as
individual columns), so existing callers stay identical in behavior.
"""

from abc import ABC, abstractmethod
import torch
from src.gnn.preprocessing.stats import Stats
import numpy as np

EPS = 1e-8


def make_isotropic_groups_from_split(split_idx, total_dim):
    """
    Reproduce the legacy "first split_idx channels in xyz triplets, rest individual" pooling layout as an
    explicit list of (start, length) groups. Backward-compatible bridge for callers that historically only
    expressed an integer split.

    Args:
        split_idx: int
            Number of leading columns to pool as consecutive triplets. Must be a non-negative multiple of 3.
        total_dim: int
            Total number of columns in the tensor for which stats are computed.
    return: list of (int, int)
        Groups in form (start, length); each length-3 entry pools 3 columns together, each length-1 entry
        leaves a column anisotropic.
    """
    if split_idx < 0 or total_dim < 0:
        raise ValueError(f"split_idx and total_dim must be non-negative; got {split_idx} and {total_dim}.")
    if split_idx % 3 != 0:
        raise ValueError(f"split_idx must be divisible by 3; got {split_idx}.")
    if split_idx > total_dim:
        raise ValueError(f"split_idx={split_idx} exceeds total_dim={total_dim}.")
    groups = []
    for start in range(0, split_idx, 3):
        groups.append((start, 3))
    for start in range(split_idx, total_dim):
        groups.append((start, 1))
    return groups


def _validate_isotropic_groups(groups, total_dim):
    """
    Sanity check the isotropic-group spec against an expected total dimensionality. Groups must form a
    contiguous partition of [0, total_dim).

    Args:
        groups: list of (int, int)
            Group spec as (start, length) tuples.
        total_dim: int
            Expected total number of channels covered by the groups.
    return: None
        Raises ValueError on malformed input.
    """
    expected_start = 0
    for start, length in groups:
        if start != expected_start:
            raise ValueError(f"Non-contiguous isotropic_groups at start={start} (expected {expected_start}).")
        if length < 1:
            raise ValueError(f"Group length must be >= 1, got {length} at start={start}.")
        expected_start = start + length
    if expected_start != total_dim:
        raise ValueError(
            f"isotropic_groups cover {expected_start} channels but tensor has {total_dim} columns."
        )


def build_affine_normalizer_tensors(stats_mean, stats_std, normalizer_method):
    """
    Build affine tensors so normalization can be applied as x * scale + bias.

    Args:
        stats_mean: torch.Tensor
            Feature-wise center statistics.
        stats_std: torch.Tensor
            Feature-wise scale statistics.
        normalizer_method: str
            Normalizer identifier. If "none", identity affine is returned.
    return: (torch.Tensor, torch.Tensor, torch.Tensor)
        (scale, bias, inv_scale) affine normalization tensors.
    """
    method = str(normalizer_method).lower()
    if method == "none":
        scale = torch.ones_like(stats_std)
        bias = torch.zeros_like(stats_mean)
        inv_scale = torch.ones_like(stats_std)
        return scale, bias, inv_scale
    else:
        inv_scale = stats_std + EPS
        scale = 1.0 / inv_scale
        bias = -stats_mean * scale
        return scale, bias, inv_scale


def apply_affine_normalization(tensor, scale, bias):
    """
    Apply precomputed affine normalization.

    Args:
        tensor: torch.Tensor
            Raw tensor to normalize.
        scale: torch.Tensor
            Multiplicative factor.
        bias: torch.Tensor
            Additive factor.
    return: torch.Tensor
        Normalized tensor.
    """
    return tensor * scale + bias


def apply_affine_denormalization(tensor, stats_mean, inv_scale, normalizer_method):
    """
    Reverse affine normalization.

    Args:
        tensor: torch.Tensor
            Normalized tensor.
        stats_mean: torch.Tensor
            Feature-wise center statistics.
        inv_scale: torch.Tensor
            Feature-wise denominator (std + eps style).
        normalizer_method: str
            Normalizer identifier.
    return: torch.Tensor
        Tensor in raw space.
    """
    method = str(normalizer_method).lower()
    if method == "none":
        return tensor
    else:
        return tensor * inv_scale + stats_mean

def create_normalizer(normalize_method, isotropic):
    """
    Factory pattern, creates and returns proper Normalizer object

    Args:
        normalize_method: str
            string that indicates normalizer method, e.g. "z" | "robust"
        isotropic: bool
            true iff you want isotropic norm on features that have 3 coordinates

    :return Normalizer
    """
    if normalize_method == "z":
        if isotropic is True:
            return ZIsotropicNormalizer()
        else:
            return ZAnisotropicZNormalizer()
    elif normalize_method == "robust":
        if isotropic is True:
            return RobustIsotropicNormalizer()
        else:
            return RobustAnisotropicNormalizer()
    elif normalize_method.lower() == "none":
        return NoNormalizer()
    else:
        raise NotImplementedError(f"Unknown normalizer method: {normalize_method}")

class Normalizer(ABC):
    # stats are saved for every feature, so everyhting is very modular and I can forget about everything
    stats: Stats

    def normalize(self, tensor, stats):
        """
        Normalize tensor using statistics. Stats.mean, Stats.std are torch.Tensor with shape (#features, 1) such that
        every feature (column -- or another dimension, but only one -- of tensor) is divided by its mean

        Args
            tensor: torch.Tensor
                tensor to normalize
            stats: Stats
                stats object

        :return torch.Tensor
        """
        return (tensor - stats.mean) / (stats.std + EPS)

    def denormalize(self, tensor, stats):
        """
        Denormalize tensor using statistics.

        Args:
            tensor: torch.Tensor
                tensor to normalize
            stats: Stats
                stats object

        :return torch.Tensor
        """
        return tensor * (stats.std + EPS) + stats.mean

    @abstractmethod
    def compute_stats(self, tensor_list, isotropic_groups):
        """
        Args:
            tensor_list: list[torch.Tensor]
                list of tensors with shape (N, F); every tensor in the list must have the same F.
            isotropic_groups: list of (int, int)
                Contiguous channel groups in (start, length) form. Length >= 2 groups pool their
                columns into a single shared statistic; length-1 groups are computed per column.

        return: Stats
            stats with the requested pooling layout.
        """
        raise NotImplementedError("compute_stats not implemented in abstract class normalizer")

class NoNormalizer(Normalizer):

    def compute_stats(self, tensor_list, isotropic_groups):
        tensor_iterator = iter(tensor_list)
        first_tensor = next(tensor_iterator)
        num_features = first_tensor.shape[1]
        _validate_isotropic_groups(isotropic_groups, num_features)

        mean = torch.zeros(num_features, device=first_tensor.device, dtype=first_tensor.dtype)
        std = torch.ones(num_features, device=first_tensor.device, dtype=first_tensor.dtype)
        return Stats(mean=mean, std=std)

class ZIsotropicNormalizer(Normalizer):

    def compute_stats(self, tensor_list, isotropic_groups):
        first_tensor = tensor_list[0]
        num_features = first_tensor.shape[1]
        _validate_isotropic_groups(isotropic_groups, num_features)

        sum_val = torch.zeros(num_features, device=first_tensor.device, dtype=first_tensor.dtype)
        sum_sq = torch.zeros(num_features, device=first_tensor.device, dtype=first_tensor.dtype)
        count = torch.zeros(num_features, device=first_tensor.device, dtype=first_tensor.dtype)

        for tensor in tensor_list:
            sum_val += tensor.sum(dim=0)
            sum_sq += (tensor ** 2).sum(dim=0)
            count += tensor.shape[0]

        # Pool stats within each isotropic group with length >= 2; length-1 groups already hold the right
        # per-column accumulators and are left untouched.
        for start, length in isotropic_groups:
            if length < 2:
                continue
            end = start + length
            pooled_sum = sum_val[start:end].sum()
            pooled_sum_sq = sum_sq[start:end].sum()
            pooled_count = count[start] * length
            sum_val[start:end] = pooled_sum
            sum_sq[start:end] = pooled_sum_sq
            count[start:end] = pooled_count

        mean = sum_val / count
        var = torch.clamp((sum_sq / count) - (mean ** 2), min=0.0)
        std = torch.sqrt(var)

        return Stats(mean=mean, std=std)

class ZAnisotropicZNormalizer(Normalizer):

    def compute_stats(self, tensor_list, isotropic_groups):
        first_tensor = tensor_list[0]
        num_features = first_tensor.shape[1]
        _validate_isotropic_groups(isotropic_groups, num_features)
        sum_val = 0
        sum_sq = 0
        count = 0
        for tensor in tensor_list:
            # tensor has shape (N, #features)
            sum_val += tensor.sum(dim=0)
            sum_sq += (tensor ** 2).sum(dim=0)
            count += tensor.shape[0]
        mean = sum_val / count
        var = torch.clamp((sum_sq / count) - (mean ** 2), min=0.0)
        std = torch.sqrt(var)
        return Stats(mean=mean, std=std)

class RobustIsotropicNormalizer(Normalizer):

    def compute_stats(self, tensor_list, isotropic_groups):
        first_tensor = tensor_list[0]
        num_features = first_tensor.shape[1]
        _validate_isotropic_groups(isotropic_groups, num_features)

        mean_np = np.zeros(num_features, dtype=np.float64)
        std_np = np.zeros(num_features, dtype=np.float64)

        for start, length in isotropic_groups:
            end = start + length
            if length >= 2:
                pooled_values = []
                for tensor in tensor_list:
                    pooled_values.append(tensor[:, start:end].detach().cpu().reshape(-1).numpy())
                pooled_values = np.concatenate(pooled_values, axis=0)

                median = np.quantile(pooled_values, 0.5)
                q1 = np.quantile(pooled_values, 0.25)
                q3 = np.quantile(pooled_values, 0.75)
                iqr = q3 - q1

                mean_np[start:end] = median
                std_np[start:end] = iqr
            else:
                values = []
                for tensor in tensor_list:
                    values.append(tensor[:, start].detach().cpu().numpy())
                values = np.concatenate(values, axis=0)

                median = np.quantile(values, 0.5)
                q1 = np.quantile(values, 0.25)
                q3 = np.quantile(values, 0.75)
                iqr = q3 - q1

                mean_np[start] = median
                std_np[start] = iqr

        mean = torch.tensor(mean_np, device=first_tensor.device, dtype=first_tensor.dtype)
        std = torch.tensor(std_np, device=first_tensor.device, dtype=first_tensor.dtype)

        return Stats(mean=mean, std=std)

class RobustAnisotropicNormalizer(Normalizer):

    def compute_stats(self, tensor_list, isotropic_groups):
        first_tensor = tensor_list[0]
        num_features = first_tensor.shape[1]
        _validate_isotropic_groups(isotropic_groups, num_features)
        all_data_np = np.concatenate([tensor.detach().cpu().numpy() for tensor in tensor_list], axis=0)

        median_np = np.quantile(all_data_np, 0.5, axis=0)
        q1_np = np.quantile(all_data_np, 0.25, axis=0)
        q3_np = np.quantile(all_data_np, 0.75, axis=0)
        iqr_np = q3_np - q1_np

        mean = torch.tensor(median_np, device=first_tensor.device, dtype=first_tensor.dtype)
        std = torch.tensor(iqr_np, device=first_tensor.device, dtype=first_tensor.dtype)

        return Stats(mean=mean, std=std)