from abc import ABC, abstractmethod
import torch
from src.gnn.preprocessing.stats import Stats
import numpy as np

EPS = 1e-8

class NormalizerFactory:

    @staticmethod
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
    def compute_stats(self, tensor_list, split_idx):
        """
        Args:
            tensor_list: list[torch.Tensor]
                list of tensors with shape (N, F)
            split_idx: int
                first split_idx features are grouped in xyz triplets

        return: Stats
            robust stats with isotropic pooling on triplets
        """
        raise NotImplementedError("compute_stats not implemented in abstract class normalizer")

class ZIsotropicNormalizer(Normalizer):

    def compute_stats(self, tensor_list, split_idx):
        assert split_idx % 3 == 0

        sum_val = torch.zeros(tensor_list[0].shape[1],device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        sum_sq = torch.zeros(tensor_list[0].shape[1], device=tensor_list[0].device,dtype=tensor_list[0].dtype)
        count = torch.zeros(tensor_list[0].shape[1], device=tensor_list[0].device,dtype=tensor_list[0].dtype)

        for tensor in tensor_list:
            # sum across first dimension, i.e. (N, #features) --> (#features) tensor of means (for each feature,
            # the mean along all coordinates)
            sum_val += tensor.sum(dim=0)
            sum_sq += (tensor ** 2).sum(dim=0)
            count += tensor.shape[0]

        for i in range(0, split_idx, 3):
            pooled_sum = sum_val[i] + sum_val[i + 1] + sum_val[i + 2]
            pooled_sum_sq = sum_sq[i] + sum_sq[i + 1] + sum_sq[i + 2]
            pooled_count = count[i] * 3

            sum_val[i] = pooled_sum
            sum_val[i + 1] = pooled_sum
            sum_val[i + 2] = pooled_sum

            sum_sq[i] = pooled_sum_sq
            sum_sq[i + 1] = pooled_sum_sq
            sum_sq[i + 2] = pooled_sum_sq

            count[i] = pooled_count
            count[i + 1] = pooled_count
            count[i + 2] = pooled_count

        mean = sum_val / count
        var = torch.clamp((sum_sq / count) - (mean ** 2), min=0.0)
        std = torch.sqrt(var)

        return Stats(mean=mean, std=std)

class ZAnisotropicZNormalizer(Normalizer):

    def compute_stats(self, tensor_list, split_idx):
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

    def compute_stats(self, tensor_list, split_idx):
        first_tensor = tensor_list[0]
        num_features = first_tensor.shape[1]

        assert split_idx % 3 == 0, "split_idx must be divisible by 3"

        mean_np = np.zeros(num_features, dtype=np.float64)
        std_np = np.zeros(num_features, dtype=np.float64)

        for i in range(0, split_idx, 3):
            pooled_values = []
            for tensor in tensor_list:
                pooled_values.append(tensor[:, i:i + 3].detach().cpu().reshape(-1).numpy())
            pooled_values = np.concatenate(pooled_values, axis=0)

            median = np.quantile(pooled_values, 0.5)
            q1 = np.quantile(pooled_values, 0.25)
            q3 = np.quantile(pooled_values, 0.75)
            iqr = q3 - q1

            mean_np[i] = median
            mean_np[i + 1] = median
            mean_np[i + 2] = median

            std_np[i] = iqr
            std_np[i + 1] = iqr
            std_np[i + 2] = iqr

        for i in range(split_idx, num_features):
            values = []
            for tensor in tensor_list:
                values.append(tensor[:, i].detach().cpu().numpy())
            values = np.concatenate(values, axis=0)

            median = np.quantile(values, 0.5)
            q1 = np.quantile(values, 0.25)
            q3 = np.quantile(values, 0.75)
            iqr = q3 - q1

            mean_np[i] = median
            std_np[i] = iqr

        mean = torch.tensor(mean_np, device=first_tensor.device, dtype=first_tensor.dtype)
        std = torch.tensor(std_np, device=first_tensor.device, dtype=first_tensor.dtype)

        return Stats(mean=mean, std=std)

class RobustAnisotropicNormalizer(Normalizer):

    def compute_stats(self, tensor_list, split_idx):
        first_tensor = tensor_list[0]
        all_data_np = np.concatenate([tensor.detach().cpu().numpy() for tensor in tensor_list], axis=0)

        median_np = np.quantile(all_data_np, 0.5, axis=0)
        q1_np = np.quantile(all_data_np, 0.25, axis=0)
        q3_np = np.quantile(all_data_np, 0.75, axis=0)
        iqr_np = q3_np - q1_np

        mean = torch.tensor(median_np, device=first_tensor.device, dtype=first_tensor.dtype)
        std = torch.tensor(iqr_np, device=first_tensor.device, dtype=first_tensor.dtype)

        return Stats(mean=mean, std=std)