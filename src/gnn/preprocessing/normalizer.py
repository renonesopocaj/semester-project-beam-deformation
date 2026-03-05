from abc import ABC, abstractmethod
import torch
from src.gnn.preprocessing.stats import Stats

EPS = 1e-8

class NormalizerFactory:

    @staticmethod
    def create_normalizer(normalize_method, isotropic):
        if normalize_method == "z":
            if isotropic is True:
                return ZIsotropicNormalizer()
            else:
                return ZAnisotropicZNormalizer()
        else:
            raise NotImplementedError

class Normalizer(ABC):
    # stats are saved for every feature, so everyhting is very modular and I can forget about everything
    stats: Stats

    def normalize(self, tensor, stats):
        """
        Normalize tensor using statistics. Stats.mean, Stats.std are torch.Tensor with shape (#features, 1) such that
        every feature (column -- or another dimension, but only one -- of tensor) is divided by its mean

        Args
            tensor: torch.Tensor
            stats: Stats,
            eps: float = 1e-8
        :return torch.Tensor
        """
        return (tensor - stats.mean) / (stats.std + EPS)

    def denormalize(self, tensor, stats):
        """
        Denormalize tensor using statistics.

        Args:
           tensor: torch.Tensor
           stats: Stats
           eps: float
        :return torch.Tensor
        """
        return tensor * (stats.std + EPS) + stats.mean

    @abstractmethod
    def compute_stats(self, tensor_list):
        raise NotImplementedError("compute_stats not implemented in abstract class normalizer")

class ZIsotropicNormalizer(Normalizer):

    def compute_stats(self, tensor_list):
        """
        Each feature has 3 coordinates to sum along

        Args:
            tensor_list: List[torch.tensor]
        :return:
        """
        sum_val = 0
        sum_sq = 0
        count = 0
        for tensor in tensor_list:
            # sum across first dimension, i.e. (N, 3, 4) --> (4) tensor of means (for each feature, the mean along all
            # coordinates)
            sum_val += tensor.sum(dim=(0, 1))
            sum_sq += (tensor ** 2).sum(dim=(0,1))
            count += tensor.shape[0] * 3 # count coordinates (3)
        mean = sum_val / count
        var = (sum_sq / count) - (mean ** 2)
        std = torch.sqrt(var)
        return Stats(mean=mean, std=std)

class ZAnisotropicZNormalizer(Normalizer):

    def compute_stats(self, tensor_list):
        """
        Compute mean and std statistics from a list of tensors.

        Args:
            tensor_list: list[torch.Tensor]
                a list of tensors

        :return Stats object (mean: torch.Tensor, std: torch.Tensor)
        """
        sum_val = 0
        sum_sq = 0
        count = 0
        for tensor in tensor_list:
            # sum across first dimension, i.e. (N, 3, 4) --> (3, 4) tensor of means (for each coordinate, the mean of
            # feature)
            sum_val += tensor.sum(dim=0)
            sum_sq += (tensor ** 2).sum(dim=0)
            count += tensor.shape[0]
        mean = sum_val / count
        var = (sum_sq / count) - (mean ** 2)
        std = torch.sqrt(var)
        return Stats(mean=mean, std=std)

class RobustIsotropicNormalizer(Normalizer):
    pass

class RobustAnisotropicNormalizer(Normalizer):
    pass