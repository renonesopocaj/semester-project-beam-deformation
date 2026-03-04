from abc import ABC, abstractmethod
import torch
from stats import Stats

class NormalizerFactory:
    @staticmethod
    def create_normalizer(normalize_method, isotropic):
        if normalize_method == "z":
            if isotropic is True:
                pass
            else:
                pass
        else:
            raise NotImplementedError

class Normalizer(ABC):

    @abstractmethod
    def normalize(self, tensor_list):
        raise NotImplementedError

class ZIsotropicNormalizer(Normalizer):

    def normalize(self, tensor_list):
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

    def normalize(self, tensor_list):
        """
        Each feature and coordinate will be computed independently

        Args:
            tensor_list: List[torch.tensor]
        :return:
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