"""
noiser.py
FIXME: DEPRECATED
Contains abstract class Noiser and GaussianNoiser to add noise to training data.
"""

# import torch
# from abc import ABC, abstractmethod
#
# class Noiser(ABC):
#
#     @abstractmethod
#     def add_noise_train(self, x):
#         raise NotImplementedError("abstract method add_noise_train")
#
# class GaussianNoiser(Noiser):
#
#     def __init__(self, noise_scale):
#         self.noise_scale = noise_scale
#
#     def add_noise_train(self, x):
#         # Add noise to training data
#         noise = self.noise_scale * torch.randn_like(x)
#         x = x + noise
#         return x
#
# class NoNoiser(Noiser):
#
#     def add_noise_train(self, x):
#         return x, v