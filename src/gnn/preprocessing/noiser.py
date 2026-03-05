import torch
from abc import ABC, abstractmethod

class Noiser(ABC):

    @abstractmethod
    def add_noise_train(self, x, v, dt):
        raise NotImplementedError("abstract method add_noise_train")

class GaussianNoiser(Noiser):

    def __init__(self, noise_scale):
        self.noise_scale = noise_scale

    def add_noise_train(self, x, v, dt):
        # Add noise to training data
        noise = self.noise_scale * torch.randn_like(v)
        x = x + noise
        v = v - noise / dt  # adjust velocity accordingly
        # Obtain the next-time-step velocities and positions (same tensors but shifted of 1 timestep)
        return x, v

class NoNoiser(Noiser):

    def add_noise_train(self, x, v, dt):
        return x, v