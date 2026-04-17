"""
layers.py

Auxiliary layers used by ClofNet variants.

This file contains two kinds of components:
1. Scalar/edge embedding helpers, especially `GaussianLayer`.
2. An attention-heavy equivariant block used in alternative experiments for
   molecular conformation generation.

Paper mapping used below:
- Eq. (13)-(15): feature embedding before message passing.
- Eq. (20)-(24): Graph Transformer Block (GTB) attention refinement.
- Eq. (2), Eq. (8), Eq. (19): complete-frame construction, vectorization and
  coordinate update.
- Appendix A.3.3: scalarization adds low extra cost compared with message
  passing, so feature expansion around those scalars is practical.

Some modules here are explicit implementation extensions rather than exact paper
objects. In particular, `GaussianLayer` is a code-level alternative to the
paper's "Fourier(s_ij) or MLP(s_ij)" embedding choice from Eq. (15), and the
multi-head attention implementation below is a richer realization of the GTB
idea from Eq. (20)-(24).
"""

import math
import torch
import torch.nn as nn


def gaussian(x, mean, std):
    """
    Evaluate Gaussian basis functions with broadcasted parameters.
    The functional form is the standard normal-basis expansion:
        exp(-0.5 * ((x - mean) / std)^2) / (sqrt(2 pi) * std)

    Args:
        x: torch.Tensor
            Input values to expand.
        mean: torch.Tensor
            Gaussian means broadcastable to `x`.
        std: torch.Tensor
            Gaussian standard deviations broadcastable to `x`.
    return: torch.Tensor
        Gaussian basis values with the same broadcasted shape.
    """
    pi = 3.14159
    normalizer = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (normalizer * std)


class GaussianLayer(nn.Module):
    """
    Learnable Gaussian basis expansion for scalar distances.

    Paper relation:
    - The closest place in the paper is Eq. (15), where the scalarized features
      s_ij are embedded with either `Fourier(s_ij)` or `MLP(s_ij)` before graph
      message passing.
    - This class is not a literal equation from the paper. It is a practical
      code-level replacement for that embedding step when distances are treated
      as typed scalar inputs.

    Each edge type learns an affine transform
                x_typed = mul[type] * x + bias[type]
    before the scalar is expanded into K Gaussian channels.
    """

    def __init__(self, K, edge_types):
        """
        K=128 edge_types=1024
        Args:
            K: int
                Number of Gaussian basis channels.
            edge_types: int
                Number of discrete edge types that receive independent affine
                scale and bias parameters.
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)

        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        """
        Args:
            x: torch.Tensor
                Scalar distances with shape [E] or [E, 1].
            edge_types: torch.Tensor
                Integer edge type ids with shape [E].
        return: torch.Tensor
            Gaussian-expanded distance features with shape [E, K].
        """
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, self.K)

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)