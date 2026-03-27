"""
helpers_gcl.py
"""

import torch
from torch import nn

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    TensorFlow-style unsorted segment sum using `scatter_add_`.

    Relation to the paper:
    - This helper implements the neighborhood aggregation pattern appearing in the sum over neighbors
        in Eqs. (17), (19) and (22).
    - The input `segment_ids` indicates which node each edge contribution is aggregated into.

    Args:
        data: torch.Tensor
            Values to aggregate, shape [E, D].
        segment_ids: torch.Tensor
            Destination node indices, shape [E].
        num_segments: int
            Number of output segments / nodes.

    return: torch.Tensor
        Aggregated tensor, shape [num_segments, D].
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)

    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """
    TensorFlow-style unsorted segment mean using `scatter_add_`.

    Relation to the paper:
    - This helper implements normalized neighbor aggregation for coordinate
      updates, analogous to the averaged neighbor contribution used in code for
      Eqs. (19) and (22).

    Args:
        data: torch.Tensor
            Values to aggregate, shape [E, D].
        segment_ids: torch.Tensor
            Destination node indices, shape [E].
        num_segments: int
            Number of output segments / nodes.

    return: torch.Tensor
        Mean-aggregated tensor, shape [num_segments, D].
    """
    result_shape = (num_segments, data.size(1))

    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))

    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)

    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))

    return result / count.clamp(min=1)

class MLP(nn.Module):
    """
    Small fully-connected adapter network.

    This helper is not a named block in the paper. It is a generic MLP utility
    used to parameterize the learnable functions that the paper denotes with
    symbols such as phi_m, phi_h, phi_q, phi_k and phi_v in Appendix A.3.1.

    The architecture is:
        Linear -> LeakyReLU -> Linear -> LeakyReLU
        -> Linear -> LeakyReLU -> Linear

    Args:
        size_in: int
            Input feature dimension.
        size_out: int
            Output feature dimension.
        size_hidden: int
            Hidden feature dimension used in all intermediate layers.
    """

    def __init__(self, size_in, size_out, size_hidden):
        super().__init__()
        # FIXME: call activation factory instead of this
        self.net = nn.Sequential(nn.Linear(size_in, size_hidden),
                                nn.LeakyReLU(0.2),
                                nn.Linear(size_hidden, size_hidden),
                                nn.LeakyReLU(0.2),
                                nn.Linear(size_hidden, size_hidden),
                                nn.LeakyReLU(0.2),
                                nn.Linear(size_hidden, size_out))

    def forward(self, x):
        """
        Run the MLP.

        Args:
            x: torch.Tensor
                Input tensor of shape [..., nin].

            return: torch.Tensor
                Output tensor of shape [..., nout].
        """
        return self.net(x)
