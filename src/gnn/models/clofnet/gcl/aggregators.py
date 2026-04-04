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
