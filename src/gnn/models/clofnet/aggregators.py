"""
aggregators.py

Defines the function to aggregate, with sum or mean, all the messages
"""

import torch

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Unsorted segment sum using `scatter_add_`.

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
    Unsorted segment mean using `scatter_add_`.

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
    avg_result = result / count.clamp(min=1)
    return avg_result