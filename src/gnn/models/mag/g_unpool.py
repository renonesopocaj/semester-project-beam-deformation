"""
g_unpool.py

PyTorch graph unpooling for fixed-topology MAgNET graphs.
"""

import torch
import torch.nn as nn

from src.gnn.models.mag.g_pool import _pool_index_from_subgraphs


class GUnpool(nn.Module):
    """
    Replicate pooled clique features back to parent graph nodes.
    """
    pool_index: torch.Tensor
    nodes: int

    def __init__(self, subgraphs, nodes):
        """
        Args:
            subgraphs: list[torch.Tensor]
                Non-overlapping cliques used during the corresponding pooling step.
            nodes: int
                Number of nodes in the parent graph.
        return:
            None
        """
        super().__init__()
        self.nodes = nodes
        pool_index = _pool_index_from_subgraphs(subgraphs)
        if pool_index.numel() != nodes:
            raise ValueError(f"GUnpool nodes={nodes} but subgraph partition implies N={pool_index.numel()}.")
        self.register_buffer("pool_index", pool_index)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                Pooled features `[B, P, C]`.
        return:
            torch.Tensor
                Unpooled parent features `[B, N, C]`.
        """
        return x.index_select(1, self.pool_index)
