"""
g_pool.py

PyTorch graph pooling for fixed-topology MAgNET graphs.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter


def _pool_index_from_subgraphs(subgraphs):
    """
    Map each parent node index to its pooled clique id (order matches `subgraphs`).

    Args:
        subgraphs: list[torch.Tensor]
            Disjoint node index tensors, one per clique.
    return:
        torch.Tensor
            Long tensor `pool_index` of shape `[N]` with `pool_index[j] == i` iff `j in subgraphs[i]`.
    """
    if len(subgraphs) == 0:
        raise ValueError("subgraphs must be non-empty.")
    device = subgraphs[0].device
    node_ids = torch.cat([s.long().reshape(-1) for s in subgraphs], dim=0)
    clique_ids = torch.cat([torch.full((s.numel(),), i, dtype=torch.long, device=device)
                            for i, s in enumerate(subgraphs)], dim=0)
    num_nodes = int(node_ids.max().item()) + 1
    pool_index = torch.empty((num_nodes,), dtype=torch.long, device=device)
    pool_index[node_ids] = clique_ids
    return pool_index


class GPool(nn.Module):
    """
    Clique pooling over `[batch, nodes, channels]` tensors.
    """

    pool_index: torch.Tensor
    num_pools: int
    aggr: str

    def __init__(self, subgraphs, aggr):
        """
        Args:
            subgraphs: list[torch.Tensor]
                Non-overlapping cliques from the parent graph.
            aggr: str
                Pooling aggregation: `max`, `mean`, `sum`, or `min`.
        return:
            None
        """
        super().__init__()
        self.aggr = aggr
        pool_index = _pool_index_from_subgraphs(subgraphs)
        self.register_buffer("pool_index", pool_index)
        self.num_pools = len(subgraphs)
        self._scatter_reduce = {"max": "max", "min": "min", "sum": "sum", "mean": "mean"}[aggr]

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                Dense graph features `[B, N, C]`.
        return:
            torch.Tensor
                Pooled features `[B, P, C]`.
        """
        batch_size, num_nodes, channels = x.shape[0], x.shape[1], x.shape[2]
        if num_nodes != self.pool_index.numel():
            raise ValueError(f"GPool expected N={self.pool_index.numel()} nodes, got x with N={num_nodes}.")
        batch_offsets = torch.arange(batch_size, dtype=torch.long, device=x.device) * self.num_pools
        index = self.pool_index.view(1, num_nodes) + batch_offsets.view(batch_size, 1)
        pooled = scatter(x.reshape(batch_size * num_nodes, channels),
                         index.reshape(batch_size * num_nodes), dim=0, dim_size=batch_size * self.num_pools,
                         reduce=self._scatter_reduce)
        return pooled.reshape(batch_size, self.num_pools, channels)
