"""
pool_utils.py

Topology utilities for the PyTorch MAgNET implementation.
"""

import torch


def edge_index_to_adjacency(edge_index, num_nodes, device):
    """
    Build a symmetric boolean adjacency matrix with self-loops from PyG edge_index.

    Args:
        edge_index: torch.Tensor
            Edge indices `[2, E]`, using PyG source-to-target convention.
        num_nodes: int
            Number of nodes in one fixed-topology graph.
        device: torch.device
            Device for the adjacency tensor.
    return:
        torch.Tensor
            Boolean adjacency `[N, N]` where row is target node and column is source node.
    """
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    source = edge_index[0].long().to(device)
    target = edge_index[1].long().to(device)
    adjacency[target, source] = True
    adjacency[source, target] = True
    adjacency.fill_diagonal_(True)
    return adjacency


def adjacency_window(adjacency, window_size):
    """
    Expand aggregation support to `A ** window_size` in the boolean semiring.

    Args:
        adjacency: torch.Tensor
            Boolean adjacency `[N, N]` with self-loops.
        window_size: int
            Graph window size, e.g. 2 for MAgNET's common `A^2` support.
    return:
        torch.Tensor
            Boolean adjacency with expanded support.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    current = adjacency
    adjacency_float = adjacency.float()
    for _ in range(window_size - 1):
        current = (current.float() @ adjacency_float) > 0
    current.fill_diagonal_(True)
    return current


def new_adjacency(subgraphs, adjacency):
    """
    Create pooled adjacency from parent adjacency and clique subgraphs.

    Args:
        subgraphs: list[torch.Tensor]
            Non-overlapping node lists.
        adjacency: torch.Tensor
            Parent boolean adjacency `[N, N]`.
    return:
        torch.Tensor
            Pooled boolean adjacency `[P, P]`.
    """
    pooled_size = len(subgraphs)
    pooled_adjacency = torch.zeros((pooled_size, pooled_size), dtype=torch.bool, device=adjacency.device)
    for row_idx, row_nodes in enumerate(subgraphs):
        for col_idx, col_nodes in enumerate(subgraphs):
            pooled_adjacency[row_idx, col_idx] = adjacency[row_nodes][:, col_nodes].any()
    pooled_adjacency.fill_diagonal_(True)
    return pooled_adjacency


def adjacency_pool(adjacency, pool_seed):
    """
    Split the graph into non-overlapping cliques and build the pooled adjacency.

    Args:
        adjacency: torch.Tensor
            Boolean adjacency `[N, N]`.
        pool_seed: int
            Deterministic seed for the random clique construction.
    return:
        tuple
            `(pooled_adjacency, subgraphs)`.
    """
    adjacency_cpu = adjacency.detach().to("cpu").bool()
    reduced = adjacency_cpu.clone()
    remaining = torch.arange(adjacency_cpu.shape[0], dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(pool_seed))
    subgraphs = []
    while remaining.numel() > 0:
        random_idx = torch.randint(remaining.numel(), (1,), generator=generator).item()
        seed_node = remaining[random_idx]
        clique = torch.tensor([seed_node.item()], dtype=torch.long)
        connected = torch.nonzero(reduced[seed_node], as_tuple=False).flatten()
        connected = connected[connected != seed_node]
        for node in connected:
            if torch.isin(node, remaining) and adjacency_cpu[clique, node].all():
                clique = torch.cat((clique, node.reshape(1)))
        reduced[:, clique] = False
        reduced[clique, :] = False
        subgraphs.append(clique.to(adjacency.device))
        keep_mask = ~torch.isin(remaining, clique)
        remaining = remaining[keep_mask]
    pooled_adjacency = new_adjacency(subgraphs, adjacency)
    return pooled_adjacency, subgraphs


def pooled_adjacencies(adjacency, num_poolings, pool_seed):
    """
    Generate pooled adjacency hierarchy and saved subgraphs for unpooling.

    Args:
        adjacency: torch.Tensor
            Finest-level adjacency `[N, N]`.
        num_poolings: int
            Number of coarsening operations.
        pool_seed: int
            Seed for first pooling level; later levels use incremented seeds.
    return:
        tuple
            `(adjacency_list, subgraph_list)`.
    """
    adjacency_list = [adjacency]
    subgraph_list = []
    for level in range(num_poolings):
        pooled_adjacency, subgraphs = adjacency_pool(adjacency_list[-1], int(pool_seed) + level)
        if pooled_adjacency.shape[0] == adjacency_list[-1].shape[0]:
            raise ValueError("Pooling did not reduce the graph; reduce num_poolings or increase adjacency window.")
        adjacency_list.append(pooled_adjacency)
        subgraph_list.append(subgraphs)
    return adjacency_list, subgraph_list
