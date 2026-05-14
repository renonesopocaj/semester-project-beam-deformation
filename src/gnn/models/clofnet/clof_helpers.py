"""
clof_helpers.py

Defines helper functions for ClofNet.
"""

def split_edge_features(edge_attr, split_idx_edge):
    """
    Split edge attribute tensor into stacked R^3-valued chunks and remaining scalars.

    Args:
        edge_attr: torch.Tensor
            Edge features of shape [E, C_e].
        split_idx_edge: int
            Leading width of vector triplets (must be divisible by 3).
    return: (vec_chunks, scalars)
        vec_chunks: torch.Tensor [E, k, 3] with k = split_idx_edge // 3
        scalars: torch.Tensor [E, C_e - split_idx_edge]
    """
    vec_flat = edge_attr[:, :split_idx_edge]
    scalars = edge_attr[:, split_idx_edge:]
    num_triplets = split_idx_edge // 3
    vec_chunks = vec_flat.reshape(edge_attr.shape[0], num_triplets, 3)
    return vec_chunks, scalars

def split_node_features(x, split_idx_node):
    """
    Split PyG node tensor into boundary flag, stacked R^3-valued chunks, and remaining scalars.

    Args:
        x: torch.Tensor
            Full node features of shape [N, C] with layout [bc | vec (3k) | scalars].
        split_idx_node: int
            Width of the vector-valued block (must be divisible by 3); starts at column index 1 after bc.
    return: (bc, vec_chunks, scalars)
        bc: torch.Tensor [N, 1]
        vec_chunks: torch.Tensor [N, k, 3] with k = split_idx_node // 3
        scalars: torch.Tensor [N, C - 1 - split_idx_node]
    """
    bc = x[:, 0:1]
    vec_flat = x[:, 1 : 1 + split_idx_node]
    scalars = x[:, 1 + split_idx_node :]
    num_triplets = split_idx_node // 3
    vec_chunks = vec_flat.reshape(x.shape[0], num_triplets, 3)
    return bc, vec_chunks, scalars