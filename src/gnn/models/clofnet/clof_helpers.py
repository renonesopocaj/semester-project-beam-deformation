"""
clof_helpers.py

Defines helper functions for ClofNet.
"""

import torch
from torch import nn


def make_mlp(n_layers, in_dim, hidden_dim, out_dim, act_fn_factory, final_activation, mlp_layer_norm):
    """
    Build a sequential MLP with n_layers linear layers.
    When mlp_layer_norm is True, each hidden block becomes Linear → LayerNorm → Activation.
    The final (output) layer never has a LayerNorm; it gets an activation only if final_activation is True.

    Args:
        n_layers: int
            Total number of Linear layers.
        in_dim: int
            Input dimension of the first layer.
        hidden_dim: int
            Width of all intermediate layers.
        out_dim: int
            Output dimension of the last layer.
        act_fn_factory: callable
            Zero-argument factory returning an activation module.
        final_activation: bool
            Whether to append an activation after the last linear layer.
        mlp_layer_norm: bool
            Whether to insert LayerNorm after each hidden Linear (before activation).
    return: nn.Sequential
    """
    layers = []
    for i in range(n_layers):
        d_in = in_dim if i == 0 else hidden_dim
        d_out = out_dim if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            if mlp_layer_norm:
                layers.append(nn.LayerNorm(d_out))
            layers.append(act_fn_factory())
        elif final_activation:
            layers.append(act_fn_factory())
    return nn.Sequential(*layers)


def make_vec_mlp(hidden_dim, act_fn_factory, tanh, mlp_layer_norm):
    """
    Build the vectorization MLP shared by all MP block variants:
        Linear(H, H) → [LayerNorm(H)] → act → Linear(H, 3, bias=False) → [Tanh]
    The final Linear is xavier-initialized with gain 0.001.

    Args:
        hidden_dim: int
        act_fn_factory: callable
        tanh: bool
            Whether to append a Tanh at the end.
        mlp_layer_norm: bool
            Whether to insert LayerNorm after the first hidden Linear.
    return: nn.Sequential
    """
    linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
    torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
    layers = [nn.Linear(hidden_dim, hidden_dim)]
    if mlp_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(act_fn_factory())
    layers.append(linlayer_final)
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


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