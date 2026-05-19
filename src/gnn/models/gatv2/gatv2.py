"""
gatv2.py

GATv2Net model: encoder (MLP embedding of nodes and edges) - processor (stacked GATv2Conv attention layers) -
decoder (two-head MLP for acceleration and von Mises stress prediction).

Based on "How Attentive are Graph Attention Networks?" (Brody et al., 2021) https://arxiv.org/abs/2105.14491.
Compared to standard GAT, GATv2 computes dynamic attention where every node can attend to any other node,
fixing the static attention problem.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
from src.gnn.models.models_helpers import build_mlp_d


class GATv2Processor(nn.Module):
    """
    Processor made of L stacked GATv2Conv layers. Each layer optionally applies a residual connection
    and layer normalisation after the attention aggregation.
    """

    hidden_dim: int
    num_layers: int
    convs: nn.ModuleList
    layer_norms: nn.ModuleList
    layer_norm_enable: bool
    residual_skip: bool

    def __init__(self, hidden_dim, num_layers, heads, dropout_attn, negative_slope,
                 add_self_loops, share_weights, residual_skip, layer_norm_enable):
        """
        Args:
            hidden_dim: int
                Latent dimension H. Input and output of every GATv2Conv layer.
            num_layers: int
                Number of stacked GATv2Conv layers.
            heads: int
                Number of attention heads (concat=False so output stays at hidden_dim).
            dropout_attn: float
                Dropout on normalised attention coefficients.
            negative_slope: float
                LeakyReLU angle for the attention scoring function.
            add_self_loops: bool
                Whether to add self-loops to edge_index inside GATv2Conv.
            share_weights: bool
                If True, source and target linear transforms share the same weight matrix.
            residual_skip: bool
                If True, adds a skip connection (input + output) per layer.
            layer_norm_enable: bool
                If True, applies LayerNorm after each GATv2Conv layer.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm_enable = layer_norm_enable
        self.residual_skip = residual_skip
        # concat=False averages multi-head outputs → output dim == hidden_dim
        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                negative_slope=negative_slope,
                dropout=dropout_attn,
                add_self_loops=add_self_loops,
                edge_dim=hidden_dim,
                share_weights=share_weights,
            )
            for _ in range(num_layers)
        ])
        if layer_norm_enable:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        else:
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def forward(self, node_embedding, edge_index, edge_embedding):
        """
        Args:
            node_embedding: torch.Tensor
                Shape [N, H].
            edge_index: torch.Tensor
                Shape [2, E].
            edge_embedding: torch.Tensor
                Shape [E, H].
        return: torch.Tensor
            Updated node embedding of shape [N, H].
        """
        x = node_embedding
        for conv, norm in zip(self.convs, self.layer_norms):
            out = conv(x, edge_index, edge_attr=edge_embedding)
            if self.residual_skip:
                out = out + x
            x = norm(out)
        return x


class GATv2Net(nn.Module):
    """
    GATv2Net Model.

    Made by:
    1) 1 encoder/embedder: two MLPs, one for node features [N, C_n] → [N, H] and one for edge features [E, C_e] → [E, H]
    2) L GATv2Conv attention layers acting on the embedded node and edge representations
    3) One final decoder/head: two MLPs, one for acceleration (out=3) and one for von Mises stress (out=1)

    The edge embeddings produced by the encoder are passed as edge_attr to every GATv2Conv layer so that
    attention scores and aggregated messages are conditioned on edge features throughout the processor.
    """

    encoder_node: nn.Module
    encoder_edge: nn.Module
    processor: GATv2Processor
    acc_head: nn.Module
    mises_head: nn.Module

    def __init__(self, node_channels, edge_channels, hidden_dim, num_layers, heads,
                 dropout_attn, negative_slope, add_self_loops, share_weights, residual,
                 layer_norm_encoder, layer_norm_processor, layer_norm_decoder,
                 num_layers_encoder, num_layers_heads, dropout_encoder, dropout_head, stress_out_dim):
        """
        Args:
            node_channels: int
                Input feature dimension for nodes (C_n).
            edge_channels: int
                Input feature dimension for edges (C_e).
            hidden_dim: int
                Latent embedding dimension H used throughout encoder, processor, and decoder.
            num_layers: int
                Number of GATv2Conv message-passing layers in the processor.
            heads: int
                Number of attention heads per GATv2Conv layer.
            dropout_attn: float
                Attention coefficient dropout probability.
            negative_slope: float
                LeakyReLU negative slope for attention scoring.
            add_self_loops: bool
                Whether GATv2Conv adds self-loops to edge_index.
            share_weights: bool
                Whether source and target linear projections share weights inside GATv2Conv.
            residual: bool
                Whether to add a residual skip connection per processor layer (separate from the
                built-in GATv2Conv residual flag, which we leave False).
            layer_norm_encoder: bool
                LayerNorm in encoder MLPs.
            layer_norm_processor: bool
                LayerNorm after each GATv2Conv layer.
            layer_norm_decoder: bool
                LayerNorm in decoder head MLPs.
            num_layers_encoder: int
                Number of layers in the encoder MLPs.
            num_layers_heads: int
                Number of layers in the decoder head MLPs.
            dropout_encoder: float
                Dropout in encoder MLPs.
            dropout_head: float
                Dropout in decoder head MLPs.
        """
        super().__init__()
        # Node encoder: [N, C_n] --> [N, H]
        self.encoder_node = build_mlp_d(
            in_size=node_channels, hidden_size=hidden_dim, out_size=hidden_dim,
            num_layers=num_layers_encoder, layer_norm=layer_norm_encoder,
            use_sigmoid=False, use_softmax=False, dropout=dropout_encoder,
        )
        # Edge encoder: [E, C_e] --> [E, H]
        self.encoder_edge = build_mlp_d(
            in_size=edge_channels, hidden_size=hidden_dim, out_size=hidden_dim,
            num_layers=num_layers_encoder, layer_norm=layer_norm_encoder,
            use_sigmoid=False, use_softmax=False, dropout=dropout_encoder,
        )
        # Processor: L x GATv2Conv layers
        self.processor = GATv2Processor(
            hidden_dim=hidden_dim, num_layers=num_layers, heads=heads,
            dropout_attn=dropout_attn, negative_slope=negative_slope,
            add_self_loops=add_self_loops, share_weights=share_weights,
            residual_skip=residual, layer_norm_enable=layer_norm_processor,
        )
        # Decoder heads: [N, H] --> [N, 3] and [N, H] --> [N, 1]
        self.acc_head = build_mlp_d(
            in_size=hidden_dim, hidden_size=hidden_dim, out_size=3,
            num_layers=num_layers_heads, layer_norm=layer_norm_decoder,
            use_sigmoid=False, use_softmax=False, dropout=dropout_head,
        )
        # mises_head keeps its historical name for backward compat; its width depends on stress_out_dim
        # (1 = von Mises scalar, 6 = sig6 stress tensor).
        self.mises_head = build_mlp_d(
            in_size=hidden_dim, hidden_size=hidden_dim, out_size=int(stress_out_dim),
            num_layers=num_layers_heads, layer_norm=layer_norm_decoder,
            use_sigmoid=False, use_softmax=False, dropout=dropout_head,
        )

    def forward(self, ingraph):
        """
        Args:
            ingraph: PyG Data or Batch
                Must contain x [N, C_n], edge_attr [E, C_e], edge_index [2, E].
        return: torch.Tensor
            Predictions of shape [N, 4]: first 3 columns are acceleration, last column is von Mises stress.
        """
        node_features = ingraph.x
        edge_features = ingraph.edge_attr
        edge_index = ingraph.edge_index
        # Encode raw features to latent dimension H
        node_embedding = self.encoder_node(node_features)   # [N, H]
        edge_embedding = self.encoder_edge(edge_features)   # [E, H]
        # Apply L GATv2Conv attention layers
        node_embedding = self.processor(node_embedding, edge_index, edge_embedding)  # [N, H]
        # Decode to predictions
        acc = self.acc_head(node_embedding)         # [N, 3]
        stress = self.mises_head(node_embedding)    # [N, stress_out_dim]
        return torch.hstack((acc, stress))          # [N, 3 + stress_out_dim]
