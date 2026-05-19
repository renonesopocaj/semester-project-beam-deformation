"""
gatmgn.py

GatMGNNet model: encoder - processor (interleaved MGN and GATv2 message-passing layers) - decoder.

The processor alternates one MGN GraphNetBlock (which updates both node and edge embeddings via
scatter-add aggregation) with one GATv2Conv layer (which updates only node embeddings using the
latest edge embedding as attention-conditioning edge_attr). This gives the model both the
edge-state tracking of MeshGraphNet and the dynamic attention of GATv2.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
from src.gnn.models.meshgraphnet.meshgraphnet import Encoder, GraphNetBlock, Decoder


class GatMGNProcessor(nn.Module):
    """
    Processor that interleaves MGN GraphNetBlock and GATv2Conv layers.

    Each step i consists of:
    1. GraphNetBlock: updates node_embedding and edge_embedding via scatter-add aggregation.
    2. GATv2Conv: updates node_embedding using the freshly updated edge_embedding as edge_attr.
       An optional residual skip and LayerNorm are applied after the GATv2Conv output.
    """

    hidden_dim: int
    num_layers: int
    mgn_blocks: nn.ModuleList
    gat_convs: nn.ModuleList
    gat_layer_norms: nn.ModuleList
    residual_skip: bool

    def __init__(self, hidden_dim, num_layers, num_layers_graphnetblock, heads, dropout_attn,
                 negative_slope, add_self_loops, share_weights, residual_skip, layer_norm_enable,
                 dropout_processor):
        """
        Args:
            hidden_dim: int
                Latent dimension H shared by all layers.
            num_layers: int
                Number of (GraphNetBlock, GATv2Conv) interleaved pairs.
            num_layers_graphnetblock: int
                Number of MLP layers inside each GraphNetBlock.
            heads: int
                Number of attention heads for GATv2Conv (concat=False keeps output at H).
            dropout_attn: float
                Dropout on attention coefficients inside GATv2Conv.
            negative_slope: float
                LeakyReLU negative slope for GATv2Conv attention scoring.
            add_self_loops: bool
                Whether GATv2Conv adds self-loops to edge_index.
            share_weights: bool
                Whether GATv2Conv shares source and target weight matrices.
            residual_skip: bool
                If True, adds node_embedding + GATv2Conv(node_embedding) per GATv2 step.
            layer_norm_enable: bool
                If True, applies LayerNorm after each GATv2Conv step.
            dropout_processor: float
                Dropout inside each GraphNetBlock MLP.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual_skip = residual_skip
        # MGN blocks: edge MLP input is [h_i, h_j, e_ij] = [3H], node MLP input is [agg, h_i] = [2H]
        self.mgn_blocks = nn.ModuleList([
            GraphNetBlock(
                in_size_edge_MLP=3 * hidden_dim,
                in_size_node_MLP=2 * hidden_dim,
                latent_and_out_size=hidden_dim,
                num_layers_graphnetblock=num_layers_graphnetblock,
                enable_layer_norm=layer_norm_enable,
                dropout=dropout_processor,
            )
            for _ in range(num_layers)
        ])
        # GATv2Conv layers: concat=False so output stays at hidden_dim
        self.gat_convs = nn.ModuleList([
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
            self.gat_layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        else:
            self.gat_layer_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def forward(self, node_embedding, edge_index, edge_embedding):
        """
        Args:
            node_embedding: torch.Tensor
                Shape [N, H].
            edge_index: torch.Tensor
                Shape [2, E].
            edge_embedding: torch.Tensor
                Shape [E, H].
        return: (node_embedding, edge_embedding)
            Updated tensors of shape [N, H] and [E, H].
        """
        for mgn_block, gat_conv, gat_norm in zip(self.mgn_blocks, self.gat_convs, self.gat_layer_norms):
            # MGN step: update both node and edge embeddings
            node_embedding, edge_embedding = mgn_block(edge_index, node_embedding, edge_embedding)
            # GATv2 step: update only node embedding, conditioned on the freshly updated edge_embedding
            gat_out = gat_conv(node_embedding, edge_index, edge_attr=edge_embedding)
            if self.residual_skip:
                gat_out = gat_out + node_embedding
            node_embedding = gat_norm(gat_out)
        return node_embedding, edge_embedding


class GatMGNNet(nn.Module):
    """
    GatMGNNet Model.

    Made by:
    1) Encoder: two MLPs embedding node features [N, C_n] → [N, H] and edge features [E, C_e] → [E, H].
    2) Processor: L interleaved pairs of (GraphNetBlock, GATv2Conv).
       Each GraphNetBlock updates both node and edge embeddings; the subsequent GATv2Conv layer
       refines node embeddings using the updated edge embedding as attention context.
    3) Decoder: two head MLPs predicting acceleration [N, 3] and von Mises stress [N, 1],
       concatenated to [N, 4].
    """

    encoder: nn.Module
    processor: GatMGNProcessor
    decoder: nn.Module

    def __init__(self, node_channels, edge_channels, hidden_dim, num_layers, num_layers_graphnetblock,
                 heads, dropout_attn, negative_slope, add_self_loops, share_weights, residual,
                 layer_norm_encoder, layer_norm_processor, layer_norm_decoder,
                 num_layers_encoder, num_layers_heads, dropout_encoder, dropout_head, dropout_processor,
                 stress_out_dim):
        """
        Args:
            node_channels: int
                Input feature dimension for nodes (C_n).
            edge_channels: int
                Input feature dimension for edges (C_e).
            hidden_dim: int
                Latent embedding dimension H used throughout encoder, processor, and decoder.
            num_layers: int
                Number of interleaved (GraphNetBlock, GATv2Conv) pairs in the processor.
            num_layers_graphnetblock: int
                Number of MLP layers inside each GraphNetBlock.
            heads: int
                Number of attention heads per GATv2Conv layer.
            dropout_attn: float
                Attention coefficient dropout probability.
            negative_slope: float
                LeakyReLU negative slope for GATv2Conv attention scoring.
            add_self_loops: bool
                Whether GATv2Conv adds self-loops to edge_index.
            share_weights: bool
                Whether source and target linear projections share weights inside GATv2Conv.
            residual: bool
                Whether to add a residual skip connection after each GATv2Conv step.
            layer_norm_encoder: bool
                LayerNorm in encoder MLPs.
            layer_norm_processor: bool
                LayerNorm after each GATv2Conv step and inside each GraphNetBlock.
            layer_norm_decoder: bool
                LayerNorm in decoder head MLPs.
            num_layers_encoder: int
                Number of layers in encoder MLPs.
            num_layers_heads: int
                Number of layers in decoder head MLPs.
            dropout_encoder: float
                Dropout in encoder MLPs.
            dropout_head: float
                Dropout in decoder head MLPs.
            dropout_processor: float
                Dropout inside GraphNetBlock MLPs.
        """
        super().__init__()
        self.encoder = Encoder(
            node_channels=node_channels,
            edge_channels=edge_channels,
            latent_and_out_size=hidden_dim,
            layer_norm_enable=layer_norm_encoder,
            num_layers_encoder=num_layers_encoder,
            dropout=dropout_encoder,
        )
        self.processor = GatMGNProcessor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_layers_graphnetblock=num_layers_graphnetblock,
            heads=heads,
            dropout_attn=dropout_attn,
            negative_slope=negative_slope,
            add_self_loops=add_self_loops,
            share_weights=share_weights,
            residual_skip=residual,
            layer_norm_enable=layer_norm_processor,
            dropout_processor=dropout_processor,
        )
        self.decoder = Decoder(
            in_and_latent_size=hidden_dim,
            enable_layer_norm=layer_norm_decoder,
            num_layers_heads=num_layers_heads,
            dropout=dropout_head,
            stress_out_dim=stress_out_dim,
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
        node_embedding, edge_embedding = self.encoder(node_features, edge_features)
        # Interleaved MGN + GATv2 processor
        node_embedding, _ = self.processor(node_embedding, edge_index, edge_embedding)
        # Decode to predictions
        return self.decoder(node_embedding)
