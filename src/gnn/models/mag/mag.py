"""
mag.py

PyTorch implementation of MAgNET's fixed-topology graph U-Net.
"""

import torch
import torch.nn as nn
from src.gnn.models.mag.g_pool import GPool
from src.gnn.models.mag.g_unpool import GUnpool
from src.gnn.models.mag.pool_utils import adjacency_window, edge_index_to_adjacency, pooled_adjacencies


class MAgLayer(nn.Module):
    """
    Multi-channel Aggregation layer with node-specific, edge-specific trainable kernels.
    """
    num_nodes: int
    in_channels: int
    out_channels: int
    activation: str
    use_bias: bool
    kernel: nn.Parameter
    bias: nn.Parameter

    def __init__(self, in_channels, out_channels, adjacency, activation, use_bias):
        """
        Args:
            in_channels: int
                Input channel count.
            out_channels: int
                Output channel count.
            adjacency: torch.Tensor
                Boolean adjacency `[N, N]`, row target and column source.
            activation: str
                `leakyrelu`, `relu`, or `linear`.
            use_bias: bool
                Whether to add node/channel-specific bias.
        """
        super().__init__()
        edge_nodes = torch.nonzero(adjacency.bool(), as_tuple=False)
        self.num_nodes = adjacency.shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.use_bias = use_bias
        self.register_buffer("target_nodes", edge_nodes[:, 0].long())
        self.register_buffer("source_nodes", edge_nodes[:, 1].long())
        self.kernel = nn.Parameter(torch.empty(edge_nodes.shape[0], in_channels, out_channels))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(self.num_nodes, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Args:
            None
        return:
            None
        """
        nn.init.xavier_uniform_(self.kernel.reshape(-1, self.out_channels))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                Dense features `[B, N, C_in]`.
        return:
            torch.Tensor
                Dense features `[B, N, C_out]`.
        """
        messages = torch.einsum("bec,eco->beo", x[:, self.source_nodes, :], self.kernel)
        out = torch.zeros((x.shape[0], self.num_nodes, self.out_channels), dtype=x.dtype, device=x.device)
        out.index_add_(1, self.target_nodes, messages)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        if self.activation == "linear":
            return out
        if self.activation == "leakyrelu":
            return torch.nn.functional.leaky_relu(out, negative_slope=0.2)
        if self.activation == "relu":
            return torch.relu(out)
        raise ValueError(f"Unknown activation '{self.activation}'.")


class MAgLevel(nn.Module):
    """
    One or more MAg layers at the same graph level.
    """

    layers: nn.ModuleList

    def __init__(self, in_channels, out_channels, adjacency, num_layers, activation):
        """
        Args:
            in_channels: int
                Input channels for the first layer.
            out_channels: int
                Output channels for all layers in this level.
            adjacency: torch.Tensor
                Boolean adjacency for this graph level.
            num_layers: int
                Number of MAg layers.
            activation: str
                Hidden activation.
        return:
            None
        """
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(MAgLayer(current_channels, out_channels, adjacency, activation, True))
            current_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                Dense features `[B, N, C]`.
        return:
            torch.Tensor
                Dense features after this level.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MAGNet(nn.Module):
    """
    MAgNET graph U-Net with MAg, clique gPool, gUnpool, and skip concatenation.
    """

    node_channels: int
    edge_channels: int
    output_channels: int
    num_nodes: int
    num_poolings: int
    encoder_levels: nn.ModuleList
    decoder_levels: nn.ModuleList
    pool_layers: nn.ModuleList
    unpool_layers: nn.ModuleList
    bottleneck: MAgLevel
    output_layer: MAgLayer

    def __init__(self, node_channels, output_channels, hidden_channels, num_poolings, layers_per_level,
                 window_size, pool_seed, pool_aggr, activation, graph_template):
        """
        Args:
            node_channels: int
                Raw node feature width.
            output_channels: int
                Node-wise prediction width.
            hidden_channels: int or list[int]
                Channel width per graph level. A list must have `num_poolings + 1` entries.
            num_poolings: int
                Number of encoder pooling operations.
            layers_per_level: int
                Number of MAg layers before/after each pooling and at the bottleneck.
            window_size: int
                Aggregation adjacency power, e.g. 2 for `A^2`.
            pool_seed: int
                Deterministic clique pooling seed.
            pool_aggr: str
                Pool aggregation (`max`, `mean`, `sum`, or `min`).
            activation: str
                Hidden activation (`leakyrelu` or `relu`).
            graph_template:
                PyG `Data`/`Batch` used to construct fixed MAg kernels and pooling hierarchy.
        """
        super().__init__()
        self.node_channels = node_channels
        self.output_channels = output_channels
        self.num_poolings = num_poolings
        self.channels = self._normalize_channels(hidden_channels, num_poolings)
        template_edge_index, self.num_nodes = self._extract_template_graph(graph_template)
        adjacency = edge_index_to_adjacency(template_edge_index, self.num_nodes, template_edge_index.device)
        adjacency_list, subgraph_list = pooled_adjacencies(adjacency, num_poolings, pool_seed)
        window_adjacencies = [adjacency_window(level_adjacency, window_size) for level_adjacency in adjacency_list]
        self.register_buffer("template_edge_index", template_edge_index.detach().clone().long())
        self.node_counts = [level_adjacency.shape[0] for level_adjacency in adjacency_list]
        self.encoder_levels = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_channels = node_channels
        for level in range(num_poolings):
            self.encoder_levels.append(MAgLevel(current_channels, self.channels[level], window_adjacencies[level], layers_per_level, activation))
            self.pool_layers.append(GPool(subgraph_list[level], pool_aggr))
            current_channels = self.channels[level]
        self.bottleneck = MAgLevel(current_channels, self.channels[-1], window_adjacencies[-1], layers_per_level, activation)
        current_channels = self.channels[-1]
        decoder_levels, unpool_layers = [], []
        for level in reversed(range(num_poolings)):
            unpool_layers.append(GUnpool(subgraph_list[level], self.node_counts[level]))
            decoder_levels.append(MAgLevel(current_channels + self.channels[level], self.channels[level], window_adjacencies[level], layers_per_level, activation))
            current_channels = self.channels[level]
        self.unpool_layers = nn.ModuleList(unpool_layers)
        self.decoder_levels = nn.ModuleList(decoder_levels)
        self.output_layer = MAgLayer(current_channels, output_channels, window_adjacencies[0], "linear", True)

    def _normalize_channels(self, hidden_channels, num_poolings):
        """
        Args:
            hidden_channels: int or list[int]
                Channel configuration.
            num_poolings: int
                Number of poolings.
        return:
            list[int]
                One width per encoder level plus bottleneck.
        """
        if isinstance(hidden_channels, int):
            return [hidden_channels for _ in range(num_poolings + 1)]
        channels = list(hidden_channels)
        if len(channels) != num_poolings + 1:
            raise ValueError("hidden_channels list length must equal num_poolings + 1.")
        return channels

    def _extract_template_graph(self, graph_template):
        """
        Args:
            graph_template:
                PyG `Data`/`Batch`.
        return:
            tuple
                `(edge_index, num_nodes)` for one fixed-topology graph.
        """
        if hasattr(graph_template, "ptr") and graph_template.ptr is not None:
            num_nodes = int((graph_template.ptr[1] - graph_template.ptr[0]).item())
            edge_mask = (graph_template.edge_index[0] < num_nodes) & (graph_template.edge_index[1] < num_nodes)
            return graph_template.edge_index[:, edge_mask].detach().cpu(), num_nodes
        return graph_template.edge_index.detach().cpu(), int(graph_template.x.shape[0])

    def _to_dense_fixed_batch(self, ingraph):
        """
        Args:
            ingraph:
                PyG `Data`/`Batch`.
        return:
            torch.Tensor
                Dense features `[B, N, C]`.
        """
        if hasattr(ingraph, "ptr") and ingraph.ptr is not None:
            node_counts = ingraph.ptr[1:] - ingraph.ptr[:-1]
            if not torch.all(node_counts == self.num_nodes):
                raise ValueError("MAGNet requires every batched graph to have the fixed template node count.")
            return ingraph.x.reshape(int(node_counts.numel()), self.num_nodes, ingraph.x.shape[1])
        if ingraph.x.shape[0] != self.num_nodes:
            raise ValueError("MAGNet input graph does not match the fixed template node count.")
        return ingraph.x.unsqueeze(0)

    def forward(self, ingraph):
        """
        Args:
            ingraph:
                PyG fixed-topology `Data`/`Batch` with node features.
        return:
            torch.Tensor
                Predictions `[N_total, 4]`.
        """
        x = self._to_dense_fixed_batch(ingraph)
        skips = []
        for level, pool in zip(self.encoder_levels, self.pool_layers):
            x = level(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for unpool, level, skip in zip(self.unpool_layers, self.decoder_levels, reversed(skips)):
            x = unpool(x)
            x = torch.cat((x, skip), dim=2)
            x = level(x)
        x = self.output_layer(x)
        return x.reshape(-1, self.output_channels)