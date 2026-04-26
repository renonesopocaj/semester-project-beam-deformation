"""
Smoke tests for MAGNet forward pass (PyG graph in, [N, 4] predictions out).
"""

import pytest
import torch

pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data

from src.gnn.models.mag.g_pool import GPool
from src.gnn.models.mag.g_unpool import GUnpool
from src.gnn.models.mag.mag import MAgLayer, MAGNet


def _ring_edge_index(num_nodes):
    """
    Build an undirected ring edge_index [2, E] for num_nodes >= 3.

    Args:
        num_nodes: int
            Number of nodes in the ring.
    return:
        torch.Tensor
            edge_index [2, 2 * num_nodes]
    """
    row = torch.arange(num_nodes, dtype=torch.long)
    col = (row + 1) % num_nodes
    senders = torch.cat([row, col], dim=0)
    receivers = torch.cat([col, row], dim=0)
    return torch.stack([senders, receivers], dim=0)


def test_mag_layer_matches_node_specific_aggregation_formula():
    """
    MAgLayer computes one trainable kernel per target/source/channel tuple selected by adjacency.

    Args:
        None
    return:
        None
    """
    adjacency = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    batch_size = 2
    in_channels = 2
    out_channels = 3
    x = torch.randn(batch_size, adjacency.shape[0], in_channels)
    layer = MAgLayer(in_channels, out_channels, adjacency, "linear", True)
    with torch.no_grad():
        layer.kernel.copy_(torch.randn_like(layer.kernel))
        layer.bias.copy_(torch.randn_like(layer.bias))
    expected = torch.zeros(batch_size, adjacency.shape[0], out_channels)
    for edge_idx in range(layer.kernel.shape[0]):
        target = layer.target_nodes[edge_idx]
        source = layer.source_nodes[edge_idx]
        expected[:, target, :] += x[:, source, :] @ layer.kernel[edge_idx]
    expected = expected + layer.bias.unsqueeze(0)
    actual = layer(x)
    assert torch.allclose(actual, expected)


def test_pool_and_unpool_match_clique_operations():
    """
    GPool reduces clique features per channel and GUnpool replicates pooled features to clique nodes.

    Args:
        None
    return:
        None
    """
    subgraphs = [torch.tensor([0, 2]), torch.tensor([1])]
    x = torch.tensor([[[1.0, 4.0], [5.0, 2.0], [3.0, 8.0]]])
    pooled = GPool(subgraphs, "max")(x)
    expected_pooled = torch.tensor([[[3.0, 8.0], [5.0, 2.0]]])
    assert torch.equal(pooled, expected_pooled)
    unpooled = GUnpool(subgraphs, 3)(pooled)
    expected_unpooled = torch.tensor([[[3.0, 8.0], [5.0, 2.0], [3.0, 8.0]]])
    assert torch.equal(unpooled, expected_unpooled)


def test_magnet_single_graph_no_edge_attr():
    """
    One PyG Data graph without edge_attr: output rows match num_nodes and last dim is 4.

    Args:
        None
    return:
        None
    """
    num_nodes = 32
    node_channels = 7
    edge_channels = 0
    output_channels = 4
    hidden_channels = [12, 18]
    num_poolings = 1
    layers_per_level = 1
    window_size = 2
    pool_seed = 10
    pool_aggr = "max"
    activation = "leakyrelu"
    x = torch.randn(num_nodes, node_channels)
    edge_index = _ring_edge_index(num_nodes)
    graph = Data(x=x, edge_index=edge_index)
    model = MAGNet(
        node_channels,
        edge_channels,
        output_channels,
        hidden_channels,
        num_poolings,
        layers_per_level,
        window_size,
        pool_seed,
        pool_aggr,
        activation,
        graph,
    )
    model.eval()
    with torch.no_grad():
        pred = model(graph)
    assert pred.shape == (num_nodes, 4)


def test_magnet_batched_fixed_topology():
    """
    Batched fixed-topology graphs: output rows match total nodes; last dim is 4.

    Args:
        None
    return:
        None
    """
    node_channels = 6
    edge_channels = 0
    output_channels = 4
    hidden_channels = [10, 16]
    num_poolings = 1
    layers_per_level = 2
    window_size = 2
    pool_seed = 2
    pool_aggr = "mean"
    activation = "relu"
    sizes = [28, 28]
    graphs = []
    for n in sizes:
        edge_index = _ring_edge_index(n)
        graphs.append(
            Data(
                x=torch.randn(n, node_channels),
                edge_index=edge_index,
            )
        )
    batch = Batch.from_data_list(graphs)
    model = MAGNet(
        node_channels,
        edge_channels,
        output_channels,
        hidden_channels,
        num_poolings,
        layers_per_level,
        window_size,
        pool_seed,
        pool_aggr,
        activation,
        batch,
    )
    model.eval()
    with torch.no_grad():
        pred = model(batch)
    assert pred.shape == (sum(sizes), 4)


def test_magnet_has_pool_unpool_layers():
    """
    MAGNet builds the graph U-Net pool and unpool hierarchy.

    Args:
        None
    return:
        None
    """
    num_nodes = 16
    node_channels = 3
    edge_channels = 0
    output_channels = 4
    hidden_channels = [8, 12]
    num_poolings = 1
    layers_per_level = 1
    window_size = 2
    pool_seed = 1
    pool_aggr = "max"
    activation = "leakyrelu"
    graph = Data(
        x=torch.randn(num_nodes, node_channels),
        edge_index=_ring_edge_index(num_nodes),
    )
    model = MAGNet(
        node_channels,
        edge_channels,
        output_channels,
        hidden_channels,
        num_poolings,
        layers_per_level,
        window_size,
        pool_seed,
        pool_aggr,
        activation,
        graph,
    )
    assert len(model.pool_layers) == num_poolings
    assert len(model.unpool_layers) == num_poolings


def test_magnet_rejects_variable_size_batch():
    """
    True MAg kernels are node-specific, so one model expects one fixed graph topology.

    Args:
        None
    return:
        None
    """
    node_channels = 3
    graph = Data(x=torch.randn(12, node_channels), edge_index=_ring_edge_index(12))
    other = Data(x=torch.randn(14, node_channels), edge_index=_ring_edge_index(14))
    model = MAGNet(node_channels, 0, 4, [8, 12], 1, 1, 2, 1, "max", "leakyrelu", graph)
    batch = Batch.from_data_list([graph, other])
    with pytest.raises(ValueError):
        model(batch)
