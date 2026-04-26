"""
test_gunet_forward.py

Smoke tests for GUNet forward pass (PyG graph in, [N, 4] predictions out).
"""

import pytest
import torch

pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data

import torch.nn as nn

from src.gnn.models.gunet.gunet import GUNet


class _StubGraphUNet(nn.Module):
    """
    Fake U-Net segment returning a fixed-width embedding so decoder heads run without GraphUNet sparse ops.
    """

    hidden_channels: int

    def __init__(self, hidden_channels):
        """
        Args:
            hidden_channels: int
                Output feature width (matches GUNet hidden_channels / head input).
        return:
            None
        """
        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: torch.Tensor
                Node features [N, *].
            edge_index: torch.Tensor
                Unused; same signature as GraphUNet.forward.
            batch: torch.Tensor or None
                Unused batch vector.
        return:
            torch.Tensor
                Zeros with shape [N, hidden_channels].
        """
        return torch.zeros(x.size(0), self.hidden_channels, dtype=x.dtype, device=x.device)


def _device_for_graph_unet_smoke():
    """
    GraphUNet uses sparse CSR adjacency tricks that can fail on CPU without MKL.

    Args:
        None
    return:
        torch.device Prefer CUDA when available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _forward_or_skip(model, graph):
    """
    Run GUNet forward on graph; skip test if this host's CPU PyTorch lacks MKL sparse support.

    Args:
        model: GUNet
            Model under test.
        graph: Data or Batch
            PyG input graph.
    return:
        torch.Tensor Predictions [N, 4].
    """
    device = _device_for_graph_unet_smoke()
    model = model.to(device)
    graph = graph.to(device)
    model.eval()
    try:
        with torch.no_grad():
            return model(graph)
    except RuntimeError as err:
        msg = str(err)
        if device.type == "cpu" and ("MKL" in msg or "SparseCsr" in msg):
            pytest.skip("GraphUNet adjacency augmentation not supported on this CPU PyTorch build.")
        raise


def _ring_edge_index(num_nodes):
    """
    Build an undirected ring edge_index [2, E] for num_nodes >= 3.

    Args:
        num_nodes: int
            Number of nodes in the ring.
    return:
        edge_index tensor [2, 2 * num_nodes]
    """
    row = torch.arange(num_nodes, dtype=torch.long)
    col = (row + 1) % num_nodes
    senders = torch.cat([row, col], dim=0)
    receivers = torch.cat([col, row], dim=0)
    return torch.stack([senders, receivers], dim=0)


def test_gunet_single_graph_output_shape():
    """
    One PyG Data graph: output rows match num_nodes and last dim is 4.
    """
    num_nodes = 64
    in_channels = 8
    hidden_channels = 32
    depth = 2
    pool_ratios = [0.6, 0.5]
    x = torch.randn(num_nodes, in_channels)
    edge_index = _ring_edge_index(num_nodes)
    graph = Data(x=x, edge_index=edge_index)
    dropout_encoder = 0.0
    dropout_head = 0.05
    model = GUNet(in_channels, hidden_channels, True, depth, pool_ratios, True, True,
                  2, 2, dropout_encoder=dropout_encoder, dropout_head=dropout_head)
    pred = _forward_or_skip(model, graph)
    assert pred.shape == (num_nodes, 4)


def test_gunet_batched_output_shape():
    """
    Batched graphs: output rows match total nodes; last dim is 4.
    """
    in_channels = 6
    hidden_channels = 24
    depth = 2
    pool_ratios = [0.55, 0.5]
    sizes = [40, 52]
    graphs = []
    for n in sizes:
        graphs.append(Data(x=torch.randn(n, in_channels), edge_index=_ring_edge_index(n),))
    batch = Batch.from_data_list(graphs)
    model = GUNet(in_channels, hidden_channels,True, depth, pool_ratios, True,
                  False, 2, 2,)
    pred = _forward_or_skip(model, batch)
    assert pred.shape == (sum(sizes), 4)


def test_gunet_pyg_interface_and_output_dim_with_stub_unet():
    """
    CPU-safe: replaces GraphUNet with a stub to assert Data in -> [N, 4] out (trainer contract).
    """
    num_nodes = 12
    in_channels = 5
    hidden_channels = 11
    dropout_encoder = 0.0
    dropout_head = 0.05
    graph = Data(x=torch.randn(num_nodes, in_channels), edge_index=_ring_edge_index(num_nodes),)
    model = GUNet(in_channels, hidden_channels,  False,  1,  [0.5],  True,
                  False,  2,  2, dropout_encoder=dropout_encoder,
                  dropout_head=dropout_head)
    model.graph_unet = _StubGraphUNet(hidden_channels)
    pred = model(graph)
    assert pred.shape == (num_nodes, 4)
