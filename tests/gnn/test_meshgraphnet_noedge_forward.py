"""
test_meshgraphnet_noedge_forward.py

Smoke tests for the MeshGraphNet variant that uses topology and node features but does not read edge_attr.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNetNoEdge
from src.gnn.train import create_model


def _ring_edge_index(num_nodes):
    """
    Build an undirected ring edge_index.

    Args:
        num_nodes: int
            Number of nodes in the ring.
        return: edge_index
            Edge index tensor with shape [2, 2 * num_nodes].
    """
    row = torch.arange(num_nodes, dtype=torch.long)
    col = (row + 1) % num_nodes
    senders = torch.cat([row, col], dim=0)
    receivers = torch.cat([col, row], dim=0)
    return torch.stack([senders, receivers], dim=0)


def _make_noedge_mgn():
    """
    Construct a small MeshGraphNetNoEdge for forward smoke tests.

    Args:
        return: model
            Small MeshGraphNetNoEdge instance.
    """
    return MeshGraphNetNoEdge(node_channels=5, encoder_latent_size=8, encoder_decoder_latent_size=8,
                              processor_latent_size=8, num_msgs=2, num_layers_graphnetblock=2,
                              layer_norm_enable_encoder=True, layer_norm_enable_decoder=False,
                              layer_norm_enable_processor=True, num_layers_heads=2, num_layers_encoder=2,
                              dropout_encoder=0.0, dropout_head=0.0, dropout_processor=0.0,
                              activation_checkpointing=False, stress_out_dim=1)


def _make_train_cfg_noedge():
    """
    Build the minimal TrainYaml-like object consumed by create_model for MGN_no_edge.

    Args:
        return: train_cfg
            Namespace with the MGN_no_edge model hyperparameters.
    """
    return SimpleNamespace(
        model_name="MGN_no_edge",
        hidden_processor_dim=8,
        num_messages=2,
        num_layers_graphnetblock=2,
        layer_norm_enable_encoder=True,
        layer_norm_enable_decoder=False,
        layer_norm_enable_processor=True,
        num_layers_heads=2,
        num_layers_encoder=2,
        dropout_encoder=0.0,
        dropout_head=0.0,
        dropout_processor=0.0,
        activation_checkpointing=False,
    )


def test_meshgraphnet_noedge_forward_does_not_require_edge_attr():
    """
    A graph with x and edge_index but no edge_attr produces one prediction row per node.

    Args:
        return: None
            Asserts the no-edge forward shape and finite values.
    """
    num_nodes = 12
    graph = Data(x=torch.randn(num_nodes, 5), edge_index=_ring_edge_index(num_nodes))
    model = _make_noedge_mgn()
    model.eval()
    with torch.no_grad():
        pred = model(graph)
    assert pred.shape == (num_nodes, 4)
    assert torch.isfinite(pred).all()


def test_create_model_builds_meshgraphnet_noedge():
    """
    create_model maps model_name='MGN_no_edge' to MeshGraphNetNoEdge and the model ignores edge_attr.

    Args:
        return: None
            Asserts factory selection and forward output width.
    """
    num_nodes = 10
    preprocess_cfg = SimpleNamespace(node_channels=5, edge_channels=99)
    train_cfg = _make_train_cfg_noedge()
    graph = Data(x=torch.randn(num_nodes, 5), edge_index=_ring_edge_index(num_nodes))
    model = create_model(train_cfg, preprocess_cfg, torch.device("cpu"), graph, 6)
    assert isinstance(model, MeshGraphNetNoEdge)
    model.eval()
    with torch.no_grad():
        pred = model(graph)
    assert pred.shape == (num_nodes, 9)
