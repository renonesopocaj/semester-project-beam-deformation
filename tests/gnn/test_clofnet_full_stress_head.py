"""
test_clofnet_full_stress_head.py

Run with:
    python -m pytest tests/gnn/test_clofnet_full_stress_head.py

Tests that CLOFNet uses the same stress-head inputs for von Mises and full-stress modes while changing
only the number of predicted stress channels.
"""

import torch
from torch import nn
from torch_geometric.data import Data

from src.gnn.models.clofnet.clofnet import ClofNetDirectAcceleration


NUM_NODES = 5
NODE_CHANNELS = 6
EDGE_CHANNELS = 5
SPLIT_IDX_NODE = 3
SPLIT_IDX_EDGE = 3
HIDDEN = 8
FUSE_HID = 8


def _silu_factory():
    return nn.SiLU()


def _make_direct_clofnet(stress_out_dim):
    """
    Build the smallest direct-acceleration CLOFNet variant needed to exercise the shared stress head.

    Args:
        stress_out_dim: int
            Number of stress channels emitted by stress_mlp: 1 for von Mises, 6 for full stress.
    return: ClofNetDirectAcceleration
        Model configured with fixed synthetic feature dimensions.
    """
    return ClofNetDirectAcceleration(
        in_dim_node_embed=NODE_CHANNELS - SPLIT_IDX_NODE,
        in_dim_edge_scalars=EDGE_CHANNELS - SPLIT_IDX_EDGE,
        hidden_dim=HIDDEN,
        fuse_hid_dim=FUSE_HID,
        act_fn_factory=_silu_factory,
        n_layers=1,
        coords_weight=1.0,
        recurrent=False,
        enable_norm_diff=True,
        tanh=False,
        attention_enable=False,
        clamp_enable=True,
        vel_mlp_dim_out=3,
        split_idx_node=SPLIT_IDX_NODE,
        split_idx_edge=SPLIT_IDX_EDGE,
        node_channels=NODE_CHANNELS,
        edge_channels=EDGE_CHANNELS,
        use_neighbor_diff_norms=False,
        frame_source="displacement",
        include_u_tm1=False,
        rescalarize_per_layer=False,
        include_v_branch=False,
        include_force=False,
        stress_out_dim=stress_out_dim,
        m_ij_init="zero",
        eps_mode="soft",
        n_layers_gate_mlp=1,
        n_layers_node_mlp=1,
        n_layers_message_mlp=1,
        normalize_delta_a=False,
        mlp_layer_norm=False,
    )


def _synthetic_batch():
    """
    Build a deterministic PyG graph with vector and scalar feature chunks matching the test model.

    return: Data
        Synthetic graph carrying the fields required by ClofNetBase.forward.
    """
    torch.manual_seed(0)
    row = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 0], dtype=torch.long)
    col = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3, 0, 4], dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    x_pos_t = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [2.0, 0.1, 0.4],
            [3.0, 0.3, 0.7],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rest_pos = x_pos_t - 0.1 * torch.randn(NUM_NODES, 3)
    return Data(
        x=torch.randn(NUM_NODES, NODE_CHANNELS),
        edge_attr=torch.randn(edge_index.shape[1], EDGE_CHANNELS),
        edge_index=edge_index,
        x_pos_t=x_pos_t,
        x_vel_t=torch.randn(NUM_NODES, 3),
        rest_pos=rest_pos,
        x_pos_tm1_prev=x_pos_t - 0.05 * torch.randn(NUM_NODES, 3),
        num_graphs=1,
    )


def test_clofnet_full_stress_head_keeps_von_mises_head_inputs():
    """
    Full-stress mode keeps the same stress_mlp input features and widens only the output channels.
    """
    vm_model = _make_direct_clofnet(1)
    full_model = _make_direct_clofnet(6)
    assert int(vm_model.stress_mlp[0].in_features) == int(full_model.stress_mlp[0].in_features)
    assert int(vm_model.stress_mlp[0].in_features) == 2 * HIDDEN
    assert int(vm_model.stress_mlp[2].out_features) == 1
    assert int(full_model.stress_mlp[2].out_features) == 6


def test_clofnet_full_stress_forward_shape_and_finite_values():
    """
    Full-stress CLOFNet predicts acceleration plus six stress tensor channels.
    """
    model = _make_direct_clofnet(6)
    batch = _synthetic_batch()
    out = model(batch)
    assert out.shape == (NUM_NODES, 9)
    assert torch.isfinite(out).all()
