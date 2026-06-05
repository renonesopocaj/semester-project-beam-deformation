"""
test_recurrent_multihead_clofnet.py

Run with:
    python -m pytest tests/gnn/test_recurrent_multihead_clofnet.py

Tests the recurrent multi-head CLOFNet block that combines MultiHeadMPBlock field updates with
RecurrentMPBlock-style branch messages.
"""

import torch
from torch import nn

from src.gnn.models.clofnet.clof_mp_blocks import RecurrentMultiHeadMPBlock


HIDDEN = 8
FUSE_HID = 5
NUM_NODES = 4
NUM_EDGES = 6


def _silu_factory():
    return nn.SiLU()


def _make_block(include_u_tm1, include_v_branch, include_force, attention):
    """
    Build a small recurrent multi-head block for shape and recurrence checks.

    Args:
        include_u_tm1: bool
            Whether to enable the u(t-1) branch.
        include_v_branch: bool
            Whether to enable the velocity branch.
        include_force: bool
            Whether to enable the force branch.
        attention: bool
            Whether to enable branch attention.
    return: RecurrentMultiHeadMPBlock
        Configured block with hidden width HIDDEN.
    """
    return RecurrentMultiHeadMPBlock(
        in_node_mlp_dim=2 * HIDDEN,
        output_node_dim=HIDDEN,
        hidden_dim=HIDDEN,
        in_mij_mlp_dim=2 * HIDDEN + 1 + FUSE_HID + HIDDEN,
        act_fn_factory=_silu_factory,
        recurrent=True,
        coords_weight=1.0,
        attention=attention,
        tanh=False,
        clamp=False,
        vel_mlp_dim_out=3,
        include_u_tm1=include_u_tm1,
        include_v_branch=include_v_branch,
        include_force=include_force,
        n_layers_node_mlp=2,
        n_layers_gate_mlp=2,
        n_layers_message_mlp=2,
        mlp_layer_norm=False,
    )


def _inputs():
    """
    Build deterministic tensors for one directed toy graph.

    return: tuple
        h, edge_index, frame vectors, radial, fuse_embed, a, v, x, u_tm1, f tensors.
    """
    torch.manual_seed(0)
    h = torch.randn(NUM_NODES, HIDDEN)
    row = torch.tensor([0, 1, 1, 2, 2, 0])
    col = torch.tensor([1, 0, 2, 1, 0, 2])
    edge_index = (row, col)
    coord_diff_e3 = torch.randn(NUM_EDGES, 3)
    coord_cross_e3 = torch.randn(NUM_EDGES, 3)
    coord_vertical_e3 = torch.randn(NUM_EDGES, 3)
    radial = torch.rand(NUM_EDGES, 1)
    fuse_embed = torch.randn(NUM_EDGES, FUSE_HID)
    a = torch.randn(NUM_NODES, 3)
    v = torch.randn(NUM_NODES, 3)
    x = torch.randn(NUM_NODES, 3)
    u_tm1 = torch.randn(NUM_NODES, 3)
    f = torch.randn(NUM_NODES, 3)
    return h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x, u_tm1, f


def _message_state(branch_names, fill_value):
    """
    Build previous recurrent branch messages.

    Args:
        branch_names: list[str]
            Active branch names.
        fill_value: float
            Constant value used for every previous message tensor.
    return: dict[str, torch.Tensor]
        Previous branch messages with shape [NUM_EDGES, HIDDEN].
    """
    state = {}
    for branch_name in branch_names:
        state[branch_name] = torch.full((NUM_EDGES, HIDDEN), fill_value)
    return state


def test_recurrent_multihead_block_shapes_all_branches():
    """
    All enabled branches return finite node fields and next recurrent messages with expected shapes.
    """
    block = _make_block(True, True, True, True)
    h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x, u_tm1, f = _inputs()
    branch_names = ["a", "x", "v", "u_tm1", "f"]
    state = _message_state(branch_names, 0.0)
    out = block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, state, a, v, x,
                u_tm1, f)
    h_new, a_new, v_new, x_new, u_tm1_new, f_new, next_state, m_agg = out
    assert h_new.shape == (NUM_NODES, HIDDEN)
    assert a_new.shape == (NUM_NODES, 3)
    assert v_new.shape == (NUM_NODES, 3)
    assert x_new.shape == (NUM_NODES, 3)
    assert u_tm1_new.shape == (NUM_NODES, 3)
    assert f_new.shape == (NUM_NODES, 3)
    assert m_agg.shape == (NUM_NODES, HIDDEN)
    assert sorted(next_state.keys()) == sorted(branch_names)
    for branch_name in branch_names:
        assert next_state[branch_name].shape == (NUM_EDGES, HIDDEN)
        assert torch.isfinite(next_state[branch_name]).all()
    for tensor in [h_new, a_new, v_new, x_new, u_tm1_new, f_new, m_agg]:
        assert torch.isfinite(tensor).all()


def test_recurrent_multihead_messages_depend_on_previous_messages():
    """
    The next message changes when only the previous recurrent message state changes.
    """
    torch.manual_seed(7)
    block = _make_block(False, True, False, False)
    block.eval()
    h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x, u_tm1, f = _inputs()
    branch_names = ["a", "x", "v"]
    zero_state = _message_state(branch_names, 0.0)
    one_state = _message_state(branch_names, 1.0)
    with torch.no_grad():
        out_zero = block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed,
                         zero_state, a, v, x, u_tm1, f)
        out_one = block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed,
                        one_state, a, v, x, u_tm1, f)
    next_zero = out_zero[6]
    next_one = out_one[6]
    assert not torch.allclose(next_zero["a"], next_one["a"])
    assert not torch.allclose(next_zero["x"], next_one["x"])
    assert not torch.allclose(next_zero["v"], next_one["v"])
