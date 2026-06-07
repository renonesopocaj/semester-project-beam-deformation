"""
test_recurrent_multihead_clofnet.py

Run with:
    python -m pytest tests/gnn/test_recurrent_multihead_clofnet.py

Tests the recurrent multi-head CLOFNet block that combines MultiHeadMPBlock field updates with
RecurrentMPBlock-style branch messages.
"""

import pytest
import torch
from torch import nn

from src.gnn.models.clofnet.clof_mp_blocks import RecurrentMPBlock, RecurrentMultiHeadMPBlock


HIDDEN = 8
FUSE_HID = 5
NUM_NODES = 4
NUM_EDGES = 6


def _silu_factory():
    return nn.SiLU()


class _ConstantMessageMLP(nn.Module):
    """
    Test double for message MLPs that emits a constant update with the requested hidden width.
    """

    value: float
    hidden_dim: int

    def __init__(self, value, hidden_dim):
        super().__init__()
        self.value = value
        self.hidden_dim = hidden_dim

    def forward(self, features):
        """
        Args:
            features: torch.Tensor [E, F]
                Message input features; only E, device, and dtype define the output.
        return: torch.Tensor [E, hidden_dim]
            Constant message update.
        """
        return torch.full((features.shape[0], self.hidden_dim), self.value, device=features.device,
                          dtype=features.dtype)


def _make_recurrent_block(resnet_style):
    """
    Build a small single-message recurrent block for residual-message checks.

    Args:
        resnet_style: bool
            Whether the block should add m_prev to the message MLP output.
    return: RecurrentMPBlock
        Configured block with hidden width HIDDEN.
    """
    return RecurrentMPBlock(
        in_node_mlp_dim=2 * HIDDEN,
        output_node_dim=HIDDEN,
        hidden_dim=HIDDEN,
        in_mij_mlp_dim=2 * HIDDEN + 1 + FUSE_HID + HIDDEN,
        act_fn_factory=_silu_factory,
        recurrent=True,
        resnet_style=resnet_style,
        coords_weight=1.0,
        attention=False,
        tanh=False,
        clamp=False,
        n_layers_node_mlp=2,
        n_layers_gate_mlp=2,
        n_layers_message_mlp=2,
        mlp_layer_norm=False,
    )


def _make_block_with_gate_dim(include_u_tm1, include_v_branch, include_force, attention, vel_mlp_dim_out,
                              resnet_style):
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
        vel_mlp_dim_out: int
            Output width for gates multiplied by 3D vector fields. Must be 1 or 3.
        resnet_style: bool
            Whether the block should add previous branch messages to the message MLP outputs.
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
        resnet_style=resnet_style,
        coords_weight=1.0,
        attention=attention,
        tanh=False,
        clamp=False,
        vel_mlp_dim_out=vel_mlp_dim_out,
        include_u_tm1=include_u_tm1,
        include_v_branch=include_v_branch,
        include_force=include_force,
        n_layers_node_mlp=2,
        n_layers_gate_mlp=2,
        n_layers_message_mlp=2,
        mlp_layer_norm=False,
    )


def _make_block(include_u_tm1, include_v_branch, include_force, attention):
    """
    Build a small recurrent multi-head block with componentwise vector gates.

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
    return _make_block_with_gate_dim(include_u_tm1, include_v_branch, include_force, attention, 3, False)


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


def test_recurrent_multihead_block_rejects_non_broadcastable_gate_width():
    """
    Gate outputs must broadcast with [N, 3] vector fields before torch.compile tracing starts.
    """
    _make_block_with_gate_dim(False, True, False, False, 1, False)
    _make_block_with_gate_dim(False, True, False, False, 3, False)
    with pytest.raises(ValueError, match="must be 1 or 3"):
        _make_block_with_gate_dim(False, True, False, False, 2, False)


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


def test_recurrent_block_resnet_style_adds_previous_message():
    """
    resnet_style controls whether a single recurrent message stores phi_m(...) or m_prev + phi_m(...).
    """
    h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, _, _, _, _ = _inputs()
    m_prev = torch.full((NUM_EDGES, HIDDEN), 3.0)
    plain_block = _make_recurrent_block(False)
    resnet_block = _make_recurrent_block(True)
    plain_block.message_mij_mlp = _ConstantMessageMLP(2.0, HIDDEN)
    resnet_block.message_mij_mlp = _ConstantMessageMLP(2.0, HIDDEN)
    with torch.no_grad():
        plain_out = plain_block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial,
                                fuse_embed, m_prev, a)
        resnet_out = resnet_block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial,
                                  fuse_embed, m_prev, a)
    assert torch.allclose(plain_out[2], torch.full_like(m_prev, 2.0))
    assert torch.allclose(resnet_out[2], torch.full_like(m_prev, 5.0))


def test_recurrent_multihead_resnet_style_adds_previous_branch_messages():
    """
    resnet_style applies independently to each recurrent multi-head branch.
    """
    h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x, u_tm1, f = _inputs()
    branch_names = ["a", "x", "v"]
    state = _message_state(branch_names, 3.0)
    plain_block = _make_block_with_gate_dim(False, True, False, False, 3, False)
    resnet_block = _make_block_with_gate_dim(False, True, False, False, 3, True)
    for block in [plain_block, resnet_block]:
        block.message_mij_mlp_a = _ConstantMessageMLP(2.0, HIDDEN)
        block.message_mij_mlp_x = _ConstantMessageMLP(2.0, HIDDEN)
        block.message_mij_mlp_v = _ConstantMessageMLP(2.0, HIDDEN)
    with torch.no_grad():
        plain_out = plain_block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial,
                                fuse_embed, state, a, v, x, u_tm1, f)
        resnet_out = resnet_block(h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial,
                                  fuse_embed, state, a, v, x, u_tm1, f)
    plain_next = plain_out[6]
    resnet_next = resnet_out[6]
    for branch_name in branch_names:
        assert torch.allclose(plain_next[branch_name], torch.full_like(state[branch_name], 2.0))
        assert torch.allclose(resnet_next[branch_name], torch.full_like(state[branch_name], 5.0))
