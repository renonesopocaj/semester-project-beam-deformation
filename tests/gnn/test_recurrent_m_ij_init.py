"""
test_recurrent_m_ij_init.py

Tests for RecurrentMessageInitializer covering all five m_ij_init modes.

What we test:
- Every mode produces m_ij^0 with the correct shape [E, hidden_dim] and finite values.
- `zero` gives an exactly all-zero tensor.
- `gaussian` draws different values on two consecutive forwards (it is stochastic per call).
- Learned modes (`edge_encoder`, `context`, `context_stochastic`) produce non-zero outputs
  from a freshly initialised module and carry gradients back to their MLP parameters.
- `context_stochastic` is stochastic during training (two forwards differ) but deterministic
  at eval (two forwards are identical).
- An invalid m_ij_init string raises ValueError when constructing ClofNetRecurrent.
"""

import torch
import pytest
from torch import nn

from src.gnn.models.clofnet.clof_mp_blocks import RecurrentMessageInitializer
from src.gnn.models.clofnet.clofnet import ClofNetRecurrent


HIDDEN = 8
FUSE_HID = 12
E = 6       # number of edges
N = 4       # number of nodes
N_LAYERS = 2


def _silu_factory():
    return nn.SiLU()


def _make_initializer(m_ij_init):
    return RecurrentMessageInitializer(
        hidden_dim=HIDDEN,
        fuse_hid_dim=FUSE_HID,
        m_ij_init=m_ij_init,
        n_layers_message_mlp=2,
        act_fn_factory=_silu_factory,
        mlp_layer_norm=False,
    )


def _inputs():
    torch.manual_seed(0)
    h = torch.randn(N, HIDDEN)
    row = torch.tensor([0, 1, 1, 2, 2, 0])
    col = torch.tensor([1, 0, 2, 1, 0, 2])
    edge_index = (row, col)
    radial = torch.randn(E, 1)
    fuse_embed = torch.randn(E, FUSE_HID)
    return h, edge_index, radial, fuse_embed


@pytest.mark.parametrize("mode", ["zero", "gaussian", "edge_encoder", "context", "context_stochastic"])
def test_output_shape_and_finite(mode):
    """All modes output [E, hidden_dim] with no NaN or Inf."""
    init = _make_initializer(mode)
    init.train()
    h, edge_index, radial, fuse_embed = _inputs()
    m = init(h, edge_index, radial, fuse_embed)
    assert m.shape == (E, HIDDEN), f"shape mismatch for mode {mode}"
    assert torch.isfinite(m).all(), f"non-finite values for mode {mode}"


def test_zero_mode_is_exactly_zero():
    init = _make_initializer("zero")
    h, edge_index, radial, fuse_embed = _inputs()
    m = init(h, edge_index, radial, fuse_embed)
    assert (m == 0).all()


def test_gaussian_differs_across_forwards():
    """Two stochastic draws should almost surely differ."""
    init = _make_initializer("gaussian")
    h, edge_index, radial, fuse_embed = _inputs()
    m1 = init(h, edge_index, radial, fuse_embed)
    m2 = init(h, edge_index, radial, fuse_embed)
    assert not torch.equal(m1, m2)


@pytest.mark.parametrize("mode", ["edge_encoder", "context", "context_stochastic"])
def test_learned_mode_nonzero_output(mode):
    """Learned initialisers should produce non-trivial (not all-zero) output."""
    torch.manual_seed(42)
    init = _make_initializer(mode)
    init.train()
    h, edge_index, radial, fuse_embed = _inputs()
    m = init(h, edge_index, radial, fuse_embed)
    assert m.abs().sum().item() > 0, f"mode {mode} produced all-zero output"


@pytest.mark.parametrize("mode", ["edge_encoder", "context"])
def test_learned_deterministic_mode_gradients(mode):
    """MLP parameters receive gradient through the initialiser output."""
    init = _make_initializer(mode)
    h, edge_index, radial, fuse_embed = _inputs()
    h = h.requires_grad_(False)
    m = init(h, edge_index, radial, fuse_embed)
    m.sum().backward()
    mlp = init.psi_init_mlp if mode == "edge_encoder" else init.context_init_mlp
    for p in mlp.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0


def test_context_stochastic_gradients():
    init = _make_initializer("context_stochastic")
    init.train()
    h, edge_index, radial, fuse_embed = _inputs()
    m = init(h, edge_index, radial, fuse_embed)
    m.sum().backward()
    for name, mlp in [("mu", init.context_mu_mlp), ("sigma", init.context_sigma_mlp)]:
        for p in mlp.parameters():
            assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name} MLP has no gradient"


def test_context_stochastic_train_is_stochastic():
    """Two training-mode forwards should differ due to sampled noise."""
    init = _make_initializer("context_stochastic")
    init.train()
    h, edge_index, radial, fuse_embed = _inputs()
    with torch.no_grad():
        m1 = init(h, edge_index, radial, fuse_embed)
        m2 = init(h, edge_index, radial, fuse_embed)
    assert not torch.equal(m1, m2)


def test_context_stochastic_eval_is_deterministic():
    """At eval time the stochastic mode should return the same result each call."""
    init = _make_initializer("context_stochastic")
    init.eval()
    h, edge_index, radial, fuse_embed = _inputs()
    with torch.no_grad():
        m1 = init(h, edge_index, radial, fuse_embed)
        m2 = init(h, edge_index, radial, fuse_embed)
    assert torch.equal(m1, m2)


def test_invalid_m_ij_init_raises():
    """Constructing ClofNetRecurrent with an unknown mode must raise ValueError."""
    common = dict(
        in_dim_node_embed=4,
        in_dim_edge_scalars=4,
        hidden_dim=HIDDEN,
        fuse_hid_dim=FUSE_HID,
        act_fn_factory=_silu_factory,
        n_layers=1,
        coords_weight=1.0,
        recurrent=True,
        enable_norm_diff=True,
        tanh=False,
        attention_enable=False,
        clamp_enable=False,
        vel_mlp_dim_out=1,
        split_idx_node=0,
        split_idx_edge=0,
        node_channels=5,
        edge_channels=4,
        use_neighbor_diff_norms=False,
        frame_source="abs",
        include_u_tm1=False,
        rescalarize_per_layer=False,
        include_v_branch=False,
        include_force=False,
        eps_mode="soft",
        n_layers_gate_mlp=1,
        n_layers_node_mlp=1,
        n_layers_message_mlp=1,
        normalize_delta_a=False,
        mlp_layer_norm=False,
    )
    with pytest.raises(ValueError, match="m_ij_init"):
        ClofNetRecurrent(m_ij_init="bad_mode", **common)
