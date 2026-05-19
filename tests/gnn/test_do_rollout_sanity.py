"""
Informal specification:
- `do_rollout` must run end-to-end on a minimal two-step trajectory without raising,
  when the model returns normalized predictions compatible with identity-like stats.
- After each step, stored rollout entries must have consistent tensor shapes and
  channel widths implied by the preprocessing feature layout (node/edge channels).
"""

from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from src.gnn.mesh_numerical_helpers.time_discretizer import create_time_discretizer
from src.gnn.preprocessing.build_features import make_scripted_feature_builders, probe_feature_layout
from src.gnn.preprocessing.normalizer import create_normalizer
from src.gnn.preprocessing.stats import Stats
from src.gnn.training.online_edge_augmentation import SPLIT_ROLLOUT_TEST
from src.gnn.training.rollout import do_rollout
from src.gnn.training.target_strategy import VonMisesStrategy


class _ZeroPredModel(torch.nn.Module):
    """Returns (N, 4) zeros in normalized target space (matches zero-mean unit-std target stats)."""

    def forward(self, batch):
        n = int(batch.x.shape[0])
        return torch.zeros(n, 4, device=batch.x.device, dtype=batch.x.dtype)


def _minimal_preprocess_cfg():
    """
    Single valid layout: edge-only displacement block, no valued node features.
    All four per-node granular toggles are explicitly False so build_features produces
    node_channels == 1 (boundary-condition flag only).
    """
    cfg = SimpleNamespace()
    cfg.normalizer_method = "z"
    cfg.isotropic_normalizer = True
    cfg.time_discretization_method = "second_order_central"
    for name in (
        "enable_past_history",
        "enable_absolute_pos",
        "enable_node_norms",
        "enable_edge_velocity",
        "enable_force_edge",
        "enable_stretch_edge",
        "enable_rotation_angle",
        "enable_abs_pos_edge",
        "enable_abs_rest_pos_edge",
        "enable_abs_pos_edge_norm",
        "enable_rest_pos_edge_norm",
        "enable_force_edge_norm",
        "enable_edge_velocity_norm",
        "enable_velocity_node",
        "enable_displacement_node",
        "enable_force_node",
        "enable_rest_pos_node",
    ):
        setattr(cfg, name, False)
    cfg.enable_displ_edge = True
    return cfg


def _make_graph(device, dt, node_stats, edge_stats, normalizer, node_b, edge_b, step_tag):
    n = 2
    x_t = torch.randn(n, 3, device=device, dtype=torch.float32)
    v_t = torch.randn(n, 3, device=device, dtype=torch.float32)
    x_tm1 = torch.randn(n, 3, device=device, dtype=torch.float32)
    rest = torch.randn(n, 3, device=device, dtype=torch.float32)
    nf = torch.randn(n, 3, device=device, dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    valued, _ = node_b(x_t, v_t, x_tm1, rest, nf)
    edge_raw, _ = edge_b(x_t, v_t, x_tm1, rest, nf, edge_index)
    x_full = torch.cat(
        [torch.zeros(n, 1, device=device, dtype=torch.float32), normalizer.normalize(valued, node_stats)], dim=1
    )
    edge_attr = normalizer.normalize(edge_raw, edge_stats)
    x_next = x_t + v_t * dt + 0.01 * float(step_tag)
    v_next = v_t + 0.01 * float(step_tag)
    return Data(
        x=x_full,
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_pos_t=x_t,
        x_pos_tm1_prev=x_tm1,
        x_vel_t=v_t,
        rest_pos=rest,
        node_force=nf,
        y_acc_t=torch.randn(n, 3, device=device, dtype=torch.float32),
        y_von_mises=torch.randn(n, 1, device=device, dtype=torch.float32) * 0.1,
        y_sig6_t=torch.zeros(n, 6, device=device, dtype=torch.float32),
        x_next=x_next,
        v_next=v_next,
        node_force_next=nf,
        x_element_connectivity=torch.zeros(0, 4, dtype=torch.long, device=device),
    )


def test_do_rollout_two_steps_zero_model():
    """
    Informal specification:
    `do_rollout` completes for two ground-truth graphs, a fake zero-output model, and
    identity-like stats; outputs are finite and edge/node channel widths stay aligned
    with `probe_feature_layout` for the same preprocess config.
    """
    device = torch.device("cpu")
    dt = 0.01
    preprocess_cfg = _minimal_preprocess_cfg()
    node_ch, edge_ch, _, _ = probe_feature_layout(preprocess_cfg, device)
    preprocess_cfg.node_channels = int(node_ch)
    preprocess_cfg.edge_channels = int(edge_ch)
    node_b, edge_b = make_scripted_feature_builders(preprocess_cfg)
    normalizer = create_normalizer(preprocess_cfg.normalizer_method, preprocess_cfg.isotropic_normalizer)
    probe_x = torch.zeros(2, 3, device=device)
    probe_nf = torch.zeros(2, 3, device=device)
    probe_ei = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    valued0, _ = node_b(probe_x, probe_x, probe_x, probe_x, probe_nf)
    edge0, _ = edge_b(probe_x, probe_x, probe_x, probe_x, probe_nf, probe_ei)
    f_node = int(valued0.shape[1])
    f_edge = int(edge0.shape[1])
    node_stats = Stats(mean=torch.zeros(f_node, dtype=torch.float32), std=torch.ones(f_node, dtype=torch.float32))
    edge_stats = Stats(mean=torch.zeros(f_edge, dtype=torch.float32), std=torch.ones(f_edge, dtype=torch.float32))
    target_stats = Stats(
        mean=torch.zeros(4, dtype=torch.float32),
        std=torch.ones(4, dtype=torch.float32),
    )
    g0 = _make_graph(device, dt, node_stats, edge_stats, normalizer, node_b, edge_b, 0.0)
    g1 = _make_graph(device, dt, node_stats, edge_stats, normalizer, node_b, edge_b, 1.0)
    assert int(g0.x.shape[1]) == int(node_ch) and int(g0.edge_attr.shape[1]) == int(edge_ch)
    assert int(g1.x.shape[1]) == int(node_ch) and int(g1.edge_attr.shape[1]) == int(edge_ch)
    loader = [g0, g1]
    time_discretizer = create_time_discretizer(preprocess_cfg.time_discretization_method)
    model = _ZeroPredModel()
    target_strategy = VonMisesStrategy("von_mises", exp_mises=False, stress_weight=1.0,
                                       full_stress_loss_mode="components", full_stress_vm_weight=0.0)
    true_roll, pred_roll = do_rollout(
        model=model,
        test_loader=loader,
        node_stats=node_stats,
        edge_stats=edge_stats,
        target_stats=target_stats,
        dt=dt,
        device=device,
        skip_first_num=0,
        rollout_steps=2,
        normalizer=normalizer,
        preprocess_cfg=preprocess_cfg,
        time_discretizer=time_discretizer,
        exp_mises=False,
        random_online=0.0,
        rollout_seed_base=7,
        rollout_online_tag=SPLIT_ROLLOUT_TEST,
        traj_idx=0,
        target_strategy=target_strategy,
    )
    assert len(true_roll) == 2 and len(pred_roll) == 2
    for tr, pr in zip(true_roll, pred_roll):
        assert tr.x_pos_t.shape == pr.x_pos_t.shape == (2, 3)
        assert torch.isfinite(tr.x_pos_t).all() and torch.isfinite(pr.x_pos_t).all()
        assert int(tr.edge_index.shape[0]) == 2
        assert int(pr.edge_index.shape[0]) == 2
