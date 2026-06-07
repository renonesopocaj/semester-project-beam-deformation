"""
Informal specification:
- For every valid boolean feature-toggle configuration used by preprocessing builders,
  the derived layout (node/edge channels and split indices) must be self-consistent.
- The same layout contract must be consumed coherently by:
  1) `src.gnn.train.create_model` (ClofNet branch arguments),
  2) `src.gnn.training.trainer.LitModule._with_online_random_edges_eval_step`,
  3) `src.gnn.training.rollout._build_next_graph`.
- The test intentionally validates propagation/coherence of the feature-layout contract,
  not model quality or training metrics.
"""

import itertools
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

import src.gnn.train as train_module
import src.gnn.training.rollout as rollout_module
from src.gnn.preprocessing.build_features import make_scripted_feature_builders, probe_feature_layout
from src.gnn.preprocessing.normalizer import create_normalizer
from src.gnn.preprocessing.stats import Stats, StatsModule
from src.gnn.training.online_edge_augmentation import SPLIT_PREDICT
from src.gnn.training.trainer import LitModule


FEATURE_FLAG_NAMES = [
    "enable_past_history",
    "enable_absolute_pos",
    "enable_node_norms",
    "enable_velocity_node",
    "enable_displacement_node",
    "enable_force_node",
    "enable_rest_pos_node",
    "enable_displ_edge",
    "enable_edge_velocity",
    "enable_force_edge",
    "enable_abs_pos_edge",
    "enable_abs_rest_pos_edge",
    "enable_abs_pos_edge_norm",
    "enable_rest_pos_edge_norm",
    "enable_force_edge_norm",
    "enable_edge_velocity_norm",
    "enable_stretch_edge",
    "enable_rotation_angle",
]


class _DummyClofModel:
    """
    Lightweight stand-in for ClofNet variants to inspect constructor kwargs in create_model.
    """

    kwargs: dict

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def _make_preprocess_cfg(flag_values):
    cfg = SimpleNamespace()
    cfg.normalizer_method = "z"
    cfg.isotropic_normalizer = True
    cfg.time_discretization_method = "second_order_central"
    cfg.target_mode = "von_mises"
    for name, value in zip(FEATURE_FLAG_NAMES, flag_values):
        setattr(cfg, name, bool(value))
    return cfg


def _make_train_cfg_for_clofnet():
    train_cfg = SimpleNamespace()
    train_cfg.model_name = "clofnet"
    train_cfg.clofnet_activation = "relu"
    train_cfg.leakyrelu_negative_slope = 0.01
    train_cfg.hidden_dim = 8
    train_cfg.embedding_edge_dim = 8
    train_cfg.n_layers = 1
    train_cfg.coords_weight = 1.0
    train_cfg.recurrent = False
    train_cfg.enable_norm_diff = False
    train_cfg.tanh = False
    train_cfg.attention_enable = False
    train_cfg.clamp_enable = False
    train_cfg.out_coord_mlp_dim = 8
    train_cfg.clofnet_variant = "direct"
    train_cfg.use_neighbor_diff_norms = False
    train_cfg.frame_source = "rest"
    train_cfg.include_u_tm1 = False
    train_cfg.rescalarize_per_layer = False
    train_cfg.include_v_branch = True
    train_cfg.include_force = True
    train_cfg.m_ij_init = "ones"
    train_cfg.resnet_style = False
    train_cfg.n_layers_gate_mlp = 1
    train_cfg.n_layers_node_mlp = 1
    train_cfg.n_layers_message_mlp = 1
    train_cfg.eps_mode = "none"
    train_cfg.normalize_delta_a = False
    train_cfg.mlp_layer_norm = False
    return train_cfg


def _build_minibatch_tensors(num_nodes, device):
    x_t = torch.randn(num_nodes, 3, dtype=torch.float32, device=device)
    v_t = torch.randn(num_nodes, 3, dtype=torch.float32, device=device)
    x_tm1_prev = torch.randn(num_nodes, 3, dtype=torch.float32, device=device)
    rest_pos = torch.randn(num_nodes, 3, dtype=torch.float32, device=device)
    node_force_t = torch.randn(num_nodes, 3, dtype=torch.float32, device=device)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long, device=device
    )
    return x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index


def _is_invalid_empty_edge_feature_combo(cfg):
    """
    Returns True when edge feature builder would produce an empty list and fail torch.cat.
    Note: enable_abs_pos_edge / enable_abs_rest_pos_edge are no longer gated by enable_absolute_pos.
    """
    has_any = False
    has_any = has_any or bool(cfg.enable_displ_edge)
    has_any = has_any or bool(cfg.enable_edge_velocity)
    has_any = has_any or bool(cfg.enable_force_edge)
    has_any = has_any or bool(cfg.enable_abs_pos_edge)
    has_any = has_any or bool(cfg.enable_abs_rest_pos_edge)
    has_any = has_any or bool(cfg.enable_abs_pos_edge_norm)
    has_any = has_any or bool(cfg.enable_rest_pos_edge_norm)
    has_any = has_any or bool(cfg.enable_force_edge_norm)
    has_any = has_any or bool(cfg.enable_edge_velocity_norm)
    has_any = has_any or bool(cfg.enable_stretch_edge)
    has_any = has_any or bool(cfg.enable_rotation_angle)
    if bool(cfg.enable_past_history):
        has_any = has_any or bool(cfg.enable_displ_edge)
        has_any = has_any or bool(cfg.enable_abs_pos_edge)
    return not has_any


def test_feature_layout_coherence_through_preprocess_train_trainer_rollout(monkeypatch):
    """
    Informal specification:
    for every valid preprocessing feature-toggle combination, the feature layout contract
    inferred in preprocessing must be consumed consistently by train.create_model, the trainer
    online-edge augmentation path, and rollout graph rebuilding.
    """
    monkeypatch.setattr(train_module, "ClofNetDirectAcceleration", _DummyClofModel)
    monkeypatch.setattr(train_module, "ClofNetRecurrent", _DummyClofModel)
    monkeypatch.setattr(train_module, "ClofNetSeparateMessages", _DummyClofModel)
    monkeypatch.setattr(train_module, "ClofNetRecurrentSeparateMessages", _DummyClofModel)
    monkeypatch.setattr(train_module, "ClofNetExtraLayer", _DummyClofModel)

    device = torch.device("cpu")
    train_cfg = _make_train_cfg_for_clofnet()
    graph_template = Data()
    total_seen = 0
    total_valid = 0

    for flag_values in itertools.product([False, True], repeat=len(FEATURE_FLAG_NAMES)):
        total_seen += 1
        preprocess_cfg = _make_preprocess_cfg(flag_values)
        if _is_invalid_empty_edge_feature_combo(preprocess_cfg):
            continue
        total_valid += 1

        node_channels, edge_channels, split_idx_node_probe, split_idx_edge_probe = probe_feature_layout(
            preprocess_cfg, device
        )
        preprocess_cfg.node_channels = int(node_channels)
        preprocess_cfg.edge_channels = int(edge_channels)
        x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index = _build_minibatch_tensors(3, device)
        node_builder, edge_builder = make_scripted_feature_builders(preprocess_cfg)
        node_features, split_idx_node_builder = node_builder(x_t, v_t, x_tm1_prev, rest_pos, node_force_t)
        edge_features, split_idx_edge_builder = edge_builder(
            x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index
        )
        assert int(split_idx_node_builder) == int(split_idx_node_probe)
        assert int(split_idx_edge_builder) == int(split_idx_edge_probe)
        assert int(node_channels) == int(1 + node_features.shape[1])
        assert int(edge_channels) == int(edge_features.shape[1])

        target_mode = getattr(preprocess_cfg, "target_mode")
        stress_out_dim = 6 if target_mode == "full_stress" else 1
        model_obj = train_module.create_model(train_cfg, preprocess_cfg, device, graph_template, stress_out_dim)
        assert isinstance(model_obj, _DummyClofModel)
        kwargs = model_obj.kwargs
        assert int(kwargs["split_idx_node"]) == int(split_idx_node_probe)
        assert int(kwargs["split_idx_edge"]) == int(split_idx_edge_probe)
        assert int(kwargs["node_channels"]) == int(node_channels)
        assert int(kwargs["edge_channels"]) == int(edge_channels)
        assert int(kwargs["in_dim_node_embed"]) == int(node_channels - split_idx_node_probe)
        assert int(kwargs["in_dim_edge_scalars"]) == int(edge_channels - split_idx_edge_probe)
        assert int(kwargs["stress_out_dim"]) == int(stress_out_dim)

        edge_stats = Stats(
            mean=torch.zeros(edge_channels, dtype=torch.float32),
            std=torch.ones(edge_channels, dtype=torch.float32),
        )
        trainer_shim = SimpleNamespace(
            random_online=0.5,
            _online_random_edge_augmenter=None,
            preprocess_cfg=preprocess_cfg,
            used_dataset_normalizer=create_normalizer(preprocess_cfg.normalizer_method, preprocess_cfg.isotropic_normalizer),
            edge_stats=StatsModule.from_stats(edge_stats),
            train_seed=123,
        )
        batch = Data(
            x=torch.cat([torch.zeros(x_t.shape[0], 1, dtype=torch.float32), node_features.cpu()], dim=1),
            edge_attr=edge_features.cpu(),
            edge_index=edge_index.cpu(),
            x_pos_t=x_t.cpu(),
            x_pos_tm1_prev=x_tm1_prev.cpu(),
            x_vel_t=v_t.cpu(),
            rest_pos=rest_pos.cpu(),
            node_force=node_force_t.cpu(),
        )
        augmented_batch = LitModule._with_online_random_edges_eval_step(
            trainer_shim, batch, batch_idx=0, split_code=SPLIT_PREDICT
        )
        assert int(augmented_batch.edge_attr.shape[1]) == int(edge_channels)
        assert trainer_shim._online_random_edge_augmenter is not None

        node_stats = Stats(
            mean=torch.zeros(node_features.shape[1], dtype=torch.float32),
            std=torch.ones(node_features.shape[1], dtype=torch.float32),
        )
        ground_truth_graph = Data(
            x=torch.cat([torch.zeros(x_t.shape[0], 1, dtype=torch.float32), node_features.cpu()], dim=1),
            edge_index=edge_index.cpu(),
            node_force_next=node_force_t.cpu(),
            x_pos_t=x_t.cpu(),
            x_pos_tm1_prev=x_tm1_prev.cpu(),
            x_vel_t=v_t.cpu(),
            rest_pos=rest_pos.cpu(),
            node_force=node_force_t.cpu(),
        )
        next_graph = rollout_module._build_next_graph(
            ground_truth_graph=ground_truth_graph,
            x_predict_tp1=x_t.cpu(),
            v_predict_tp1=v_t.cpu(),
            x_t=x_tm1_prev.cpu(),
            rest_pos=rest_pos.cpu(),
            node_stats=node_stats,
            edge_stats=edge_stats,
            normalizer=create_normalizer(preprocess_cfg.normalizer_method, preprocess_cfg.isotropic_normalizer),
            node_feature_builder=node_builder,
            edge_feature_builder=edge_builder,
        )
        assert int(next_graph.x.shape[1]) == int(node_channels)
        assert int(next_graph.edge_attr.shape[1]) == int(edge_channels)

    assert total_seen == 2 ** len(FEATURE_FLAG_NAMES)
    assert total_valid > 0


def test_onlyedge_and_nonodepos_configs_produce_node_free_graphs():
    """
    Checks that when all four per-node valued-feature flags are False, build_features produces:
    - node_channels == 1 (boundary-condition flag only, no valued columns)
    - split_idx_node == 0
    - edge_channels > 0 and consistent with build_edge_feature_names
    This covers the intent of onlyedge.yaml and nonodepos_yesedge.yaml.
    """
    from src.gnn.eda_helpers.eda_features_helpers import build_edge_feature_names

    device = torch.device("cpu")
    cfg = SimpleNamespace(
        enable_absolute_pos=False,
        enable_past_history=True,
        enable_velocity_node=False,
        enable_displacement_node=False,
        enable_force_node=False,
        enable_rest_pos_node=False,
        enable_node_norms=False,
        enable_edge_velocity=True,
        enable_force_edge=True,
        enable_displ_edge=True,
        enable_abs_pos_edge=True,
        enable_abs_rest_pos_edge=True,
        enable_abs_pos_edge_norm=True,
        enable_rest_pos_edge_norm=True,
        enable_force_edge_norm=True,
        enable_edge_velocity_norm=True,
        enable_stretch_edge=False,
        enable_rotation_angle=False,
        normalizer_method="z",
        isotropic_normalizer=True,
    )

    node_channels, edge_channels, split_idx_node, split_idx_edge = probe_feature_layout(cfg, device)

    assert node_channels == 1, f"Expected node_channels=1 (bc only), got {node_channels}"
    assert split_idx_node == 0, f"Expected split_idx_node=0, got {split_idx_node}"
    assert edge_channels > 0, f"Expected positive edge_channels, got {edge_channels}"

    edge_names = build_edge_feature_names(cfg)
    assert len(edge_names) == edge_channels, (
        f"EDA edge name count ({len(edge_names)}) != edge_channels ({edge_channels})"
    )
