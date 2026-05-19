import torch
import yaml

from src.gnn.misc_helpers.yamlobject import PreprocessYaml
from src.gnn.preprocessing.build_features import probe_feature_layout


def pytest_addoption(parser):
    """
    Informal specification: config-validation tests should accept a Hydra-like config selector and
    optional overrides from CLI, so users can preflight custom config combinations before real runs.
    """
    parser.addoption("--cn", action="store", help="Hydra config name to validate for the selected test file.")
    parser.addoption(
        "--override",
        action="append",
        default=[],
        help="Hydra override for config validation; repeat the flag for multiple overrides.",
    )


# ---------------------------------------------------------------------------
# Shared preprocess config helpers
# ---------------------------------------------------------------------------

_MINIMAL_PREPROCESS_KEYS = {
    "dataset_size": 1,
    "device": "cpu",
    "dt": 0.01,
    "seed": 0,
    "raw_data_dir": "x",
    "base_dir": "x",
    "save_dir": "x",
    "limit_timesteps": 1,
    "mises_log": True,
    "isotropic_normalizer": True,
    "normalizer_method": "z",
    "dataset_type": "x",
    "time_discretization_method": "second_order_central",
    "target_mode": "von_mises",
    "extra_info_dir": "x",
}


def preprocess_cfg_from_feature_yaml(feature_yaml_path):
    """
    Build a PreprocessYaml object from a feature preset YAML and a minimal set of required
    non-feature keys, then re-probe node/edge channel counts via probe_feature_layout so the
    returned config is always consistent with the current build_features implementation.

    This mirrors the preprocessing pipeline in preprocess.py and should be used by any test
    that needs a fully-initialised preprocess config without a real dataset on disk.

    Args:
        feature_yaml_path: Path
            Path to a feature preset YAML file (e.g. conf/preprocess/features/onlyedge.yaml).
    return: PreprocessYaml
        Fully-initialised preprocess config with node_channels and edge_channels set.
    """
    feature_dict = yaml.safe_load(feature_yaml_path.read_text(encoding="utf-8"))
    merged = {**_MINIMAL_PREPROCESS_KEYS, **feature_dict}
    cfg = PreprocessYaml(merged)
    sync_feature_layout(cfg)
    return cfg


def sync_feature_layout(preprocess_cfg):
    """
    Re-probe build_features with the current feature toggles and write the resulting
    node_channels and edge_channels back onto the config object in-place.

    Use after loading a saved preprocess config.yaml to ensure stale or missing channel
    counts are replaced by the values that the current code would compute.

    Args:
        preprocess_cfg: PreprocessYaml or SimpleNamespace
            Mutable config object with feature toggle attributes.
    """
    device = torch.device("cpu")
    node_channels, edge_channels, _, _ = probe_feature_layout(preprocess_cfg, device)
    preprocess_cfg.node_channels = int(node_channels)
    preprocess_cfg.edge_channels = int(edge_channels)
