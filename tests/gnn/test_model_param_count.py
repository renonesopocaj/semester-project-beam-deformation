"""
test_model_param_count.py

Prints and verifies the parameter count of a model built from a chosen config.
Change MODEL_CONFIG_NAME to any name in conf/train/model/ to probe a different architecture.

Run this test from the fenicsproject conda environment so Hydra imports correctly (the base
Anaconda Python may pick up a different top-level ``hydra`` package and break compose).

Example:

    conda run -n fenicsproject python -m pytest tests/gnn/test_model_param_count.py -v -s
"""

from pathlib import Path

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml
from src.gnn.train import create_model


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_DIR = REPO_ROOT / "conf" / "train"
TRAIN_CONFIG_NAME = "train"

# ---- Change this constant to select any config in conf/train/model/ ----
# MODEL_CONFIG_NAME = "transolver_5L_HD256_AH8_S32_R1"
# MODEL_CONFIG_NAME = "transolver_5L_HD512_AH10_S32_R1"
# MODEL_CONFIG_NAME = "transolver_3L_HD512_AH10_S32_R1"
# MODEL_CONFIG_NAME = "transolver_5L_HD256_AH30_S32_R1"
# MODEL_CONFIG_NAME = "transolver_10L_HD256_AH30_S32_R4"
# MODEL_CONFIG_NAME = "transolver_3L_HD512_AH5_S32_R4"
# MODEL_CONFIG_NAME = "transolver_7L_HD312_AH15_S32_R4"
# ------ Clofnet -------
# MODEL_CONFIG_NAME = "clofnet_rec_L15_softeps"
# MODEL_CONFIG_NAME = "clofnet_rec_L15_hardeps"
# MODEL_CONFIG_NAME = "clofnet_rec_L8_hardeps"
# MODEL_CONFIG_NAME = "clofnet_rec_L8_softeps"
# MODEL_CONFIG_NAME = "clofnet_rec_L8_softeps_framesourceu"
MODEL_CONFIG_NAME = "clofnet_rec_L8_gaussian_softeps"
MODEL_CONFIG_NAME = "mgn_H400_L11_1layermlp"

# ------------------------------------------------------------------------

_SYNTHETIC_NUM_NODES = 64


def _make_synthetic_graph_template(node_channels):
    """
    Build a minimal undirected ring graph for models that require a topology
    template at construction time (magnet, transolver).

    Args:
        node_channels: int
            Width of the synthetic node feature matrix.
    return: Data
        PyG Data with edge_index and x of shape [_SYNTHETIC_NUM_NODES, node_channels].
    """
    n = _SYNTHETIC_NUM_NODES
    row = torch.arange(n, dtype=torch.long)
    col = (row + 1) % n
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    return Data(x=torch.zeros(n, node_channels), edge_index=edge_index)


def test_model_param_count():
    """
    Informal specification: given the MODEL_CONFIG_NAME constant set at the top of this file,
    the test composes the full Hydra train config with that model override, loads the
    preprocess config from the path resolved by the train config (skipping the test when the
    dataset is not present), instantiates the model on CPU without any training or dataset
    loading, and asserts that the parameter count is positive while printing it clearly.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(TRAIN_CONFIG_DIR)):
        cfg = compose(config_name=TRAIN_CONFIG_NAME, overrides=[f"model={MODEL_CONFIG_NAME}"])
    OmegaConf.resolve(cfg)
    train_cfg = TrainYaml(cfg)

    preprocess_config_path = Path(train_cfg.preprocess_config_path)
    if not preprocess_config_path.exists():
        pytest.skip(
            f"Preprocess config not found at '{preprocess_config_path}'. "
            "Run preprocessing first or set dataset_dir to an existing processed dataset."
        )

    with open(preprocess_config_path) as fh:
        preprocess_dict = yaml.safe_load(fh)
    preprocess_cfg = PreprocessYaml(preprocess_dict)

    node_channels = getattr(preprocess_cfg, "node_channels", _SYNTHETIC_NUM_NODES)
    graph_template = _make_synthetic_graph_template(node_channels)
    device = torch.device("cpu")

    target_mode = getattr(preprocess_cfg, "target_mode")
    stress_out_dim = 6 if target_mode == "full_stress" else 1
    model = create_model(train_cfg, preprocess_cfg, device, graph_template, stress_out_dim)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel config '{MODEL_CONFIG_NAME}': {total_params:,} parameters")
    assert total_params > 0, f"Model built from config '{MODEL_CONFIG_NAME}' has zero parameters."
