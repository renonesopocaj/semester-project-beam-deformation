from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.gnn.misc_helpers.yamlobject import TrainYaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_DIR = REPO_ROOT / "conf" / "train"
SUPPORTED_TRAIN_CONFIGS = ["train", "train_sweep_defaults", "train_sanity_sweep"]


def _compose_train_config(config_name, overrides):
    """
    Informal specification: a train config is considered valid only if Hydra can compose it with
    overrides, all interpolations resolve, and TrainYaml can read the expected flattened fields.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(TRAIN_CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    TrainYaml(cfg)
    return cfg


def _selected_train_config_names(pytestconfig):
    """
    Informal specification: without an explicit selector, all supported train roots should be validated;
    with --cn, only the requested root should be tested and unknown names should fail fast.
    """
    config_name = pytestconfig.getoption("--cn")
    if config_name is None:
        return list(SUPPORTED_TRAIN_CONFIGS)
    if config_name not in SUPPORTED_TRAIN_CONFIGS:
        raise ValueError(
            "Unsupported train config '%s'. Allowed values: %s."
            % (config_name, ", ".join(SUPPORTED_TRAIN_CONFIGS))
        )
    return [config_name]


def _runtime_overrides(pytestconfig):
    """
    Informal specification: CLI overrides must be passed through unmodified so validation reproduces
    the exact config composition users plan to run with src/gnn/train.py.
    """
    return list(pytestconfig.getoption("--override"))


def test_train_config_composition_and_interpolation(pytestconfig):
    """
    Informal specification: each selected train root must compose cleanly, reject non-existent fields,
    and resolve interpolations before any training run starts.
    """
    override_list = _runtime_overrides(pytestconfig)
    config_names = _selected_train_config_names(pytestconfig)
    for config_name in config_names:
        try:
            _compose_train_config(config_name, override_list)
        except Exception as exception:
            pytest.fail(
                "Train config validation failed for -cn %s with overrides %s.\n%s"
                % (config_name, override_list, exception)
            )
