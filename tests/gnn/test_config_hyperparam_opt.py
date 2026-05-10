from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.gnn.misc_helpers.yamlobject import TrainYaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HYPERPARAM_CONFIG_DIR = REPO_ROOT / "conf" / "hyperparam"
TRAIN_CONFIG_DIR = REPO_ROOT / "conf" / "train"


def _compose_config(config_dir, config_name, overrides):
    """
    Informal specification: config preflight is successful only if Hydra composes the selected root with
    overrides and every interpolation can be resolved without runtime training side effects.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _bool_from_config_value(value):
    """
    Informal specification: wandb_offline coercion in validation should mirror hyperparam_opt behavior,
    so synthetic runtime overrides match what the real sweep launcher injects.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    lowered = str(value).strip().lower()
    return lowered not in ("false", "0")


def _val_monitor_metric_from_optimization_metric(value):
    """
    Informal specification: optimization metric aliases should map to the same checkpoint monitor key used
    by sweep runs, so config-only validation exercises the same field path as real execution.
    """
    normalized = str(value).strip().lower()
    if normalized in ("overall_abs_mae", "val/overall_abs_mae"):
        return "overall_abs_mae"
    return "overall_relative_mae"


def _available_hyperparam_config_names():
    """
    Informal specification: all non-empty yaml files under conf/hyperparam should be directly preflightable
    via -cn, and empty files must be reported as invalid sweep roots.
    """
    names = []
    for yaml_path in sorted(HYPERPARAM_CONFIG_DIR.glob("*.yaml")):
        if yaml_path.read_text(encoding="utf-8").strip() == "":
            raise ValueError("Empty hyperparam config file: %s" % yaml_path.name)
        names.append(yaml_path.stem)
    return names


def _selected_hyperparam_config_names(pytestconfig):
    """
    Informal specification: without --cn every hyperparam root is validated; with --cn only the requested
    root is validated and unknown names fail immediately with an explicit message.
    """
    available = _available_hyperparam_config_names()
    config_name = pytestconfig.getoption("--cn")
    if config_name is None:
        return available
    if config_name not in available:
        raise ValueError(
            "Unsupported hyperparam config '%s'. Allowed values: %s."
            % (config_name, ", ".join(available))
        )
    return [config_name]


def _runtime_overrides(pytestconfig):
    """
    Informal specification: user-provided override literals must be applied exactly as provided so the
    preflight test can emulate the same command-line override behavior as the sweep entrypoint.
    """
    return list(pytestconfig.getoption("--override"))


def _validate_hyperparam_top_level_keys(hyper_cfg, config_name):
    """
    Informal specification: hyperparam root configs must expose the minimum schema consumed by SweepRunner,
    otherwise sweeps should fail in preflight instead of failing later during optimization.
    """
    required = ["train_compose_config", "optimization_metric", "wandb_project", "wandb_offline", "sweeps"]
    for key in required:
        value = OmegaConf.select(hyper_cfg, key)
        if value is None:
            raise ValueError("Hyperparam config '%s' is missing required key '%s'." % (config_name, key))
    sweep_list = OmegaConf.select(hyper_cfg, "sweeps")
    if not sweep_list:
        raise ValueError("Hyperparam config '%s' has no sweeps configured." % config_name)


def _sweep_runtime_injected_overrides(hyper_cfg):
    """
    Informal specification: validation of sweep option overrides should include synthetic runtime fields that
    hyperparam_opt always injects, so train config compatibility is checked under real execution conditions.
    """
    optimization_metric = OmegaConf.select(hyper_cfg, "optimization_metric")
    wandb_project = OmegaConf.select(hyper_cfg, "wandb_project")
    wandb_offline = _bool_from_config_value(OmegaConf.select(hyper_cfg, "wandb_offline"))
    val_monitor_metric = _val_monitor_metric_from_optimization_metric(optimization_metric)
    return [
        "++wandb_run_name=config_validation",
        "++wandb_group=config_validation",
        "++wandb_project=%s" % str(wandb_project),
        "++wandb_offline=%s" % str(wandb_offline).lower(),
        "++val_monitor_metric=%s" % val_monitor_metric,
    ]


def _validate_train_config_from_hyper_sweep(train_config_name, merged_overrides):
    """
    Informal specification: every sweep option must compose into a valid train config with interpolations
    resolved and all expected train fields present, before any Optuna trial starts.
    """
    train_cfg = _compose_config(TRAIN_CONFIG_DIR, train_config_name, merged_overrides)
    TrainYaml(train_cfg)


def test_hyperparam_config_composition_and_sweep_override_compatibility(pytestconfig):
    """
    Informal specification: each selected hyperparam root must compose cleanly and every sweep option override
    must also compose against its target train root without unknown keys or broken interpolations.
    """
    names_to_validate = _selected_hyperparam_config_names(pytestconfig)
    cli_overrides = _runtime_overrides(pytestconfig)
    for config_name in names_to_validate:
        try:
            hyper_cfg = _compose_config(HYPERPARAM_CONFIG_DIR, config_name, cli_overrides)
            _validate_hyperparam_top_level_keys(hyper_cfg, config_name)
            train_config_name = str(OmegaConf.select(hyper_cfg, "train_compose_config"))
            injected = _sweep_runtime_injected_overrides(hyper_cfg)
            sweep_list = OmegaConf.select(hyper_cfg, "sweeps")
            for sweep in sweep_list:
                sweep_name = str(sweep.get("name"))
                fixed_overrides = list(sweep.get("fixed_overrides") or [])
                option_list = list(sweep.get("options") or [])
                if len(option_list) == 0:
                    raise ValueError(
                        "Hyperparam config '%s', sweep '%s' has no options."
                        % (config_name, sweep_name)
                    )
                for option in option_list:
                    option_label = str(option.get("label"))
                    option_overrides = list(option.get("overrides") or [])
                    merged = fixed_overrides + option_overrides + injected
                    _validate_train_config_from_hyper_sweep(train_config_name, merged)
        except Exception as exception:
            pytest.fail(
                "Hyperparam config validation failed for -cn %s with overrides %s.\n%s"
                % (config_name, cli_overrides, exception)
            )
