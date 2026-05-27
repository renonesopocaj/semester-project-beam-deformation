"""
Tests for _assert_target_mode_consistent in train.py.

Informal specification: when a sweep overrides target_mode in the train config
but the dataset's on-disk config.yaml was preprocessed with a different target_mode,
run_train_pipeline must raise a ValueError before any training begins rather than
silently running every trial with the dataset's mode (which would produce identical
metrics across all sweep options and is a hard-to-diagnose silent failure).
"""
import pytest
from omegaconf import OmegaConf

from src.gnn.misc_helpers.yamlobject import assert_target_mode_consistent


def _cfg(target_mode):
    """Build a minimal OmegaConf DictConfig with the given top-level target_mode."""
    return OmegaConf.create({"target_mode": target_mode})


def _cfg_no_target_mode():
    """Build a config that does not set target_mode at all (legacy / non-head sweeps)."""
    return OmegaConf.create({"seed": 42})


def test_mismatch_raises():
    """
    Tests that a train config with target_mode='full_stress' paired with a dataset
    whose config.yaml says target_mode='von_mises' raises a descriptive ValueError.
    """
    with pytest.raises(ValueError, match="was preprocessed with"):
        assert_target_mode_consistent(
            _cfg("full_stress"),
            {"target_mode": "von_mises"},
            "/fake/dataset/config.yaml",
        )


def test_mismatch_reverse_raises():
    """
    Tests that von_mises in train config vs full_stress in dataset also raises,
    not just the full_stress-to-von_mises direction.
    """
    with pytest.raises(ValueError, match="was preprocessed with"):
        assert_target_mode_consistent(
            _cfg("von_mises"),
            {"target_mode": "full_stress"},
            "/fake/dataset/config.yaml",
        )


def test_matching_full_stress_does_not_raise():
    """
    Tests that matching target_mode='full_stress' in both train config and dataset
    passes validation without raising.
    """
    assert_target_mode_consistent(
        _cfg("full_stress"),
        {"target_mode": "full_stress"},
        "/fake/dataset/config.yaml",
    )


def test_matching_von_mises_does_not_raise():
    """
    Tests that matching target_mode='von_mises' in both places passes validation.
    """
    assert_target_mode_consistent(
        _cfg("von_mises"),
        {"target_mode": "von_mises"},
        "/fake/dataset/config.yaml",
    )


def test_no_cfg_target_mode_skips_check():
    """
    Tests that when the train config does not declare target_mode (e.g. sweeps that do
    not touch the output head), no check is performed even if the dataset has a mode set.
    """
    assert_target_mode_consistent(
        _cfg_no_target_mode(),
        {"target_mode": "full_stress"},
        "/fake/dataset/config.yaml",
    )


def test_dataset_missing_target_mode_defaults_to_von_mises():
    """
    Tests that a dataset config.yaml without a target_mode key is treated as 'von_mises'
    (backward compatibility with datasets preprocessed before target_mode was introduced).
    Matching train config target_mode='von_mises' must therefore pass.
    """
    assert_target_mode_consistent(
        _cfg("von_mises"),
        {},  # no target_mode key -> defaults to von_mises
        "/fake/old_dataset/config.yaml",
    )


def test_error_message_contains_both_modes():
    """
    Tests that the raised ValueError names both the train-config mode and the dataset mode
    so the user can immediately understand what went wrong.
    """
    with pytest.raises(ValueError, match="was preprocessed with") as exc_info:
        assert_target_mode_consistent(
            _cfg("full_stress"),
            {"target_mode": "von_mises"},
            "/some/path/config.yaml",
        )
    assert "von_mises" in str(exc_info.value)
    assert "full_stress" in str(exc_info.value)
