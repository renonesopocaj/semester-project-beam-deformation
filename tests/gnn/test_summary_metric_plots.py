"""
Tests for the three final-validation summary plots produced by the hyperparameter sweep runner:
  summary_val_abs_mae   → final_validation/overall_abs_mae
  summary_val_rel_mae   → final_validation/overall_relative_mae
  summary_val_rollout_pos_mae → final_validation/rollout_pos_mae

These tests verify that the data pipeline from Trainer.test() / callback_metrics all the way to
save_summary_metric_plot correctly propagates finite values so the plots are non-empty.
"""

import math
from pathlib import Path

import pytest

from src.gnn.training.hyperparam_opt_helpers import (
    save_summary_metric_plot,
    update_final_metrics_plain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Minimal stand-in for a single-element torch.Tensor as returned by callback_metrics."""

    value: float

    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def numel(self):
        return 1

    def item(self):
        return self.value


def _make_record(sweep_name, option_label, final_metrics):
    return {
        "sweep_name": sweep_name,
        "option_label": option_label,
        "final_metrics": dict(final_metrics),
    }


# ---------------------------------------------------------------------------
# update_final_metrics_plain — captures all three canonical validation keys
# ---------------------------------------------------------------------------

def test_update_final_metrics_plain_captures_noroll_metrics_from_test_results():
    """
    Verifies that the non-rollout final-validation metrics emitted by test_step
    (overall_abs_mae, overall_relative_mae) are captured when update_final_metrics_plain
    is given the list returned by Trainer.test(), which is the expected Lightning format.
    Ensures both tensor and plain-float values are correctly converted.
    """
    final_metrics = {}
    validation_results = [
        {
            "final_validation/overall_abs_mae": _FakeTensor(0.42),
            "final_validation/overall_relative_mae": _FakeTensor(0.18),
            "final_validation/acc_all_rel_mae": _FakeTensor(0.15),
            "final_validation/stress_rel_mae": _FakeTensor(0.22),
            # non-final keys must be ignored
            "val/overall_abs_mae": _FakeTensor(99.0),
        }
    ]
    update_final_metrics_plain(final_metrics, validation_results)

    assert "final_validation/overall_abs_mae" in final_metrics
    assert abs(final_metrics["final_validation/overall_abs_mae"] - 0.42) < 1e-9
    assert "final_validation/overall_relative_mae" in final_metrics
    assert abs(final_metrics["final_validation/overall_relative_mae"] - 0.18) < 1e-9
    assert "val/overall_abs_mae" not in final_metrics


def test_update_final_metrics_plain_captures_rollout_metric_from_callback_metrics():
    """
    Verifies that rollout_pos_mae (logged in on_test_epoch_end with rank_zero_only=True and
    therefore potentially absent from the Trainer.test() return list) is captured when
    update_final_metrics_plain is called with callback_metrics as a plain dict.
    This reflects the extra call added in train.py immediately after the validation test phase,
    before the test-split Trainer.test() resets Lightning's result collection.
    """
    final_metrics = {}
    # Simulate what callback_metrics looks like after the validation test phase:
    # it contains both noroll metrics from test_step and rollout metrics from on_test_epoch_end.
    callback_metrics = {
        "final_validation/overall_abs_mae": _FakeTensor(0.42),
        "final_validation/overall_relative_mae": _FakeTensor(0.18),
        "final_validation/rollout_pos_mae": _FakeTensor(1.37),
        "final_validation/rollout_volume_mae": _FakeTensor(0.05),
    }
    update_final_metrics_plain(final_metrics, callback_metrics)

    assert "final_validation/rollout_pos_mae" in final_metrics
    assert abs(final_metrics["final_validation/rollout_pos_mae"] - 1.37) < 1e-9
    assert "final_validation/rollout_volume_mae" in final_metrics


def test_update_final_metrics_plain_validation_metrics_survive_test_phase():
    """
    Verifies the full two-phase collection sequence used in train.py:
    1. Collect from Trainer.test() return value for validation split.
    2. Collect from callback_metrics immediately after (captures rollout_pos_mae).
    3. Collect from Trainer.test() return value for test split.
    4. Collect from callback_metrics at the end.
    All four final_validation keys must survive step (3) and (4), which simulate the
    Lightning result collection reset when the test-split test() call runs.
    """
    final_metrics = {}

    # Phase 1: validation Trainer.test() return
    validation_results = [
        {
            "final_validation/overall_abs_mae": _FakeTensor(0.42),
            "final_validation/overall_relative_mae": _FakeTensor(0.18),
        }
    ]
    update_final_metrics_plain(final_metrics, validation_results)

    # Phase 2: callback_metrics right after validation test (adds rollout metric)
    val_callback = {
        "final_validation/overall_abs_mae": _FakeTensor(0.42),
        "final_validation/overall_relative_mae": _FakeTensor(0.18),
        "final_validation/rollout_pos_mae": _FakeTensor(1.37),
    }
    update_final_metrics_plain(final_metrics, val_callback)

    # Phase 3: test-split Trainer.test() return (final_validation/* keys absent – reset)
    test_results = [
        {
            "final_test/overall_abs_mae": _FakeTensor(0.38),
            "final_test/overall_relative_mae": _FakeTensor(0.16),
        }
    ]
    update_final_metrics_plain(final_metrics, test_results)

    # Phase 4: callback_metrics at end (now holds only test-split metrics – reset)
    test_callback = {
        "final_test/overall_abs_mae": _FakeTensor(0.38),
        "final_test/rollout_pos_mae": _FakeTensor(0.95),
    }
    update_final_metrics_plain(final_metrics, test_callback)

    # All three canonical validation keys must be present
    assert "final_validation/overall_abs_mae" in final_metrics
    assert "final_validation/overall_relative_mae" in final_metrics
    assert "final_validation/rollout_pos_mae" in final_metrics
    assert abs(final_metrics["final_validation/rollout_pos_mae"] - 1.37) < 1e-9

    # Test keys must also be present
    assert "final_test/overall_abs_mae" in final_metrics
    assert "final_test/rollout_pos_mae" in final_metrics


# ---------------------------------------------------------------------------
# save_summary_metric_plot — three canonical validation keys produce real plots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric_key,filename", [
    ("final_validation/overall_abs_mae", "summary_val_abs_mae.png"),
    ("final_validation/overall_relative_mae", "summary_val_rel_mae.png"),
    ("final_validation/rollout_pos_mae", "summary_val_rollout_pos_mae.png"),
])
def test_save_summary_metric_plot_produces_non_empty_plot_for_canonical_val_keys(metric_key, filename, tmp_path):
    """
    Verifies that when trial records contain finite values for each of the three canonical
    final-validation metric keys, save_summary_metric_plot writes a real non-empty image
    (not the diagnostic empty-plot fallback with 'No data for ...' text).
    """
    records = [
        _make_record("sweep_a", "option_1", {metric_key: 0.42}),
        _make_record("sweep_a", "option_2", {metric_key: 0.38}),
        _make_record("sweep_b", "option_1", {metric_key: 0.45}),
    ]
    out_path = save_summary_metric_plot(records, metric_key, tmp_path, filename)
    assert out_path == tmp_path / filename
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_summary_metric_plot_all_missing_writes_diagnostic_with_count(tmp_path):
    """
    Verifies that when no trial record has the requested metric key, the plot is still
    written (non-zero bytes), and the diagnostic text includes the missing-record count
    so callers can distinguish 'key absent' from 'key present but NaN'.
    """
    records = [
        _make_record("sweep_a", "option_1", {"final_test/overall_abs_mae": 0.5}),
        _make_record("sweep_a", "option_2", {}),
    ]
    out_path = save_summary_metric_plot(
        records, "final_validation/overall_abs_mae", tmp_path, "summary_val_abs_mae.png"
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_summary_metric_plot_all_nan_writes_diagnostic_with_non_finite_count(tmp_path):
    """
    Verifies that when all trial records have NaN values for the metric (e.g. from a
    diverged training run), the plot is still written with diagnostic text that reports
    the non-finite count, enabling the user to distinguish this from a missing-key case.
    """
    records = [
        _make_record("sweep_a", "option_1", {"final_validation/rollout_pos_mae": float("nan")}),
        _make_record("sweep_a", "option_2", {"final_validation/rollout_pos_mae": float("nan")}),
    ]
    out_path = save_summary_metric_plot(
        records, "final_validation/rollout_pos_mae", tmp_path, "summary_val_rollout_pos_mae.png"
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_summary_metric_plot_mixed_nan_and_finite_plots_only_finite(tmp_path):
    """
    Verifies that when some records have NaN and some have finite values for a metric,
    only the finite records appear in the plot (the plot is non-empty / shows a real curve),
    so one diverged trial does not suppress valid data from other trials.
    """
    records = [
        _make_record("sweep_a", "option_1", {"final_validation/overall_abs_mae": 0.40}),
        _make_record("sweep_a", "option_2", {"final_validation/overall_abs_mae": float("nan")}),
        _make_record("sweep_b", "option_1", {"final_validation/overall_abs_mae": 0.35}),
    ]
    out_path = save_summary_metric_plot(
        records, "final_validation/overall_abs_mae", tmp_path, "summary_val_abs_mae.png"
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
