from pathlib import Path

from src.gnn.training.pareto_metric_config import resolve_pareto_metric_keys
from src.gnn.training.hyperparam_opt_helpers import save_pareto_plot


def test_resolve_pareto_metric_keys_uses_test_source():
    """
    Informal specification: selecting the `test` source should deterministically map to the canonical
    final test metrics, so users can switch target evaluation namespace with one config field.
    """
    x_key, y_key = resolve_pareto_metric_keys("test", None, None)
    assert x_key == "final_test/acc_all_rel_mae"
    assert y_key == "final_test/stress_rel_mae"


def test_resolve_pareto_metric_keys_uses_validation_source():
    """
    Informal specification: selecting the `validation` source should map to canonical validation metrics,
    so Pareto comparisons can be shifted from test to validation without changing plotting code.
    """
    x_key, y_key = resolve_pareto_metric_keys("validation", None, None)
    assert x_key == "final_validation/acc_all_rel_mae"
    assert y_key == "final_validation/stress_rel_mae"


def test_save_pareto_plot_handles_missing_keys_with_diagnostic(tmp_path):
    """
    Informal specification: when selected Pareto keys are absent in every trial record, plotting should
    still produce an explicit diagnostic image instead of silently rendering an empty/blank-looking chart.
    """
    records = [
        {"final_metrics": {"final_test/acc_all_rel_mae": 10.0, "final_test/stress_rel_mae": 2.0}},
        {"final_metrics": {"final_test/acc_all_rel_mae": 12.0, "final_test/stress_rel_mae": 1.8}},
    ]
    out_dir = Path(tmp_path)
    out_path = save_pareto_plot(records, "final_validation/acc_all_rel_mae", "final_validation/stress_rel_mae", out_dir)
    assert out_path == out_dir / "pareto.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0
