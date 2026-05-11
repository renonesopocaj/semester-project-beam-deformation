from pathlib import Path
import sys

from src.gnn.training.pareto_metric_config import resolve_pareto_metric_pairs
from src.gnn.training.hyperparam_opt_helpers import (save_pareto_plot, save_sweep_performance_plot,
                                                     update_final_metrics_plain, wandb_summary_log_outputs)


class _FakeWandbRun:
    logged_payloads: list[dict]
    logged_artifacts: list

    def __init__(self):
        self.logged_payloads = []
        self.logged_artifacts = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        return False

    def log(self, payload):
        self.logged_payloads.append(payload)

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)


class _FakeWandbArtifact:
    name: str
    type: str
    files: list[tuple[str, str]]

    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path, name):
        self.files.append((path, name))


class _FakeWandbModule:
    init_calls: list[dict]
    image_calls: list[tuple[str, str]]
    artifact_calls: list[_FakeWandbArtifact]
    run: _FakeWandbRun

    def __init__(self):
        self.init_calls = []
        self.image_calls = []
        self.artifact_calls = []
        self.run = _FakeWandbRun()

    def init(self, project, name, group, reinit, mode):
        self.init_calls.append({"project": project, "name": name, "group": group, "reinit": reinit, "mode": mode})
        return self.run

    def Image(self, path, caption):
        self.image_calls.append((path, caption))
        return {"path": path, "caption": caption}

    def Artifact(self, name, type):
        artifact = _FakeWandbArtifact(name, type)
        self.artifact_calls.append(artifact)
        return artifact


class _FakeScalar:
    value: float

    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def numel(self):
        return 1

    def item(self):
        return self.value


def test_resolve_pareto_metric_pairs_uses_test_source():
    """
    Informal specification: selecting the `test` source should deterministically map to the canonical
    final test metrics, so users can switch target evaluation namespace with one config field.
    """
    pairs = resolve_pareto_metric_pairs("test", None, None)
    x_key, y_key = pairs[0]
    assert x_key == "final_test/acc_all_rel_mae"
    assert y_key == "final_test/stress_rel_mae"


def test_resolve_pareto_metric_pairs_uses_validation_source():
    """
    Informal specification: selecting the `validation` source should map to canonical validation metrics,
    so Pareto comparisons can be shifted from test to validation without changing plotting code.
    """
    pairs = resolve_pareto_metric_pairs("validation", None, None)
    x_key, y_key = pairs[0]
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


def test_update_final_metrics_plain_keeps_validation_metrics_after_test_results():
    """
    Informal specification: hyperparameter summaries should preserve final-validation scalars collected
    from the validation evaluation even after the later test evaluation contributes final-test scalars.
    """
    final_metrics = {}
    validation_results = [{"final_validation/overall_abs_mae": _FakeScalar(1.25),
                           "final_validation/rollout_pos_mae": 2.5,
                           "val/overall_abs_mae": 9.0}]
    test_results = [{"final_test/overall_abs_mae": _FakeScalar(0.75)}]

    update_final_metrics_plain(final_metrics, validation_results)
    update_final_metrics_plain(final_metrics, test_results)

    assert final_metrics == {
        "final_validation/overall_abs_mae": 1.25,
        "final_validation/rollout_pos_mae": 2.5,
        "final_test/overall_abs_mae": 0.75,
    }


def test_save_sweep_performance_plot_writes_curve_image(tmp_path):
    """
    Informal specification: a sweep-level performance trend should always render as a concrete image
    when x labels and metric values are provided, so each sweep can be inspected independently of Pareto.
    """
    out_dir = Path(tmp_path)
    out_path = save_sweep_performance_plot("noise", ["0.0001", "0.0005", "0.001"], [1.2, 1.1, 1.3], "val/overall_relative_mae", out_dir)
    assert out_path == out_dir / "sweep_noise_overall_mae.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_wandb_summary_log_outputs_uses_one_run_for_all_outputs(tmp_path, monkeypatch):
    """
    Informal specification: summary logging must create exactly one W&B run for the post-sweep outputs,
    while putting every generated plot and the trial JSON into that one run.
    """
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    plot_a = Path(tmp_path) / "pareto_0.png"
    plot_b = Path(tmp_path) / "summary_val_abs_mae.png"
    trials_json = Path(tmp_path) / "all_trials.json"
    plot_a.write_bytes(b"plot-a")
    plot_b.write_bytes(b"plot-b")
    trials_json.write_text("[]", encoding="utf-8")

    wandb_summary_log_outputs("force geom/study", [plot_a, plot_b], [trials_json], "semester-project", True)

    assert len(fake_wandb.init_calls) == 1
    assert fake_wandb.init_calls[0]["name"] == "force geom/study__summary"
    assert fake_wandb.init_calls[0]["mode"] == "offline"
    assert len(fake_wandb.run.logged_payloads) == 1
    assert sorted(fake_wandb.run.logged_payloads[0].keys()) == [
        "hyperparam_outputs/pareto_0",
        "hyperparam_outputs/summary_val_abs_mae",
    ]
    assert len(fake_wandb.image_calls) == 2
    assert len(fake_wandb.run.logged_artifacts) == 1
    assert fake_wandb.run.logged_artifacts[0].files == [(str(trials_json), "all_trials.json")]
