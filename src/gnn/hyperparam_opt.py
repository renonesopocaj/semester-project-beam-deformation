"""
hyperparam_opt.py

Config-driven sequential Optuna grid sweeps; one W&B run per trial.
Training each trial composes the Hydra root config named by `train_compose_config` in
`conf/hyperparam/sweeps.yaml` (default stem: train_default_hyperopt), i.e.
`conf/train/train_default_hyperopt.yaml`. That file is a full train-style root (same fields
and defaults layout as `conf/train/train.yaml`); it is not read by `src/gnn/train.py`.
"""

# Default Hydra `config_name` under `conf/train/` when `train_compose_config` is unset in sweeps.yaml.
TRAIN_COMPOSE_CONFIG_DEFAULT = "train_default_hyperopt"
import json
import math
import traceback
from pathlib import Path
import torch
from src.gnn.training.hyperparam_opt_helpers import save_pareto_plot, save_summary_metric_plot, save_sweep_performance_plot, wandb_summary_log_outputs
import matplotlib
import matplotlib.pyplot as plt
import optuna
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from src.gnn.train import run_train_pipeline
from src.gnn.training.pareto_metric_config import resolve_pareto_metric_pairs
matplotlib.use("Agg")
plt.ioff()

def _resolve_offline_flag(raw_value):
    """
    Converts a raw config value (bool, None, or string) to a bool for wandb offline mode.

    Args:
        raw_value: bool or None or str
            Value from OmegaConf for the wandb_offline key.
    return: bool
        True when offline mode should be used.
    """
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return True
    if str(raw_value).strip().lower() in ("false", "0"):
        return False
    return True


def _suggest_from_search_space(trial, param_name, spec):
    """
    Sample one parameter value from an Optuna trial according to its distribution spec.

    Supported distributions:
    - uniform: suggest_float over [low, high].
    - log_uniform: suggest_float over [low, high] with log=True.
    - log_uniform_with_zero: with probability zero_probability samples 0.0, otherwise
      samples from log_uniform [low, high]. Uses two Optuna parameters internally so the
      TPE sampler can learn the zero/non-zero boundary independently of the value.
    - categorical: suggest_categorical over the provided values list.

    Args:
        trial: optuna.Trial
            Active Optuna trial.
        param_name: str
            Name used both as the Optuna parameter key and in the Hydra override.
        spec: dict
            Distribution specification from the search_space YAML node.
    return: object
        Sampled value (float, bool, or element of values list).
    """
    dist = spec["distribution"]
    if dist == "uniform":
        return trial.suggest_float(param_name, spec["low"], spec["high"])
    if dist == "log_uniform":
        return trial.suggest_float(param_name, spec["low"], spec["high"], log=True)
    if dist == "log_uniform_with_zero":
        selector = trial.suggest_float(f"{param_name}__zero_selector", 0.0, 1.0)
        if selector < spec["zero_probability"]:
            return 0.0
        return trial.suggest_float(f"{param_name}__value", spec["low"], spec["high"], log=True)
    if dist == "categorical":
        return trial.suggest_categorical(param_name, spec["values"])
    raise ValueError(f"Unknown distribution '{dist}' for search_space param '{param_name}'.")


def _format_override_value(value):
    """
    Serialize a sampled value for a Hydra override string.

    Args:
        value: object
            Sampled value (bool, float, int, or str).
    return: str
        String representation safe for use after '=' in a Hydra override.
    """
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value)


def format_swept_value_label(option_overrides):
    """
    Args:
        option_overrides: list of str
            Override literals like key=value for the active option.
    return: str
            One value label or a tuple-like label when multiple values are swept together.
    """
    values = []
    ix = 0
    while ix < len(option_overrides):
        ov = str(option_overrides[ix])
        if "=" in ov:
            values.append(ov.split("=", 1)[1])
        else:
            values.append(ov)
        ix = ix + 1
    if len(values) == 0:
        return "(no-option-overrides)"
    if len(values) == 1:
        return values[0]
    return "(" + ", ".join(values) + ")"

def _parse_val_monitor_metric(optimization_metric):
    """
    Maps an optimization_metric short name to the val_monitor_metric value expected by training.

    Supported values: overall_abs_mae → overall_abs_mae,
    overall_relative_mae (or the legacy full key val/overall_relative_mae) → overall_relative_mae.

    Args:
        optimization_metric: str
            Value from sweeps.yaml optimization_metric field.
    return: str
        val_monitor_metric override value for training.
    """
    norm = optimization_metric.strip().lower()
    if norm == "overall_abs_mae" or norm == "val/overall_abs_mae":
        return "overall_abs_mae"
    return "overall_relative_mae"


class SweepOption:
    """
    ??????????????
    """

    label: str
    overrides: list
    description: str

    def __init__(self, label, overrides, description):
        """
        Args:
            label: str
                Short stable id for logs.
            overrides: iterable
                Hydra override strings appended for this candidate.
            description: str
                Optional notes; unused by the runner unless you extend it.
        """
        self.label = label
        self.overrides = [str(x) for x in overrides]
        self.description = description


class Sweep:
    """
    Wrapper object for a sweep in the sense of iterable of hyperparameters.

    Supports two exclusive modes:
    - options mode: a finite list of SweepOption; uses GridSampler (existing behaviour).
    - search_space mode: a dict of parameter distribution specs; uses TPESampler with n_trials.
    Exactly one of options or search_space must be non-None.
    """

    name: str
    fixed_overrides: list
    options: list
    search_space: dict
    n_trials: int

    def __init__(self, name, fixed_overrides, options, search_space, n_trials):
        """
        Args:
            name: str
                Sweep identifier.
            fixed_overrides: iterable
                Overrides applied to every trial of this sweep.
            options: iterable of SweepOption or None
                Mutually exclusive choices for grid mode; None when search_space mode is used.
            search_space: dict or None
                Parameter distribution specs keyed by param name; None when options mode is used.
            n_trials: int or None
                Number of TPE trials; required and used only in search_space mode.
        """
        if (options is None) == (search_space is None):
            raise ValueError(
                f"Sweep '{name}': exactly one of 'options' or 'search_space' must be set."
            )
        self.name = name
        self.fixed_overrides = [str(x) for x in fixed_overrides]
        if options is not None:
            self.options = list(options)
            self.search_space = None
            self.n_trials = None
        else:
            self.options = []
            self.search_space = dict(search_space)
            self.n_trials = int(n_trials)


class SweepRunner:
    parent_cfg: DictConfig
    train_conf_dir: Path
    repo_root: Path
    optimization_metric_key: str
    final_optimization_metric_key: str
    val_monitor_metric_override: str
    pareto_pairs: list
    train_compose_config_name: str
    sweeps_list: list
    best_so_far_overrides: list
    trial_records: list
    per_sweep_plot_records: dict

    def __init__(self, parent_cfg):
        """
        Args:
            parent_cfg: DictConfig
                Resolved top-level Hydra hyperparam yaml (study_name, sweeps, wandb_*,
                train_compose_config naming conf/train/<stem>.yaml full train root, ...).
        """
        self.parent_cfg = parent_cfg
        self.repo_root = Path(__file__).resolve().parents[2]
        self.train_conf_dir = self.repo_root / "conf" / "train"
        OmegaConf.resolve(parent_cfg)
        tcn_raw = OmegaConf.select(parent_cfg, "train_compose_config")
        if tcn_raw is None or str(tcn_raw).strip() == "":
            self.train_compose_config_name = TRAIN_COMPOSE_CONFIG_DEFAULT
        else:
            self.train_compose_config_name = str(tcn_raw).strip()
        train_compose_path = self.train_conf_dir / ("%s.yaml" % self.train_compose_config_name)
        if not train_compose_path.is_file():
            raise FileNotFoundError(
                "Train compose config missing: %s (set train_compose_config in conf/hyperparam/sweeps.yaml)"
                % train_compose_path
            )
        raw_opt_metric = str(parent_cfg.optimization_metric).strip()
        self.optimization_metric_key = raw_opt_metric
        self.val_monitor_metric_override = _parse_val_monitor_metric(raw_opt_metric)
        if self.val_monitor_metric_override == "overall_abs_mae":
            self.final_optimization_metric_key = "final_validation/overall_abs_mae"
        else:
            self.final_optimization_metric_key = "final_validation/overall_relative_mae"
        self.pareto_pairs = resolve_pareto_metric_pairs(
            OmegaConf.select(parent_cfg, "pareto_metric_source"),
            OmegaConf.select(parent_cfg, "pareto_x_metric"),
            OmegaConf.select(parent_cfg, "pareto_y_metric"),
        )
        wandb_proj = str(OmegaConf.select(parent_cfg, "wandb_project"))
        self._wandb_project = wandb_proj
        self._wandb_offline_raw = OmegaConf.select(parent_cfg, "wandb_offline")
        self.best_so_far_overrides = []
        self.trial_records = []
        self.per_sweep_plot_records = {}
        self.output_dir = Path(str(OmegaConf.select(parent_cfg, "output_dir")))
        sweep_nodes = OmegaConf.select(parent_cfg, "sweeps")
        assembled = []
        # FIXME: what is to_container?
        for sweep_raw in OmegaConf.to_container(sweep_nodes, resolve=True):
            sweep_name = sweep_raw["name"]
            fixed_ov = sweep_raw.get("fixed_overrides", [])
            search_space_raw = sweep_raw.get("search_space")
            if search_space_raw is not None:
                # search_space mode: TPE-based continuous/categorical sampling
                assembled.append(Sweep(
                    name=sweep_name, fixed_overrides=fixed_ov,
                    options=None, search_space=search_space_raw,
                    n_trials=int(sweep_raw["n_trials"]),
                ))
            else:
                # options mode (existing): grid over explicit labeled overrides
                option_objs = []
                for opt_raw in sweep_raw["options"]:
                    label = opt_raw["label"]
                    ov_list = opt_raw["overrides"]
                    desc_cell = ""
                    if "description" in opt_raw and opt_raw["description"] is not None:
                        desc_cell = str(opt_raw["description"])
                    option_objs.append(SweepOption(label=label, overrides=ov_list, description=desc_cell))
                assembled.append(Sweep(
                    name=sweep_name, fixed_overrides=fixed_ov,
                    options=option_objs, search_space=None, n_trials=None,
                ))
        self.sweeps_list = assembled

    def compose_and_run_training(self, override_list):
        """
        Composes conf/train/<train_compose_config>.yaml (full root like train.yaml; see
        SweepRunner.train_compose_config_name) under conf/train then runs training once.

        Args:
            override_list: list of str
                Full Hydra override list including carried-over + trial-specific keys.
        return: triple
            (best_val_mae float, metrics dict str->float, wandb_run_id optional str)
        """
        # Clear the global dynamo compilation cache so each trial starts fresh.
        # Without this, the cache (keyed per code object, not per model instance) accumulates
        # compilations across trials and hits the recompile_limit (default 8) by Trial 1.
        torch._dynamo.reset()
        GlobalHydra.instance().clear()
        overrides_copy = []
        for one in override_list:
            overrides_copy.append(one)
        with initialize_config_dir(version_base=None, config_dir=str(self.train_conf_dir)):
            train_cfg_out = compose(config_name=self.train_compose_config_name, overrides=overrides_copy)
            return run_train_pipeline(train_cfg_out)

    def objective_for_sweep(self, sweep_entity):
        """
        Args:
            sweep_entity: Sweep
                Sweep for which this closure is constructed.
        return: callable
            Optuna objective function.
        """
        sweep_ref = sweep_entity
        study_name_plain = str(OmegaConf.select(self.parent_cfg, "study_name"))

        def optuna_objective(trial):
            # FIXME: what the fuck is this?
            choice_space = []
            ix = 0
            while ix < len(sweep_ref.options):
                choice_space.append(ix)
                ix = ix + 1
            option_idx = trial.suggest_categorical("option_idx", choice_space)
            chosen_option = sweep_ref.options[option_idx]
            merged = []
            bi = 0
            while bi < len(self.best_so_far_overrides):
                merged.append(self.best_so_far_overrides[bi])
                bi = bi + 1
            fi = 0
            while fi < len(sweep_ref.fixed_overrides):
                merged.append(sweep_ref.fixed_overrides[fi])
                fi = fi + 1
            oj = 0
            while oj < len(chosen_option.overrides):
                merged.append(chosen_option.overrides[oj])
                oj = oj + 1
            run_slug = "%s__%s__%s" % (study_name_plain, sweep_ref.name, chosen_option.label)
            merged.append("++wandb_run_name=%s" % run_slug)
            merged.append("++wandb_group=%s" % study_name_plain)
            merged.append("++wandb_project=%s" % self._wandb_project)
            off_lit = _resolve_offline_flag(self._wandb_offline_raw)
            merged.append("++wandb_offline=%s" % str(off_lit).lower())
            merged.append("++val_monitor_metric=%s" % self.val_monitor_metric_override)
            trial.set_user_attr("option_label_for_record", chosen_option.label)
            trial.set_user_attr("chosen_option_overrides", list(chosen_option.overrides))
            score_val_mae, metrics_plain, wb_id = self.compose_and_run_training(merged)
            # Prefer the final-evaluation metric over the in-training checkpoint best so that
            # Optuna and the sweep performance plot both show final_validation/* values.
            raw_final = metrics_plain.get(self.final_optimization_metric_key)
            if raw_final is not None and math.isfinite(float(raw_final)):
                opt_score = float(raw_final)
            else:
                opt_score = score_val_mae
            record = {"study_name": study_name_plain,
                      "sweep_name": sweep_ref.name,
                      "option_label": chosen_option.label,
                      "swept_value_label": format_swept_value_label(chosen_option.overrides),
                      "swept_overrides": list(chosen_option.overrides),
                      "full_overrides": merged,
                      "best_val_mae": opt_score,
                      "final_metrics": metrics_plain,
                      "wandb_run_id": wb_id}
            self.trial_records.append(record)
            pi = 0
            while pi < len(self.pareto_pairs):
                px_key, py_key = self.pareto_pairs[pi]
                trial.set_user_attr("pareto_x_%d" % pi, metrics_plain.get(px_key))
                trial.set_user_attr("pareto_y_%d" % pi, metrics_plain.get(py_key))
                pi = pi + 1
            return opt_score

        return optuna_objective

    def _objective_for_search_space_sweep(self, sweep_entity):
        """
        Args:
            sweep_entity: Sweep
                Sweep with search_space mode active.
        return: callable
            Optuna objective function using TPE sampling.
        """
        sweep_ref = sweep_entity
        study_name_plain = str(OmegaConf.select(self.parent_cfg, "study_name"))

        def optuna_objective(trial):
            sampled_overrides = []
            sampled_values = {}
            for param_name, spec in sweep_ref.search_space.items():
                value = _suggest_from_search_space(trial, param_name, spec)
                sampled_values[param_name] = value
                sampled_overrides.append(f"++{param_name}={_format_override_value(value)}")
            merged = list(self.best_so_far_overrides) + list(sweep_ref.fixed_overrides) + sampled_overrides
            run_slug = "%s__%s__trial%d" % (study_name_plain, sweep_ref.name, trial.number)
            merged.append("++wandb_run_name=%s" % run_slug)
            merged.append("++wandb_group=%s" % study_name_plain)
            merged.append("++wandb_project=%s" % self._wandb_project)
            off_lit = _resolve_offline_flag(self._wandb_offline_raw)
            merged.append("++wandb_offline=%s" % str(off_lit).lower())
            merged.append("++val_monitor_metric=%s" % self.val_monitor_metric_override)
            trial.set_user_attr("sampled_overrides", sampled_overrides)
            trial.set_user_attr("chosen_option_overrides", sampled_overrides)
            score_val_mae, metrics_plain, wb_id = self.compose_and_run_training(merged)
            raw_final = metrics_plain.get(self.final_optimization_metric_key)
            if raw_final is not None and math.isfinite(float(raw_final)):
                opt_score = float(raw_final)
            else:
                opt_score = score_val_mae
            record = {"study_name": study_name_plain,
                      "sweep_name": sweep_ref.name,
                      "option_label": "trial_%d" % trial.number,
                      "swept_value_label": str(sampled_values),
                      "swept_overrides": sampled_overrides,
                      "full_overrides": merged,
                      "best_val_mae": opt_score,
                      "final_metrics": metrics_plain,
                      "wandb_run_id": wb_id}
            self.trial_records.append(record)
            pi = 0
            while pi < len(self.pareto_pairs):
                px_key, py_key = self.pareto_pairs[pi]
                trial.set_user_attr("pareto_x_%d" % pi, metrics_plain.get(px_key))
                trial.set_user_attr("pareto_y_%d" % pi, metrics_plain.get(py_key))
                pi = pi + 1
            return opt_score

        return optuna_objective

    def _run_search_space_sweep(self, sweep_entity):
        """
        Runs a TPE-based Bayesian optimisation sweep over the declared search_space.
        Updates self.best_so_far_overrides with the best trial's sampled overrides.

        Args:
            sweep_entity: Sweep
                Sweep with search_space mode active and n_trials set.
        """
        sampler_obj = optuna.samplers.TPESampler()
        study_obj = optuna.create_study(direction="minimize", sampler=sampler_obj)
        fun = self._objective_for_search_space_sweep(sweep_entity)
        study_obj.optimize(fun, n_trials=sweep_entity.n_trials)
        x_labels = ["trial_%d" % t.number for t in study_obj.trials]
        y_vals = [float(t.value) for t in study_obj.trials if t.value is not None]
        x_labels = x_labels[:len(y_vals)]
        sweep_plot_path = save_sweep_performance_plot(
            sweep_entity.name, x_labels, y_vals,
            self.final_optimization_metric_key, self.output_dir,
        )
        self.per_sweep_plot_records[sweep_entity.name] = str(sweep_plot_path)
        print("Saved sweep performance plot to %s" % sweep_plot_path)
        best_trial = study_obj.best_trial
        for ov in best_trial.user_attrs["sampled_overrides"]:
            self.best_so_far_overrides.append(ov)

    def run_one_sweep(self, sweep_entity):
        """
        Dispatches to grid (options) or TPE (search_space) sweep runner, then updates
        self.best_so_far_overrides with the winning configuration's overrides.

        Args:
            sweep_entity: Sweep
                One sweep stage.
        """
        if sweep_entity.search_space is not None:
            self._run_search_space_sweep(sweep_entity)
            return
        # create optuna objects for sweeping
        grid_map = {"option_idx": list(range(len(sweep_entity.options)))}
        sampler_obj = optuna.samplers.GridSampler(grid_map)
        study_obj = optuna.create_study(direction="minimize", sampler=sampler_obj)
        # FIXME: ?????
        fun = self.objective_for_sweep(sweep_entity)
        # Run one optuna trial via .optimize
        study_obj.optimize(fun, n_trials=len(sweep_entity.options))
        # Collect one point per option for the current sweep in config order.
        option_to_metric = {}
        tr_i = 0
        while tr_i < len(study_obj.trials):
            one_trial = study_obj.trials[tr_i]
            op_idx = one_trial.params.get("option_idx")
            if op_idx is not None:
                option_to_metric[int(op_idx)] = float(one_trial.value)
            tr_i = tr_i + 1
        x_labels = []
        y_vals = []
        oi = 0
        while oi < len(sweep_entity.options):
            one_option = sweep_entity.options[oi]
            if oi in option_to_metric:
                x_labels.append(format_swept_value_label(one_option.overrides))
                y_vals.append(option_to_metric[oi])
            oi = oi + 1
        sweep_plot_path = save_sweep_performance_plot(
            sweep_entity.name,
            x_labels,
            y_vals,
            self.final_optimization_metric_key,
            self.output_dir,
        )
        self.per_sweep_plot_records[sweep_entity.name] = str(sweep_plot_path)
        print("Saved sweep performance plot to %s" % sweep_plot_path)
        best_trial = study_obj.best_trial
        extra = best_trial.user_attrs["chosen_option_overrides"]
        ei = 0
        while ei < len(extra):
            self.best_so_far_overrides.append(extra[ei])
            ei = ei + 1

    def run_all_sweeps(self):
        """
        Runs every configured sweep in order.
        """
        si = 0
        while si < len(self.sweeps_list):
            self.run_one_sweep(self.sweeps_list[si])
            si = si + 1

    def report_global_best_run(self):
        """
        return: dict | None
            Record with minimum best_val_mae (final_validation/* score for the configured
            optimization metric) or None if no trials were run.
        """
        if len(self.trial_records) == 0:
            print("No trial records; nothing to report.")
            return None
        best_rec = None
        br = 0
        while br < len(self.trial_records):
            candidate = self.trial_records[br]
            if best_rec is None:
                best_rec = candidate
            else:
                if candidate["best_val_mae"] < best_rec["best_val_mae"]:
                    best_rec = candidate
            br = br + 1
        print("\n" + "=" * 80 + "\nBest run by %s among all trials (smaller is better):" % self.final_optimization_metric_key)
        print(json.dumps(best_rec, indent=2, default=str) + "\n" + "=" * 80 + "\n")
        return best_rec

@hydra.main(version_base=None, config_path="../../conf/hyperparam", config_name="sanity_sweep")
def main(cfg):
    """
    Args:
        cfg: DictConfig
            Hyperparam driver yaml (study, sweeps, pareto keys, train_compose_config, ...).
    """
    OmegaConf.resolve(cfg)
    runner = SweepRunner(cfg)
    runner.run_all_sweeps()

    outp = Path(str(OmegaConf.select(cfg, "output_dir")))
    outp.mkdir(parents=True, exist_ok=True)
    dumped_path = outp / "all_trials.json"
    with open(dumped_path, "w", encoding="utf-8") as fh_out:
        json.dump(runner.trial_records, fh_out, indent=2, default=str)

    runner.report_global_best_run()

    off_lit = _resolve_offline_flag(OmegaConf.select(cfg, "wandb_offline"))
    study_nm = str(OmegaConf.select(cfg, "study_name"))
    wandb_proj = str(OmegaConf.select(cfg, "wandb_project"))
    try:
        # Build pareto jobs: for each configured (x, y) pair produce one plot per split
        # (test and validation), then add the rollout-sum pareto for both splits.
        pareto_jobs = []
        pi = 0
        while pi < len(runner.pareto_pairs):
            px_key, py_key = runner.pareto_pairs[pi]
            # Detect which split the configured pair belongs to and build both variants.
            if "final_test/" in px_key or "final_test/" in py_key:
                original_split = "test"
                sibling_split = "validation"
            elif "final_validation/" in px_key or "final_validation/" in py_key:
                original_split = "validation"
                sibling_split = "test"
            else:
                # No recognisable prefix — keep only the original pair.
                pareto_jobs.append((px_key, py_key, "", "pareto_%d.png" % pi))
                pi = pi + 1
                continue
            # Original split
            pareto_jobs.append((
                px_key, py_key,
                original_split,
                "pareto_%d_%s.png" % (pi, original_split),
            ))
            # Sibling split — swap the prefix in both keys.
            sibling_px = px_key.replace("final_%s/" % original_split, "final_%s/" % sibling_split)
            sibling_py = py_key.replace("final_%s/" % original_split, "final_%s/" % sibling_split)
            pareto_jobs.append((
                sibling_px, sibling_py,
                sibling_split,
                "pareto_%d_%s.png" % (pi, sibling_split),
            ))
            pi = pi + 1
        # Rollout-sum pareto: x = rollout_pos_mae + rollout_volume_mae, y = overall_abs_mae
        for rollout_split in ("test", "validation"):
            pareto_jobs.append((
                ["final_%s/rollout_pos_mae" % rollout_split, "final_%s/rollout_volume_mae" % rollout_split],
                "final_%s/overall_abs_mae" % rollout_split,
                rollout_split,
                "pareto_rollout_sum_%s.png" % rollout_split,
            ))
        pareto_plot_paths = []
        ji = 0
        while ji < len(pareto_jobs):
            jx_key, jy_key, j_split, j_filename = pareto_jobs[ji]
            png_path_here = save_pareto_plot(runner.trial_records, jx_key, jy_key, outp, j_filename, j_split)
            pareto_plot_paths.append(png_path_here)
            print("Saved Pareto plot '%s' to %s" % (j_filename, png_path_here))
            ji = ji + 1

        # Fixed summary plots across all trials: final_test and final_validation splits.
        summary_metric_specs = [
            ("final_test/overall_abs_mae", "summary_test_abs_mae.png"),
            ("final_test/overall_relative_mae", "summary_test_rel_mae.png"),
            ("final_validation/overall_abs_mae", "summary_val_abs_mae.png"),
            ("final_validation/overall_relative_mae", "summary_val_rel_mae.png"),
            ("final_test/rollout_pos_mae", "summary_test_rollout_pos_mae.png"),
            ("final_validation/rollout_pos_mae", "summary_val_rollout_pos_mae.png"),
            ("final_test/rollout_volume_mae", "summary_test_vol_mae.png"),
            ("final_validation/rollout_volume_mae", "summary_val_vol_mae.png"),
        ]
        summary_plot_paths = []
        sm = 0
        while sm < len(summary_metric_specs):
            metric_key_here, filename_here = summary_metric_specs[sm]
            sp = save_summary_metric_plot(runner.trial_records, metric_key_here, outp, filename_here)
            summary_plot_paths.append(sp)
            print("Saved summary plot (%s) to %s" % (metric_key_here, sp))
            sm = sm + 1

        # Log all hyperparam_outputs files to one trailing W&B summary run.
        sweep_plot_paths = []
        sw = 0
        while sw < len(runner.sweeps_list):
            sweep_name = runner.sweeps_list[sw].name
            if sweep_name in runner.per_sweep_plot_records:
                sweep_plot_paths.append(Path(runner.per_sweep_plot_records[sweep_name]))
            sw = sw + 1
        all_plot_paths = sweep_plot_paths + list(pareto_plot_paths) + summary_plot_paths
        wandb_summary_log_outputs(study_nm, all_plot_paths, [dumped_path], wandb_proj, off_lit)
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)

if __name__ == "__main__":
    main()
