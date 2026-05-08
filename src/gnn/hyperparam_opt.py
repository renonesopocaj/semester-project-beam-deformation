"""
hyperparam_opt.py

Config-driven sequential Optuna grid sweeps; one W&B run per trial.
Training each trial composes the Hydra root config named by `train_compose_config` in
`conf/hyperparam/sweeps.yaml` (default stem: train_default_hyperopt), i.e.
`conf/train/train_default_hyperopt.yaml`. That file is a full train-style root (same fields
and defaults layout as `conf/train/train.yaml`); it is not read by `src/gnn/train.py`.
FIXME I Don't understand: does this run multiple sweeps over different hyperparameters in one run (python , or one sweep
 in one time?
"""

# Default Hydra `config_name` under `conf/train/` when `train_compose_config` is unset in sweeps.yaml.
TRAIN_COMPOSE_CONFIG_DEFAULT = "train_default_hyperopt"

import json
import math
from pathlib import Path
import wandb as wb_sdk
from src.gnn.training.hyperparam_opt_helpers import save_pareto_plot, wandb_summary_log_plot
import matplotlib
import matplotlib.pyplot as plt
import optuna
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from src.gnn.train import run_train_pipeline
matplotlib.use("Agg")
plt.ioff()

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
    Wrapper object for a sweep in the sense of iterable of hyperparameters
    """

    name: str
    fixed_overrides: list
    options: list

    def __init__(self, name, fixed_overrides, options):
        """
        Args:
            name: str
                Sweep identifier.
            fixed_overrides: iterable
                Overrides applied to every trial of this sweep.
            options: iterable of SweepOption
                Mutually exclusive choices (grid).
        """
        self.name = name
        self.fixed_overrides = [str(x) for x in fixed_overrides]
        sweep_options = []
        for option in options:
            sweep_options.append(option)
        self.options = sweep_options


class SweepRunner:
    parent_cfg: DictConfig
    train_conf_dir: Path
    repo_root: Path
    optimization_metric_key: str
    pareto_x_key: str
    pareto_y_key: str
    train_compose_config_name: str
    sweeps_list: list
    best_so_far_overrides: list
    trial_records: list

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
        self.optimization_metric_key = str(parent_cfg.optimization_metric)
        self.pareto_x_key = str(parent_cfg.pareto_x_metric)
        self.pareto_y_key = str(parent_cfg.pareto_y_metric)
        wandb_proj = str(OmegaConf.select(parent_cfg, "wandb_project"))
        self._wandb_project = wandb_proj
        self._wandb_offline_raw = OmegaConf.select(parent_cfg, "wandb_offline")
        self.best_so_far_overrides = []
        self.trial_records = []
        sweep_nodes = OmegaConf.select(parent_cfg, "sweeps")
        assembled = []
        # FIXME: what is to_container?
        for sweep_raw in OmegaConf.to_container(sweep_nodes, resolve=True):
            sweep_name = sweep_raw["name"]
            fixed_ov = sweep_raw.get("fixed_overrides", [])
            option_objs = []
            for opt_raw in sweep_raw["options"]:
                label = opt_raw["label"]
                ov_list = opt_raw["overrides"]
                desc_cell = ""
                if "description" in opt_raw and opt_raw["description"] is not None:
                    desc_cell = str(opt_raw["description"])
                option_objs.append(SweepOption(label=label, overrides=ov_list, description=desc_cell))
            assembled.append(Sweep(name=sweep_name, fixed_overrides=fixed_ov, options=option_objs))
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
            merged.append("+wandb_run_name=%s" % run_slug)
            merged.append("+wandb_group=%s" % study_name_plain)
            merged.append("+wandb_project=%s" % self._wandb_project)
            off_raw = self._wandb_offline_raw
            # FIXME: FUCKING CHANGE this fucking boilerplate shit
            if isinstance(off_raw, bool):
                off_lit = off_raw
            elif off_raw is None:
                off_lit = True
            elif str(off_raw).strip().lower() in ("false", "0"):
                off_lit = False
            else:
                off_lit = True
            merged.append("+wandb_offline=%s" % str(off_lit).lower())
            trial.set_user_attr("option_label_for_record", chosen_option.label)
            trial.set_user_attr("chosen_option_overrides", list(chosen_option.overrides))
            score_val_mae, metrics_plain, wb_id = self.compose_and_run_training(merged)
            record = {"study_name": study_name_plain,
                      "sweep_name": sweep_ref.name,
                      "option_label": chosen_option.label,
                      "full_overrides": merged,
                      "best_val_relative_mae": score_val_mae,
                      "final_metrics": metrics_plain,
                      "wandb_run_id": wb_id}
            self.trial_records.append(record)
            trial.set_user_attr("pareto_x", metrics_plain.get(self.pareto_x_key))
            trial.set_user_attr("pareto_y", metrics_plain.get(self.pareto_y_key))
            return score_val_mae

        return optuna_objective

    def run_one_sweep(self, sweep_entity):
        """
        Updates self.best_so_far_overrides with the winning option's overrides.

        Args:
            sweep_entity: Sweep
                One sweep stage.
        """
        # create optuna objects for sweeping
        grid_map = {"option_idx": list(range(len(sweep_entity.options)))}
        sampler_obj = optuna.samplers.GridSampler(grid_map)
        study_obj = optuna.create_study(direction="minimize", sampler=sampler_obj)
        # FIXME: ?????
        fun = self.objective_for_sweep(sweep_entity)
        # Run one optuna trial via .optimize
        study_obj.optimize(fun, n_trials=len(sweep_entity.options))
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
            Record with minimum best_val_relative_mae or None if empty.
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
                if candidate["best_val_relative_mae"] < best_rec["best_val_relative_mae"]:
                    best_rec = candidate
            br = br + 1
        print("\n" + "=" * 80 + "\nBest run by %s among all trials (smaller is better):" % self.optimization_metric_key)
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
    png_path_here = save_pareto_plot(runner.trial_records, runner.pareto_x_key, runner.pareto_y_key, outp)
    print("Saved Pareto plot to %s" % png_path_here)

    off_raw = OmegaConf.select(cfg, "wandb_offline")
    # FIXME weird fucking shit change this fucking boilerplate it's useless
    if isinstance(off_raw, bool):
        off_lit = off_raw
    elif off_raw is None:
        off_lit = True
    elif str(off_raw).strip().lower() in ("false", "0"):
        off_lit = False
    else:
        off_lit = True
    # FIXME: select?
    study_nm = str(OmegaConf.select(cfg, "study_name"))
    wandb_proj = str(OmegaConf.select(cfg, "wandb_project"))
    wandb_summary_log_plot(study_nm, png_path_here, wandb_proj, off_lit)


if __name__ == "__main__":
    main()
