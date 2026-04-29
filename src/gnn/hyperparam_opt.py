"""
hyperparam_opt.py

Config-driven sequential Optuna grid sweeps over Hydra train configs; one W&B run per trial.
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import optuna
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.gnn.train import run_train_pipeline

matplotlib.use("Agg")
plt.ioff()


class SweepOption:
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
        return: None
        """
        self.label = label
        self.overrides = [str(x) for x in overrides]
        self.description = description


class Sweep:
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
        return: None
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
    sweeps_list: list
    best_so_far_overrides: list
    trial_records: list

    def __init__(self, parent_cfg):
        """
        Args:
            parent_cfg: DictConfig
                Resolved top-level Hydra hyperparam yaml (study_name, sweeps, wandb_* , ...).
        return: None
        """
        self.parent_cfg = parent_cfg
        self.repo_root = Path(__file__).resolve().parents[2]
        self.train_conf_dir = self.repo_root / "conf" / "train"
        OmegaConf.resolve(parent_cfg)
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
        Composes conf/train/train.yaml under conf/train then runs training once.

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
            train_cfg_out = compose(config_name="train", overrides=overrides_copy)
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
            record = {
                "study_name": study_name_plain,
                "sweep_name": sweep_ref.name,
                "option_label": chosen_option.label,
                "full_overrides": merged,
                "best_val_relative_mae": score_val_mae,
                "final_metrics": metrics_plain,
                "wandb_run_id": wb_id,
            }
            self.trial_records.append(record)
            trial.set_user_attr("pareto_x", metrics_plain.get(self.pareto_x_key, float("nan")))
            trial.set_user_attr("pareto_y", metrics_plain.get(self.pareto_y_key, float("nan")))
            if math.isnan(score_val_mae):
                return 1e30
            return score_val_mae

        return optuna_objective

    def run_one_sweep(self, sweep_entity):
        """
        Args:
            sweep_entity: Sweep
                One sweep stage.
        return: None
                Updates self.best_so_far_overrides with the winning option's overrides.
        """
        grid_map = {"option_idx": list(range(len(sweep_entity.options)))}
        sampler_obj = optuna.samplers.GridSampler(grid_map)
        study_obj = optuna.create_study(direction="minimize", sampler=sampler_obj)
        fun = self.objective_for_sweep(sweep_entity)
        study_obj.optimize(fun, n_trials=len(sweep_entity.options))
        best_trial = study_obj.best_trial
        extra = best_trial.user_attrs["chosen_option_overrides"]
        ei = 0
        while ei < len(extra):
            self.best_so_far_overrides.append(extra[ei])
            ei = ei + 1

    def run_all_sweeps(self):
        """
        return: None
            Runs every configured sweep in order.
        """
        si = 0
        while si < len(self.sweeps_list):
            self.run_one_sweep(self.sweeps_list[si])
            si = si + 1

    def report_global_best(self):
        """
        return: dict or None
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
        print("\n" + "=" * 80)
        print("Best run by %s among all trials (smaller is better):" % self.optimization_metric_key)
        print(json.dumps(best_rec, indent=2, default=str))
        print("=" * 80 + "\n")
        return best_rec


def pareto_non_dominated_minimize(points_xy):
    """
    Args:
        points_xy: list of pair (float, float)
            Objective pairs to minimize jointly.
    return: list[int]
            Indices into points_xy that lie on the Pareto frontier (non-dominated).
    """
    n = len(points_xy)
    out_idx = []
    i = 0
    while i < n:
        dominated = False
        j = 0
        while j < n:
            if j == i:
                j = j + 1
                continue
            xi, yi = points_xy[i]
            xj, yj = points_xy[j]
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
            j = j + 1
        if dominated is False:
            out_idx.append(i)
        i = i + 1
    return out_idx


def save_pareto_plot(records_list, pareto_x_key, pareto_y_key, output_dir):
    """
    Args:
        records_list: list of dict
            Trial records containing final_metrics maps.
        pareto_x_key: str
            X axis scalar key inside final_metrics.
        pareto_y_key: str
            Y axis scalar key inside final_metrics.
        output_dir: Path
            Directory for pareto.png .
    return: Path
            Path to pareto.png .
    """
    xs = []
    ys = []
    ri = 0
    while ri < len(records_list):
        rec_item = records_list[ri]
        fm = rec_item["final_metrics"]
        xa = fm.get(pareto_x_key, float("nan"))
        ya = fm.get(pareto_y_key, float("nan"))
        xs.append(float(xa))
        ys.append(float(ya))
        ri = ri + 1
    if len(xs) == 0:
        fig_empty, axe = plt.subplots(figsize=(7, 5))
        axe.text(0.5, 0.5, "no trials to plot", ha="center")
        outp_empty = Path(output_dir) / "pareto.png"
        outp_empty.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp_empty, dpi=120)
        plt.close(fig_empty)
        return outp_empty
    pairs_xy = []
    pi = 0
    while pi < len(xs):
        pairs_xy.append((xs[pi], ys[pi]))
        pi = pi + 1
    nd_local = pareto_non_dominated_minimize(pairs_xy)
    fig, axi = plt.subplots(figsize=(7, 5))
    axi.scatter(xs, ys, c="#888888", s=42, label="all trials", alpha=0.85)
    nd_x = []
    nd_y = []
    ki = 0
    while ki < len(nd_local):
        pj = nd_local[ki]
        nd_x.append(xs[pj])
        nd_y.append(ys[pj])
        ki = ki + 1
    axi.scatter(nd_x, nd_y, c="#c0392b", s=76, marker="*", label="Pareto frontier", alpha=0.95, zorder=4)
    axi.set_xlabel(pareto_x_key)
    axi.set_ylabel(pareto_y_key)
    axi.set_title("Pareto set (multi-objective, lower-left better)")
    axi.legend(loc="best")
    axi.grid(alpha=0.25)
    out_path_png = Path(output_dir) / "pareto.png"
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=170)
    plt.close(fig)
    return out_path_png


def wandb_summary_log_plot(study_name_str, pareto_path, wandb_project, wandb_offline_flag):
    """
    Logs pareto.png to W&B inside a trailing summary run.

    Args:
        study_name_str: str
            Study basename for grouping.
        pareto_path: Path
            pareto.png filesystem path.
        wandb_project: str
            W&B project matching training runs.
        wandb_offline_flag: bool
            Whether offline mode is preferred.
        return: None
    """
    import wandb as wb_sdk

    run_name_here = "%s__summary" % study_name_str
    wb_sdk.init(project=wandb_project, name=run_name_here, group=study_name_str, reinit=True,
                offline=wandb_offline_flag)
    wb_sdk.log({"pareto_plot": wb_sdk.Image(str(pareto_path))})
    wb_sdk.finish()


@hydra.main(version_base=None, config_path="../../conf/hyperparam", config_name="sweeps")
def main(cfg):
    """
    Args:
        cfg: DictConfig
            Hyperparam driver yaml (study, sweeps, pareto keys).

    return: None
    """
    OmegaConf.resolve(cfg)
    runner = SweepRunner(cfg)
    runner.run_all_sweeps()

    outp = Path(str(OmegaConf.select(cfg, "output_dir")))
    outp.mkdir(parents=True, exist_ok=True)
    dumped_path = outp / "all_trials.json"
    with open(dumped_path, "w", encoding="utf-8") as fh_out:
        json.dump(runner.trial_records, fh_out, indent=2, default=str)

    runner.report_global_best()

    png_path_here = save_pareto_plot(runner.trial_records, runner.pareto_x_key, runner.pareto_y_key, outp)
    print("Saved Pareto plot to %s" % png_path_here)

    off_raw = OmegaConf.select(cfg, "wandb_offline")
    if isinstance(off_raw, bool):
        off_lit = off_raw
    elif off_raw is None:
        off_lit = True
    elif str(off_raw).strip().lower() in ("false", "0"):
        off_lit = False
    else:
        off_lit = True
    study_nm = str(OmegaConf.select(cfg, "study_name"))
    wandb_proj = str(OmegaConf.select(cfg, "wandb_project"))
    wandb_summary_log_plot(study_nm, png_path_here, wandb_proj, off_lit)


if __name__ == "__main__":
    main()
