"""
summary_pareto.py

Downloads summary metrics from all W&B runs in a given project and generates two Pareto scatter plots
(one for the final_test split, one for final_validation).

Pareto plots: x-axis = stress_abs_mae, y-axis = rollout_pos_mae + rollout_volume_mae.
Runs on the Pareto frontier (minimising both axes) are shown with distinct colours and markers
and appear in the legend. Dominated runs are shown in grey with no legend entry.
The JSON file is saved alongside this script; plots are also saved in the same directory.

Run:
    python -m src.gnn.scripts.summary_pareto --entity <wandb_entity> --project <wandb_project>

Example:
    python -m src.gnn.scripts.summary_pareto --project beam_deformation

To avoid fetching:
    python -m src.gnn.scripts.summary_pareto --project beam_deformation --from-json
"""

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb

matplotlib.use("Agg")
plt.ioff()

_METRIC_KEYS = [
    "final_validation/rollout_pos_mae",
    "final_test/rollout_pos_mae",
    "final_test/rollout_volume_mae",
    "final_validation/rollout_volume_mae",
    "final_validation/stress_abs_mae",
    "final_test/stress_abs_mae",
]

_PARETO_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]
_GREY = "#adb5bd"

# Map W&B run display name → custom legend label for Pareto-frontier points.
# Add entries here to rename runs in the plot legend without touching W&B.
# Runs not listed keep their original W&B name.
_RUN_NAME_OVERRIDES: dict = {
    "hp_opt_transolver_2026-05-19_22-39-30__transolver_arch__transolver_big_S128": "Transolver_7L_HD312_AH15_S128_R4",
    "hp_opt_transolver_2026-05-19_22-39-30__transolver_arch__transolver_big_S64": "Transolver_7L_HD312_AH15_S64_R4",
    "hp_opt_transolver_2026-05-19_22-39-30__transolver_arch__transolver_mid_S16": "Transolver_5L_HD256_AH30_S16_R1",
    "hp_opt_connectivity_2026-05-31_12-37-08__mgn__random20": "MGN_random20",
    "hp_opt_transolver_heads_2026-05-28_16-28-53__transolver_head_sweep__fullstress_vonmisesloss_w0.5":"Transolver_7L_HD312_AH15_S64_R4_fullstress"
}


def fetch_all_run_metrics(entity: str, project: str) -> dict:
    """
    Downloads the 6 summary metrics for every run in the given W&B project.
    Metrics absent from a run's summary are stored as None.

    Args:
        entity: str
            W&B username or team name.
        project: str
            W&B project name.
    return: dict
        Keyed by run.id; each value is {"name": run.name, <metric_key>: float|None, ...}.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    result = {}
    for run in runs:
        if run.name.startswith("hp_opt_dataset"):
            continue
        entry: dict = {"name": run.name}
        for key in _METRIC_KEYS:
            raw = run.summary.get(key)
            if raw is None:
                entry[key] = None
            else:
                try:
                    val = float(raw)
                    entry[key] = None if math.isnan(val) or math.isinf(val) else val
                except (TypeError, ValueError):
                    entry[key] = None
        result[run.id] = entry
    return result


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    """
    Saves the metrics dictionary to a JSON file.

    Args:
        metrics: dict
            Output of fetch_all_run_metrics.
        output_path: Path
            Destination file path.
    """
    with open(output_path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def compute_pareto_frontier_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes a boolean mask identifying non-dominated points when minimising both x and y.
    Point i is dominated if there exists a point j such that x[j] <= x[i] and y[j] <= y[i]
    with at least one strict inequality.

    Args:
        x: np.ndarray
            Shape (n,). First objective values (minimise).
        y: np.ndarray
            Shape (n,). Second objective values (minimise).
    return: np.ndarray
        Boolean mask of shape (n,); True = on the Pareto frontier.
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        dom_x = x <= x[i]
        dom_y = y <= y[i]
        strictly_better = (x < x[i]) | (y < y[i])
        dominates_i = dom_x & dom_y & strictly_better
        dominates_i[i] = False
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto


def _build_split_data(run_metrics: dict, split: str) -> tuple:
    """
    Extracts (names, x_arr, y_arr) for runs that have complete finite data for the given split.
    x = stress_abs_mae, y = rollout_pos_mae + rollout_volume_mae, both clipped to [0, 0.5] / [0, 0.1].

    Args:
        run_metrics: dict
            Output of fetch_all_run_metrics.
        split: str
            Either "final_test" or "final_validation".
    return: tuple[list[str], np.ndarray, np.ndarray]
        Parallel lists/arrays: run display names, x values, y values.
    """
    stress_key = f"{split}/stress_abs_mae"
    pos_key = f"{split}/rollout_pos_mae"
    vol_key = f"{split}/rollout_volume_mae"
    names, x_vals, y_vals = [], [], []
    for run_data in run_metrics.values():
        stress = run_data.get(stress_key)
        pos = run_data.get(pos_key)
        vol = run_data.get(vol_key)
        if stress is None or pos is None or vol is None:
            continue
        names.append(run_data["name"])
        x_vals.append(stress)
        y_vals.append(pos + vol)
    x_arr = np.clip(np.array(x_vals, dtype=float), 0.0, 0.5)
    y_arr = np.clip(np.array(y_vals, dtype=float), 0.0, 0.1)
    return names, x_arr, y_arr


def _assign_colors_and_markers(frontier_names_a: list, frontier_names_b: list) -> dict:
    """
    Assigns a consistent (color, marker) pair to every run name that appears on either frontier.
    The same run name always gets the same color and marker in both plots.

    Args:
        frontier_names_a: list[str]
            Run names on the first split's Pareto frontier.
        frontier_names_b: list[str]
            Run names on the second split's Pareto frontier.
    return: dict[str, tuple]
        Maps run name → (matplotlib color, marker string).
    """
    seen = {}
    rank = 0
    cmap = plt.cm.tab10
    for name in frontier_names_a + frontier_names_b:
        if name not in seen:
            seen[name] = (cmap(rank % 10), _PARETO_MARKERS[rank % len(_PARETO_MARKERS)])
            rank += 1
    return seen


def plot_pareto(run_metrics: dict, split: str, cross_split: str,
                cross_pareto_names: set, color_marker_map: dict, output_path: Path,
                show_legend: bool, figsize: tuple) -> None:
    """
    Generates a Pareto scatter plot for one evaluation split, also showing where the other
    split's Pareto-frontier runs land (hollow markers).

    Rendering layers (back to front):
      1. Grey filled: runs not on either split's Pareto frontier.
      2. Filled colored: all runs on the union of both frontiers, plotted at this split's
         coordinates. Same color+marker per run in both plots so one legend covers both.

    Args:
        run_metrics: dict
            Output of fetch_all_run_metrics.
        split: str
            Either "final_test" or "final_validation" — the primary split for this plot.
        cross_split: str
            The other split ("final_validation" or "final_test").
        cross_pareto_names: set[str]
            Run names on the OTHER split's Pareto frontier.
        color_marker_map: dict[str, tuple]
            Maps run name → (color, marker) for consistent styling across both plots.
        output_path: Path
            Destination PNG path.
        show_legend: bool
            When True, renders the legend outside the right edge of the plot.
    """
    names, x_arr, y_arr = _build_split_data(run_metrics, split)
    if len(names) == 0:
        print(f"  No complete data for split={split}, skipping plot.")
        return

    pareto_mask = compute_pareto_frontier_mask(x_arr, y_arr)
    # Union of this split's frontier and the cross-split frontier
    any_frontier_mask = pareto_mask | np.array([names[i] in cross_pareto_names for i in range(len(names))], dtype=bool)
    name_to_idx = {n: i for i, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Grey: not on any frontier
    neither_mask = ~any_frontier_mask
    if np.any(neither_mask):
        ax.scatter(x_arr[neither_mask], y_arr[neither_mask], color=_GREY, s=60, zorder=1)

    # 2. Colored filled: all frontier runs (own + cross), same symbol as in the other plot
    # Iterate in color_marker_map order so the legend is ordered consistently in both plots
    for name, (color, marker) in color_marker_map.items():
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        display = _RUN_NAME_OVERRIDES.get(name, name)
        label = textwrap.fill(display, width=30)
        ax.scatter(x_arr[idx], y_arr[idx],
                   color=color, marker=marker, s=80, zorder=2, label=label)

    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.1)
    ax.set_xlabel("Stress Absolute MAE", fontsize=11)
    ax.set_ylabel("Rollout Pos MAE + Rollout Volume MAE", fontsize=11)
    ax.set_title(f"Pareto Frontier — {split}", fontsize=12)
    ax.grid(True, alpha=0.3)
    has_legend_entries = len(color_marker_map) > 0
    if show_legend and has_legend_entries:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Pareto plot: {output_path}")


def main() -> None:
    """
    Entry point: parses CLI args, downloads metrics, saves JSON, generates Pareto plots.
    """
    parser = argparse.ArgumentParser(description="Download W&B metrics and generate Pareto plots.")
    parser.add_argument("--entity", default="jacoposenoner", help="W&B username or team name.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--from-json", action="store_true",
                        help="Skip W&B download and read metrics from the existing JSON file.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    json_path = script_dir / f"{args.project}_metrics.json"

    if args.from_json:
        print(f"Loading metrics from {json_path} ...")
        with open(json_path) as fh:
            run_metrics = json.load(fh)
        print(f"  Loaded {len(run_metrics)} runs.")
    else:
        print(f"Fetching runs from {args.entity}/{args.project} ...")
        run_metrics = fetch_all_run_metrics(args.entity, args.project)
        print(f"  Found {len(run_metrics)} runs.")
        save_metrics_json(run_metrics, json_path)
        print(f"  Saved metrics JSON: {json_path}")

    # Compute both Pareto frontiers up front so colors/markers are consistent across both plots
    names_test, x_test, y_test = _build_split_data(run_metrics, "final_test")
    names_val, x_val, y_val = _build_split_data(run_metrics, "final_validation")
    pareto_names_test = [names_test[i] for i in np.where(compute_pareto_frontier_mask(x_test, y_test))[0]]
    pareto_names_val = [names_val[i] for i in np.where(compute_pareto_frontier_mask(x_val, y_val))[0]]
    color_marker_map = _assign_colors_and_markers(pareto_names_test, pareto_names_val)

    print("Generating Pareto plot for final_test ...")
    plot_pareto(run_metrics, "final_test", "final_validation",
                set(pareto_names_val), color_marker_map,
                script_dir / "pareto_final_test.png", show_legend=True, figsize=(8, 4))

    print("Generating Pareto plot for final_validation ...")
    plot_pareto(run_metrics, "final_validation", "final_test",
                set(pareto_names_test), color_marker_map,
                script_dir / "pareto_final_validation.png", show_legend=False, figsize=(6, 4))


if __name__ == "__main__":
    main()
