"""
plot_overall.py
Contains functions for generating summary plots of model evaluation, with main access point overall_plot().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_FIG_WIDTH_CAP = 16.0
_FIG_WIDTH_FLOOR = 7.0
_FIG_HEIGHT = 5.0
_BAR_WIDTH = 1.0
_COLOR_SELECTED = "crimson"
_COLOR_STRIPE_A = "gold"
_COLOR_STRIPE_B = "royalblue"


def overall_plot(all_final_roll_err_pos, per_traj_acc_rel_mae, per_traj_stress_rel_mae,
                 final_volumes_true, final_volumes_pred, output_dir, split,
                 selected_trajectory_indices):
    """
    Generates summary bar-chart plots that aggregate metrics across all trajectories.
    Main access point called by trainer.

    Args:
        all_final_roll_err_pos: list[float]
            Per-trajectory final rollout position error.
        per_traj_acc_rel_mae: list[float]
            Per-trajectory acceleration relative MAE (no rollout).
        per_traj_stress_rel_mae: list[float]
            Per-trajectory stress relative MAE (no rollout).
        final_volumes_true: list[float]
            Per-trajectory ground-truth final volume.
        final_volumes_pred: list[float]
            Per-trajectory predicted final volume.
        output_dir: Path
            Directory where summary plots are saved.
        split: str
            Split name for plot titles (e.g. test, validation).
        selected_trajectory_indices: list[int]
            Trajectory indices highlighted in red (same as in-depth rollout plots).
    return: dict
        Nested dict with keys "images" and "videos", each mapping a plot name to its Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {"images": {}, "videos": {}}

    # 1) Final rollout position error per trajectory
    p1 = output_dir / "final_roll_err_pos_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(all_final_roll_err_pos, dtype=float),
        ylabel="Final Rollout Position MAE",
        title=f"Final Rollout Position Error per Trajectory [{split}]",
        output_path=p1,
        x_sort_caption="final rollout position MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["final_roll_err_pos_per_traj"] = p1

    # 2) Acceleration relative MAE per trajectory (no rollout)
    # FIXME: make sure this is also the one with last trajectory rollout
    p2 = output_dir / "acc_rel_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_acc_rel_mae, dtype=float),
        ylabel="Acceleration Relative MAE",
        title=f"Acceleration Relative MAE per Trajectory (no rollout) [{split}]",
        output_path=p2,
        x_sort_caption="acceleration relative MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["acc_rel_mae_noroll_per_traj"] = p2

    # 3) Stress relative MAE per trajectory (no rollout)
    # FIXME: make sure this is also the one with last trajectory rollout
    p3 = output_dir / "stress_rel_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_stress_rel_mae, dtype=float),
        ylabel="Stress Relative MAE",
        title=f"Stress Relative MAE per Trajectory (no rollout) [{split}]",
        output_path=p3,
        x_sort_caption="stress relative MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["stress_rel_mae_noroll_per_traj"] = p3

    # 4) Absolute final volume error |pred - true| per trajectory
    vol_true = np.asarray(final_volumes_true, dtype=float)
    vol_pred = np.asarray(final_volumes_pred, dtype=float)
    volume_abs_err = np.abs(vol_pred - vol_true)
    p4 = output_dir / "final_volume_per_traj.png"
    _plot_sorted_per_traj_histogram(
        volume_abs_err,
        ylabel="Absolute volume error",
        title=f"Final Rollout Absolute Volume Error per Trajectory [{split}]",
        output_path=p4,
        x_sort_caption="absolute volume error (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["final_volume_per_traj"] = p4

    return paths


def _plot_sorted_per_traj_histogram(values, ylabel, title, output_path, x_sort_caption,
                                    selected_trajectory_indices):
    """
    Gapless thin bars per trajectory, sorted by value ascending; selected trajectories in red,
    others alternating gold/royalblue by sorted rank. Annotates median and 90th percentile.

    Args:
        values: np.ndarray
            One metric per trajectory, original order traj 0 .. n-1.
        ylabel: str
            Y-axis label.
        title: str
            Plot title.
        output_path: Path
            Destination image path.
        x_sort_caption: str
            Short phrase for the x-axis (how trajectories are ordered).
        selected_trajectory_indices: list[int]
            Trajectory indices drawn in red.
    return: None
        Saves figure to output_path.
    """
    n = int(values.shape[0])
    if n == 0:
        return
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    selected_set = set(selected_trajectory_indices)
    colors = []
    for rank, traj_idx in enumerate(order):
        if int(traj_idx) in selected_set:
            colors.append(_COLOR_SELECTED)
        elif rank % 2 == 0:
            colors.append(_COLOR_STRIPE_A)
        else:
            colors.append(_COLOR_STRIPE_B)

    span = max(1, n)
    fig_width = min(_FIG_WIDTH_CAP, max(_FIG_WIDTH_FLOOR, 0.13 * span))
    fig, ax = plt.subplots(figsize=(fig_width, _FIG_HEIGHT))
    x_left = np.arange(n, dtype=float)
    ax.bar(
        x_left,
        sorted_values,
        width=_BAR_WIDTH,
        align="edge",
        color=colors,
        linewidth=0,
        edgecolor="none",
    )
    ax.set_xlim(0, n)
    ax.set_xticks([])
    ax.set_xlabel(f"Trajectory index (sorted by {x_sort_caption})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    med = float(np.median(values))
    p90 = float(np.percentile(values, 90))
    ax.text(
        0.02,
        0.98,
        f"median = {med:.6g}\n90th percentile = {p90:.6g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    legend_handles = [
        Patch(facecolor=_COLOR_SELECTED, edgecolor="none", label="Selected trajectory"),
        Patch(facecolor=_COLOR_STRIPE_A, edgecolor="none", label="Other (stripe A)"),
        Patch(facecolor=_COLOR_STRIPE_B, edgecolor="none", label="Other (stripe B)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
