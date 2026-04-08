"""
plot_overall.py
Contains functions for generating summary plots of model evaluation, with main access point overall_plot().
"""

from pathlib import Path  # Build output filenames and create directories for saved PNGs.

import matplotlib.pyplot as plt  # Create figures, axes, bar charts, legends, savefig, close.
import numpy as np  # Array conversion, sorting indices, median/percentile stats for annotations.
from matplotlib.patches import Patch  # Legend swatches with custom face colors (no bar handles needed).

_FIG_WIDTH_CAP = 16.0  # Upper cap on figure width (inches) so very many trajectories do not explode layout.
_FIG_WIDTH_FLOOR = 7.0  # Minimum figure width (inches) so few trajectories still read comfortably.
_FIG_HEIGHT = 5.0  # Fixed figure height (inches) for all summary bar charts.
_BAR_WIDTH = 1.0  # Matplotlib bar width in data units; with align="edge" bars tile x = 0..n without gaps.
_COLOR_SELECTED = "crimson"  # Bar fill for trajectories the user marked as “selected” in rollout plots.
_COLOR_STRIPE_A = "gold"  # Alternating fill for non-selected bars at even sorted ranks (visual striping).
_COLOR_STRIPE_B = "royalblue"  # Alternating fill for non-selected bars at odd sorted ranks.


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
    output_dir = Path(output_dir)  # Normalize to pathlib so joins/mkdir are consistent.
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure PNG parent folders exist before savefig.

    paths = {"images": {}, "videos": {}}  # Collect paths; videos key unused here but matches trainer contract.

    # 1) Final rollout position error per trajectory
    p1 = output_dir / "final_roll_err_pos_per_traj.png"  # Output file for sorted bar chart of rollout pos error.
    _plot_sorted_per_traj_histogram(
        np.asarray(all_final_roll_err_pos, dtype=float),  # One scalar per trajectory as float array for sort/plot.
        ylabel="Final Rollout Position MAE",  # Y-axis text: values are mean absolute position error.
        title=f"Final Rollout Position Error per Trajectory [{split}]",  # Title names metric and data split.
        output_path=p1,  # Where matplotlib writes the PNG.
        x_sort_caption="final rollout position MAE (ascending)",  # Clarifies x order in xlabel (not raw traj ids).
        selected_trajectory_indices=selected_trajectory_indices,  # Which original traj indices paint crimson.
    )
    paths["images"]["final_roll_err_pos_per_traj"] = p1  # Register path for logging/UI.

    # 2) Acceleration relative MAE per trajectory (no rollout)
    # FIXME: make sure this is also the one with last trajectory rollout
    p2 = output_dir / "acc_rel_mae_noroll_per_traj.png"  # PNG for per-traj acceleration relative MAE bars.
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_acc_rel_mae, dtype=float),  # Per-trajectory acc metric as plottable array.
        ylabel="Acceleration Relative MAE",  # Y label: relative MAE of acceleration components.
        title=f"Acceleration Relative MAE per Trajectory (no rollout) [{split}]",  # Title states no rollout.
        output_path=p2,  # Save destination.
        x_sort_caption="acceleration relative MAE (ascending)",  # X order = ascending error.
        selected_trajectory_indices=selected_trajectory_indices,  # Highlight chosen trajectories in red.
    )
    paths["images"]["acc_rel_mae_noroll_per_traj"] = p2  # Record path.

    # 3) Stress relative MAE per trajectory (no rollout)
    # FIXME: make sure this is also the one with last trajectory rollout
    p3 = output_dir / "stress_rel_mae_noroll_per_traj.png"  # PNG for stress relative MAE bars.
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_stress_rel_mae, dtype=float),  # Stress metric per trajectory.
        ylabel="Stress Relative MAE",  # Y-axis label for von Mises (or stress) relative error scale.
        title=f"Stress Relative MAE per Trajectory (no rollout) [{split}]",  # Title + split tag.
        output_path=p3,  # File path for figure.
        x_sort_caption="stress relative MAE (ascending)",  # Sort key explained under x-axis.
        selected_trajectory_indices=selected_trajectory_indices,  # Crimson bars for selected traj indices.
    )
    paths["images"]["stress_rel_mae_noroll_per_traj"] = p3  # Store path.

    # 4) Absolute final volume error |pred - true| per trajectory
    vol_true = np.asarray(final_volumes_true, dtype=float)  # Ground-truth final volumes as float array.
    vol_pred = np.asarray(final_volumes_pred, dtype=float)  # Predicted final volumes as float array.
    volume_abs_err = np.abs(vol_pred - vol_true)  # Per-trajectory scalar to plot: absolute volume mistake.
    p4 = output_dir / "final_volume_per_traj.png"  # PNG for volume error bar chart.
    _plot_sorted_per_traj_histogram(
        volume_abs_err,  # Bar heights = |V_pred - V_true| per trajectory (then sorted for display).
        ylabel="Absolute volume error",  # Y-axis: linear error in volume units.
        title=f"Final Rollout Absolute Volume Error per Trajectory [{split}]",  # Title describes metric + split.
        output_path=p4,  # Output PNG path.
        x_sort_caption="absolute volume error (ascending)",  # Bars ordered by increasing volume error.
        selected_trajectory_indices=selected_trajectory_indices,  # Same red highlight policy as other plots.
    )
    paths["images"]["final_volume_per_traj"] = p4  # Register fourth summary figure.

    return paths  # Return map of artifact names to paths (images only populated here).


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
    n = int(values.shape[0])  # Number of trajectories = number of bars to draw.
    if n == 0:
        return  # Nothing to plot; avoid empty figure.
    order = np.argsort(values, kind="stable")  # Permutation that sorts bar heights ascending (stable ties).
    sorted_values = values[order]  # Bar heights left-to-right after sorting.
    selected_set = set(selected_trajectory_indices)  # Fast membership test for red coloring by original index.
    colors = []  # One bar color per sorted position, built in loop below.
    for rank, traj_idx in enumerate(order):  # rank = x position; traj_idx = original trajectory id.
        if int(traj_idx) in selected_set:
            colors.append(_COLOR_SELECTED)  # Plot this bar crimson if traj is in selected list.
        elif rank % 2 == 0:
            colors.append(_COLOR_STRIPE_A)  # Even rank after sort: gold stripe.
        else:
            colors.append(_COLOR_STRIPE_B)  # Odd rank after sort: blue stripe.

    span = max(1, n)  # Avoid zero in width formula when n==0 (already returned); else use n.
    fig_width = min(_FIG_WIDTH_CAP, max(_FIG_WIDTH_FLOOR, 0.13 * span))  # Wider figure when more bars (~0.13"/bar).
    fig, ax = plt.subplots(figsize=(fig_width, _FIG_HEIGHT))  # Single axes for vertical bar chart.
    x_left = np.arange(n, dtype=float)  # Left edges of bars at 0,1,...,n-1 (align="edge" draws rightward).
    ax.bar(
        x_left,  # X positions of bar left edges in data coordinates.
        sorted_values,  # Bar heights = sorted metric values.
        width=_BAR_WIDTH,  # Each bar spans one unit in x so bars touch without overlap.
        align="edge",  # x_left is the left edge of each rectangle (not center).
        color=colors,  # Per-bar face color list (matplotlib cycles one color per bar).
        linewidth=0,  # No border stroke on bars.
        edgecolor="none",  # Explicitly disable bar outlines for flat fill look.
    )
    ax.set_xlim(0, n)  # Clip x view to bar region [0, n].
    ax.set_xticks([])  # Hide per-bar tick labels (too many trajectories); caption explains order instead.
    ax.set_xlabel(f"Trajectory index (sorted by {x_sort_caption})")  # Explains x is sort order not raw index.
    ax.set_ylabel(ylabel)  # Metric name and units on vertical axis.
    ax.set_title(title)  # Figure headline above axes.
    ax.grid(True, alpha=0.3, axis="y")  # Light horizontal gridlines to read bar heights.
    med = float(np.median(values))  # Median over unsorted per-traj values (population summary).
    p90 = float(np.percentile(values, 90))  # 90th percentile for annotation box.
    ax.text(
        0.02,  # X in axes fraction: near left margin inside axes.
        0.98,  # Y in axes fraction: near top inside axes.
        f"median = {med:.6g}\n90th percentile = {p90:.6g}",  # Text box content with compact numeric formatting.
        transform=ax.transAxes,  # Interpret 0..1 coords relative to axes box (not data coords).
        va="top",  # Anchor text box top to the given point (text grows downward).
        ha="left",  # Anchor text’s left edge to the point.
        fontsize=9,  # Small but readable overlay text.
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9),  # White rounded label background.
    )
    legend_handles = [
        Patch(facecolor=_COLOR_SELECTED, edgecolor="none", label="Selected trajectory"),  # Legend patch for red bars.
        Patch(facecolor=_COLOR_STRIPE_A, edgecolor="none", label="Other (stripe A)"),  # Legend for gold bars.
        Patch(facecolor=_COLOR_STRIPE_B, edgecolor="none", label="Other (stripe B)"),  # Legend for blue bars.
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)  # Fixed legend explaining color semantics.
    fig.tight_layout()  # Reduce subplot padding so title/labels fit before rasterizing.
    fig.savefig(output_path, dpi=150)  # Write PNG at 150 dpi to disk.
    plt.close(fig)  # Free figure memory after save.
