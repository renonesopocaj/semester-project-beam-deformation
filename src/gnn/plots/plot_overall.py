"""
plot_overall.py
Contains functions for generating summary plots of model evaluation, with main access point overall_plot().
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Patch

_FIG_WIDTH_CAP = 16.0
_FIG_WIDTH_FLOOR = 7.0
_FIG_HEIGHT = 5.0
_BAR_WIDTH = 1.0
_COLOR_SELECTED = "crimson"
_COLOR_STRIPE_A = "gold"
_COLOR_STRIPE_B = "royalblue"

# Ordered list of (wandb_key, brief_label, group) for the combined train/run_summary histogram.
# Groups are final_validation (green), final_test (blue), train (gray).
# The two "train" entries (monitor + final loss) are injected at call time with their actual
# resolved key; the config below covers only the final_* bars.
_COMBINED_FINAL_METRICS_CONFIG = [
    ("final_validation/rollout_pos_mae",   "roll pos",    "final_validation"),
    ("final_validation/rollout_volume_mae","roll vol",    "final_validation"),
    ("final_validation/stress_abs_mae",    "stress abs",  "final_validation"),
    ("final_validation/stress_rel_mae",    "stress rel",  "final_validation"),
    ("final_test/rollout_pos_mae",         "roll pos",    "final_test"),
    ("final_test/rollout_volume_mae",      "roll vol",    "final_test"),
    ("final_test/stress_abs_mae",          "stress abs",  "final_test"),
    ("final_test/stress_rel_mae",          "stress rel",  "final_test"),
]

_COMBINED_GROUP_PALETTES = {
    "final_validation": ["#1b4332", "#2d6a4f", "#40916c", "#74c69d"],  # dark-to-light green
    "final_test":       ["#03045e", "#0077b6", "#00b4d8", "#48cae4"],  # dark-to-light blue
    "train":            ["#6c757d", "#adb5bd"],  # two gray shades: monitor then final loss
}

_COMBINED_GROUP_DISPLAY_NAMES = {
    "final_validation": "final_validation",
    "final_test":       "final_test",
    "train":            "train (monitor / final loss)",
}


def overall_plot(all_rollout_pos_mae, per_traj_acc_rel_mae, per_traj_stress_rel_mae,
                 per_traj_acc_abs_mae, per_traj_stress_abs_mae, per_traj_overall_abs_mae,
                 all_rollout_volume_mae, output_dir, split, selected_trajectory_indices):
    """
    Generates summary bar-chart plots that plot metric vs number of trajectory, for following metrics:
    - all_rollout_pos_mae
    - per_traj_acc_rel_mae
    - per_traj_stress_rel_mae
    - per_traj_acc_abs_mae
    - per_traj_stress_abs_mae
    - per_traj_overall_abs_mae
    - all_rollout_volume_mae
    Main access point called by trainer.

    Args:
        all_rollout_pos_mae: list[float]
            Per-trajectory rollout position MAE averaged over all timesteps, nodes, and coordinates.
        per_traj_acc_rel_mae: list[float]
            Per-trajectory acceleration relative MAE (no rollout).
        per_traj_stress_rel_mae: list[float]
            Per-trajectory stress relative MAE (no rollout).
        per_traj_acc_abs_mae: list[float]
            Per-trajectory acceleration absolute MAE (no rollout).
        per_traj_stress_abs_mae: list[float]
            Per-trajectory stress absolute MAE (no rollout).
        per_traj_overall_abs_mae: list[float]
            Per-trajectory overall absolute MAE (no rollout).
        all_rollout_volume_mae: list[float]
            Per-trajectory rollout volume MAE averaged over all timesteps.
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

    # 1) Rollout position error per trajectory
    p1 = output_dir / "rollout_pos_mae_per_traj.png"  # Output file for sorted bar chart of rollout pos error.
    _plot_sorted_per_traj_histogram(
        np.asarray(all_rollout_pos_mae, dtype=float),
        ylabel="Rollout Position MAE",
        title=f"Rollout Position MAE per Trajectory averaged over all timesteps and nodes [{split}]",
        output_path=p1,
        x_sort_caption="rollout position MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["rollout_pos_mae_per_traj"] = p1  # Register path for logging/UI.

    # 2) Acceleration relative MAE per trajectory (no rollout)
    p2 = output_dir / "acc_rel_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_acc_rel_mae, dtype=float),
        ylabel="Acceleration Relative MAE",
        title=f"Acceleration Relative MAE per Trajectory (no rollout) averaged over all nodes [{split}]",
        output_path=p2,
        x_sort_caption="acceleration relative MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["acc_rel_mae_noroll_per_traj"] = p2  # Record path.

    # 3) Stress relative MAE per trajectory (no rollout)
    p3 = output_dir / "stress_rel_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_stress_rel_mae, dtype=float),
        ylabel="Stress Relative MAE",
        title=f"Stress Relative MAE per Trajectory (no rollout) averaged over all nodes [{split}]",
        output_path=p3,
        x_sort_caption="stress relative MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["stress_rel_mae_noroll_per_traj"] = p3  # Store path.

    # 4) Acceleration absolute MAE per trajectory (no rollout)
    p4 = output_dir / "acc_all_abs_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_acc_abs_mae, dtype=float),
        ylabel="Acceleration Absolute MAE",
        title=f"Acceleration Absolute MAE per Trajectory (no rollout) averaged over all nodes [{split}]",
        output_path=p4,
        x_sort_caption="acceleration absolute MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["acc_all_abs_mae_noroll_per_traj"] = p4

    # 5) Stress absolute MAE per trajectory (no rollout)
    p5 = output_dir / "stress_abs_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_stress_abs_mae, dtype=float),
        ylabel="Stress Absolute MAE",
        title=f"Stress Absolute MAE per Trajectory (no rollout) averaged over all nodes [{split}]",
        output_path=p5,
        x_sort_caption="stress absolute MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["stress_abs_mae_noroll_per_traj"] = p5

    # 6) Overall absolute MAE per trajectory (no rollout)
    p6 = output_dir / "overall_abs_mae_noroll_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(per_traj_overall_abs_mae, dtype=float),
        ylabel="Overall Absolute MAE",
        title=f"Overall Absolute MAE per Trajectory (no rollout) averaged over all nodes [{split}]",
        output_path=p6,
        x_sort_caption="overall absolute MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["overall_abs_mae_noroll_per_traj"] = p6

    # 7) Absolute rollout volume MAE per trajectory
    p7 = output_dir / "rollout_volume_mae_per_traj.png"
    _plot_sorted_per_traj_histogram(
        np.asarray(all_rollout_volume_mae, dtype=float),
        ylabel="Rollout Volume MAE",
        title=f"Rollout Volume MAE per Trajectory averaged over all timesteps [{split}]",
        output_path=p7,
        x_sort_caption="rollout volume MAE (ascending)",
        selected_trajectory_indices=selected_trajectory_indices,
    )
    paths["images"]["rollout_volume_mae_per_traj"] = p7

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
    y_bottom, y_top = ax.get_ylim()  # Current vertical range from bar heights (before label margin).
    y_span = y_top - y_bottom  # Use span to add a small strip under bars for trajectory index text.
    margin_frac = 0.12  # Fraction of y-span reserved below y_bottom for rotated labels (no bar overlap).
    ax.set_ylim(y_bottom - margin_frac * y_span, y_top)  # Extend downward so labels sit between axis and bars.
    label_y = y_bottom - 0.02 * y_span  # Slightly below original baseline; anchors rotated digits in the margin.
    # Font shrinks when many bars so adjacent columns’ trajectory numbers do not overlap horizontally.
    traj_label_font = max(3, min(6, int(180 / max(n, 1))))
    for rank in range(n):
        traj_idx = int(order[rank])  # Original trajectory index for this sorted column.
        ax.text(
            rank + 0.5 * _BAR_WIDTH,  # Horizontal center of bar (edge-aligned width-_BAR_WIDTH bar).
            label_y,
            str(traj_idx),
            rotation=90,
            va="top",
            ha="center",
            fontsize=traj_label_font,
            color="0.35",
            clip_on=False,
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


def plot_combined_train_run_summary(accumulated_final_scalars, last_monitor_key,
                                    last_monitor_value, last_train_loss, output_path):
    """
    Single combined histogram logged to W&B under train/run_summary.
    Bars left-to-right:
      1. final_validation/* (green shades) — rollout pos/vol, stress abs/rel
      2. final_test/*       (blue shades)  — same four metrics
      3. scheduler monitor bar (dark gray) — last training epoch value of val/ or train/loss
      4. train/loss bar        (light gray)— final training epoch loss

    Metrics absent from accumulated_final_scalars are silently skipped.
    The monitor bar is omitted when last_monitor_value is nan; the train/loss bar is omitted
    when last_train_loss is nan. When the monitor key equals "train/loss", the two train bars
    are labelled differently to distinguish them ("monitor" vs "final loss").

    Args:
        accumulated_final_scalars: dict
            Flat {wandb_metric_key: float} containing any final_validation/* and final_test/* values.
        last_monitor_key: str
            Resolved W&B key used by the scheduler monitor (e.g. "val/overall_relative_mae").
        last_monitor_value: float
            Last-epoch value of the scheduler monitor metric.
        last_train_loss: float
            Final training epoch train/loss value.
        output_path: Path
            Destination PNG path.
    """
    import math

    entries = [(key, brief, group) for key, brief, group in _COMBINED_FINAL_METRICS_CONFIG
               if key in accumulated_final_scalars]

    # Determine brief labels for the two train bars, avoiding duplicate labels when both resolve to train/loss
    monitor_is_train_loss = (last_monitor_key == "train/loss")
    monitor_brief = "monitor" if monitor_is_train_loss else last_monitor_key.split("/")[-1]
    train_loss_brief = "final loss" if monitor_is_train_loss else "loss"

    if not math.isnan(last_monitor_value):
        entries.append((last_monitor_key, monitor_brief, "train"))
    # Add train/loss bar only when it would not duplicate the monitor bar already added above
    if not math.isnan(last_train_loss) and not (monitor_is_train_loss and not math.isnan(last_monitor_value)):
        entries.append(("train/loss", train_loss_brief, "train"))

    if not entries:
        return

    # Resolve bar values — monitor and final-loss scalars come from dedicated args, not the dict
    def _value_for(key, brief, group):
        if group == "train" and key == last_monitor_key and brief == monitor_brief:
            return last_monitor_value
        if group == "train" and key == "train/loss":
            return last_train_loss
        return float(accumulated_final_scalars[key])

    positions = []
    brief_labels = []
    full_labels = []
    bar_colors = []
    bar_values = []
    group_shade_counter = {}
    seen_groups = []
    x = 0.0
    current_group = None
    bar_w = 0.75
    group_gap = 0.9

    for key, brief, group in entries:
        if group != current_group:
            if current_group is not None:
                x += group_gap
            current_group = group
            if group not in seen_groups:
                seen_groups.append(group)
        shade_idx = group_shade_counter.get(group, 0)
        palette = _COMBINED_GROUP_PALETTES[group]
        bar_colors.append(palette[shade_idx % len(palette)])
        group_shade_counter[group] = shade_idx + 1
        positions.append(x)
        brief_labels.append(brief)
        full_labels.append(key)
        bar_values.append(_value_for(key, brief, group))
        x += 1.0

    n = len(positions)
    fig_w = max(9.0, n * 0.95 + len(seen_groups) * group_gap)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    ax.bar(positions, bar_values, width=bar_w, color=bar_colors, linewidth=0, edgecolor="none")

    for pos, val in zip(positions, bar_values):
        ax.text(pos, val, f"{val:.4g}", ha="center", va="bottom", fontsize=8, color="0.2")

    ax.set_xticks(positions)
    ax.set_xticklabels(brief_labels, fontsize=9)

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for pos, full_name in zip(positions, full_labels):
        ax.text(pos, -0.04, full_name, transform=blend, rotation=90,
                va="top", ha="center", fontsize=6.5, color="0.4", clip_on=False)

    ax.set_title("Run Summary — final_validation / final_test / train")
    ax.set_ylabel("Metric Value")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.65, x - 0.35)

    legend_handles = [
        Patch(facecolor=_COMBINED_GROUP_PALETTES[g][0], edgecolor="none",
              label=_COMBINED_GROUP_DISPLAY_NAMES[g])
        for g in seen_groups
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)