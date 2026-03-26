"""
plot_overall.py
Contains functions for generating summary plots of model evaluation, with main access point overall_plot().
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def overall_plot(all_final_roll_err_pos, per_traj_acc_rel_mae, per_traj_stress_rel_mae,
                 final_volumes_true, final_volumes_pred, output_dir, split):
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
    return: dict
        Nested dict with keys "images" and "videos", each mapping a plot name to its Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_indices = list(range(len(all_final_roll_err_pos)))

    paths = {"images": {}, "videos": {}}

    # 1) Final rollout position error per trajectory
    p1 = output_dir / "final_roll_err_pos_per_traj.png"
    _plot_per_trajectory_bar(traj_indices, all_final_roll_err_pos, ylabel="Final Rollout Position MAE",
                             title=f"Final Rollout Position Error per Trajectory [{split}]", output_path=p1,
                             color="tomato", true_values=None)
    paths["images"]["final_roll_err_pos_per_traj"] = p1

    # 2) Acceleration relative MAE per trajectory (no rollout)
    # FIXME: make sure this is also the one with last trajectory rollout
    p2 = output_dir / "acc_rel_mae_noroll_per_traj.png"
    _plot_per_trajectory_bar(traj_indices, per_traj_acc_rel_mae, ylabel="Acceleration Relative MAE",
                             title=f"Acceleration Relative MAE per Trajectory (no rollout) [{split}]", output_path=p2,
                             color="mediumseagreen", true_values=None)
    paths["images"]["acc_rel_mae_noroll_per_traj"] = p2

    # 3) Stress relative MAE per trajectory (no rollout)
    # FIXME: make sure this is also the one with last trajectory rollout
    p3 = output_dir / "stress_rel_mae_noroll_per_traj.png"
    _plot_per_trajectory_bar(traj_indices, per_traj_stress_rel_mae, ylabel="Stress Relative MAE",
                             title=f"Stress Relative MAE per Trajectory (no rollout) [{split}]", output_path=p3,
                             color="mediumpurple", true_values=None)
    paths["images"]["stress_rel_mae_noroll_per_traj"] = p3

    # 4) Final rollout volume per trajectory (predicted vs true)
    p4 = output_dir / "final_volume_per_traj.png"
    _plot_per_trajectory_bar(traj_indices, final_volumes_pred, ylabel="Final Volume",
                             title=f"Final Rollout Volume per Trajectory [{split}]", output_path=p4, color="darkorange",
                             true_values=final_volumes_true)
    paths["images"]["final_volume_per_traj"] = p4

    return paths

def _plot_per_trajectory_bar(traj_indices, values, ylabel, title, output_path, color, true_values):
    """
    Bar chart with one bar per trajectory.

    Args:
        traj_indices: list[int]
            Trajectory indices for the x-axis.
        values: list[float]
            Values for the y-axis (one per trajectory).
        ylabel: str
            Label for the y-axis.
        title: str
            Plot title.
        output_path: Path
            Destination image path.
        color: str
            Bar color.
        true_values: list[float] or None
            If provided, draws a second bar group for ground-truth reference.
    """
    x = np.arange(len(traj_indices))
    fig, ax = plt.subplots(figsize=(max(6, len(traj_indices) * 0.8), 5))
    if true_values is not None:
        bar_width = 0.35
        ax.bar(x - bar_width / 2, true_values, bar_width, label="True", color="steelblue", alpha=0.7)
        ax.bar(x + bar_width / 2, values, bar_width, label="Predicted", color=color, alpha=0.7)
        ax.legend()
    else:
        ax.bar(x, values, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in traj_indices], rotation=45 if len(traj_indices) > 15 else 0)
    ax.set_xlabel("Trajectory")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)