"""
plot_rollout.py
Contains all files that plot metrics or diagnostics for the rollout.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from matplotlib.ticker import FuncFormatter
from src.gnn.validation.diagnostic_plot_helpers import (compute_coordinate_bounds, extract_surface_faces,
                                                          update_surface_collection)
from src.gnn.mesh_numerical_helpers.compute_volume import volume
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import extract_tetra_connectivity, to_numpy
from src.gnn.misc_helpers.sklearn_metrics_helpers import (
    format_r2_text,
    r2_linear_fit_x_to_y,
    r2_prediction_vs_target,
)
from pathlib import Path

RELATIVE_ERROR_CLIP = 5.0  # 500 %
TARGET_FIELDS = [("acc_x", "Acceleration X"),
                 ("acc_y", "Acceleration Y"),
                 ("acc_z", "Acceleration Z"),
                 ("von_mises", "von Mises")]

def plot_volume_change(pred_rollout, true_rollout, path, split_name):
    """
    Plot volume change over time for predicted vs ground truth.

    Args:
        pred_rollout: list[FIXME: TYPE]
            list of predictions (each element attributes = targets)
        true_rollout: list[FIXME: TYPE]
            list of ground truth values (each element has attributes = targets)
        path: Path
        split_name: str
            string with split name
    """
    fig, ax = plt.subplots()
    true_volumes_list, pred_volumes_list = [], []

    for pred, true in zip(pred_rollout, true_rollout):
        x_true = true.x_pos_t
        x_pred = pred.x_pos_t
        # Obtain all cells as tensors of (N, 4) i.e. 4 nodes for everyone of the N tetrahedrons
        cells = torch.as_tensor(extract_tetra_connectivity(true.x_element_connectivity), dtype=torch.long)
        true_volumes = volume(x_true, cells)
        pred_volumes = volume(x_pred, cells)
        # sum the absolute value of all volumes
        V_true = torch.abs(true_volumes).sum().detach().cpu().item()
        V_pred = torch.abs(pred_volumes).sum().detach().cpu().item()

        true_volumes_list.append(V_true)
        pred_volumes_list.append(V_pred)

    ax.plot(true_volumes_list, label="True Volumes")
    ax.plot(pred_volumes_list, label="Predicted Volumes")
    plt.xlabel("Time Step")
    plt.ylabel("Volume")
    ax.set_title(f"Volume Change\n[{split_name}, 1 sample, {len(pred_rollout)} timesteps]")
    ax.legend()
    plt.savefig(path)
    plt.close()

def make_beam_comparison_gif(pred_rollout, true_rollout, L, W, D, gif_path, fps, bbox_only_on_true, verbose, split_name):
    """
    Create side-by-side comparison GIF of predicted vs ground truth.

    Args:
        pred_rollout: list
            List of predicted graphs
        true_rollout: list
            List of ground truth graphs
        L: float
        W: float
        D: float
            Beam dimensions
        gif_path: Path
            Output GIF filename
        fps: int
            Frames per second
        bbox_only_on_true: bool
            Whether to base bounding box on ground truth only
        verbose: bool

    """
    assert len(pred_rollout) == len(true_rollout), ("Mismatch in predicted and ground truth frames.")

    frames = []

    faces = extract_surface_faces(true_rollout[0].x_element_connectivity)
    coordinates_for_bounds = []
    vm_min, vm_max = 0.0, 0.0

    for g_pred, g_true in zip(pred_rollout, true_rollout):
        coords_pred = to_numpy(g_pred.x_pos_t)
        coords_true = to_numpy(g_true.x_pos_t)
        von_mises_pred = to_numpy(g_pred.von_mises).reshape(-1)
        von_mises_true = to_numpy(g_true.von_mises).reshape(-1)

        vm_max = max(vm_max, np.max(von_mises_pred))
        vm_max = max(vm_max, np.max(von_mises_true))
        coordinates_for_bounds.append(coords_true)

        if bbox_only_on_true:
            continue
        coordinates_for_bounds.append(coords_pred)

    coordinate_min, coordinate_max, _, _ = compute_coordinate_bounds(coordinates_for_bounds)

    # Create figure
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"})

    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vm_min, vmax=vm_max)

    # Create colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("von Mises Stress")
    fig.suptitle(f"Beam Comparison\n[{split_name}, 1 sample, {len(pred_rollout)} timesteps]", fontsize=13)

    polycoll_true = None
    polycoll_pred = None

    # Configure axes
    for ax, title in zip(axarr, ["Ground Truth", "Predicted MeshGraphNet"]):
        ax.set_xlim(coordinate_min[0], coordinate_max[0])
        ax.set_ylim(coordinate_min[1], coordinate_max[1])
        ax.set_zlim(coordinate_min[2], coordinate_max[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((L * 0.1, 0.8 * W, 2.5 * D))
        ax.view_init(elev=30, azim=45)
        ax.set_title(title)

    # Iterate over time steps
    for i, (graph_pred, graph_true) in enumerate(zip(pred_rollout, true_rollout)):
        if verbose:
            print(f"Rendering frame {i}/{len(pred_rollout) - 1}")

        coords_pred = to_numpy(graph_pred.x_pos_t)
        coords_true = to_numpy(graph_true.x_pos_t)
        vm_pred = to_numpy(graph_pred.von_mises).reshape(-1)
        vm_true = to_numpy(graph_true.von_mises).reshape(-1)

        # Update both subplots
        polycoll_true = update_surface_collection(axarr[0], faces, coords_true, vm_true, cmap, norm, polycoll_true)
        polycoll_pred = update_surface_collection(axarr[1], faces, coords_pred, vm_pred, cmap, norm, polycoll_pred)

        # Render to numpy array
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)

    plt.close(fig)

    # Create GIF
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"\tGIF saved => {gif_path}")


def make_ground_truth_gif(trajectory_graphs, L, W, D, gif_path, fps, verbose, split_name, traj_idx):
    """
    Create a single-panel GIF showing ground truth mesh colored by von Mises stress.

    Args:
        trajectory_graphs: list
            List of graphs for one trajectory. Each graph must have
            x_pos_t (N,3), von_mises (N,1), and x_element_connectivity.
        L: float
        W: float
        D: float
            Beam dimensions for aspect ratio.
        gif_path: Path
            Output GIF filename.
        fps: int
            Frames per second.
        verbose: bool
            if true, prints more
        split_name: str
            name of split
        traj_idx: idx
            index of trajectory in the split
    """
    frames = []
    faces = extract_surface_faces(trajectory_graphs[0].x_element_connectivity)

    coordinates_for_bounds = []
    vm_min, vm_max = 0.0, 0.0
    for graph in trajectory_graphs:
        coords = to_numpy(graph.x_pos_t)
        vm = to_numpy(graph.von_mises).reshape(-1)
        vm_max = max(vm_max, float(np.max(vm)))
        coordinates_for_bounds.append(coords)

    coordinate_min, coordinate_max, _, _ = compute_coordinate_bounds(coordinates_for_bounds)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vm_min, vmax=vm_max)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.1)
    cbar.set_label("von Mises Stress")

    ax.set_xlim(coordinate_min[0], coordinate_max[0])
    ax.set_ylim(coordinate_min[1], coordinate_max[1])
    ax.set_zlim(coordinate_min[2], coordinate_max[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((L * 0.1, 0.8 * W, 2.5 * D))
    ax.view_init(elev=30, azim=45)

    poly_collection = None
    for i, graph in enumerate(trajectory_graphs):
        coords = to_numpy(graph.x_pos_t)
        vm = to_numpy(graph.von_mises).reshape(-1)
        poly_collection = update_surface_collection(ax, faces, coords, vm, cmap, norm, poly_collection)
        ax.set_title(f"Ground Truth, step {i}\n[{split_name}, traj {traj_idx}, {len(trajectory_graphs)} timesteps]")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    if verbose:
        print(f"\t\tGIF saved => {gif_path}")


def extract_scalar(graph, field_name):
    """
    Args:
        graph: Data
            Rollout graph containing nodewise prediction or target values.
        field_name: str
            One of `acc_x`, `acc_y`, `acc_z`, `von_mises`.
        return: np.ndarray
            Flattened scalar values for all nodes.
    """
    if field_name == "von_mises":
        return to_numpy(graph.von_mises).reshape(-1)

    component_index = {"acc_x": 0, "acc_y": 1, "acc_z": 2}[field_name]
    return to_numpy(graph.acceleration)[:, component_index].reshape(-1)


def _compute_error(true_values, pred_values, error_mode):
    """
    Args:
        true_values: np.ndarray
            Target scalar values.
        pred_values: np.ndarray
            Predicted scalar values.
        error_mode: str
            One of `absolute`, `relative`, `signed_relative`.
        return: np.ndarray
            Nodewise error values for plotting.
    """
    signed_error = pred_values - true_values
    if error_mode == "absolute":
        return np.abs(signed_error)
    if error_mode == "relative":
        denominator = np.maximum(np.abs(true_values), 1e-12)
        return np.abs(signed_error) / denominator
    if error_mode == "signed_relative":
        denominator = np.maximum(np.abs(true_values), 1e-12)
        return signed_error / denominator
    raise ValueError(f"Unsupported error mode: {error_mode}")


def _compute_limits(all_values, error_mode):
    """
    Args:
        all_values: np.ndarray
            Values aggregated across the full rollout.
        error_mode: str
            One of `absolute`, `relative`, `signed_relative`.
        return: tuple
            `(vmin, vmax)` for a stable color scale.
    """
    if error_mode == "signed_relative":
        max_abs = min(float(np.max(np.abs(all_values))), RELATIVE_ERROR_CLIP)
        if max_abs == 0.0:
            max_abs = 1e-12
        return -max_abs, max_abs

    vmax = float(np.max(all_values))
    if error_mode == "relative":
        vmax = min(vmax, RELATIVE_ERROR_CLIP)
    if vmax == 0.0:
        vmax = 1e-12
    return 0.0, vmax


def build_rollout_summary(true_rollout, pred_rollout):
    """
    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        return: dict
            Cached rollout data shared by all validation plots.
    """
    coordinates_per_step = []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        coordinates_per_step.append(to_numpy(true_graph.x_pos_t))

    coordinate_min, coordinate_max, coordinate_span, padding = compute_coordinate_bounds(coordinates_per_step)
    summary = {
        "coordinates_per_step": coordinates_per_step,
        "faces": extract_surface_faces(true_rollout[0].x_element_connectivity),
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "coordinate_span": coordinate_span,
        "padding": padding,
        "fields": {},
    }

    for field_name, field_label in TARGET_FIELDS:
        target_per_step = []
        prediction_per_step = []
        residual_errors_per_step = []
        absolute_errors_per_step = []
        signed_relative_errors_per_step = []
        edge_error_i = []
        edge_error_j = []

        for true_graph, pred_graph in zip(true_rollout, pred_rollout):
            target_values = extract_scalar(true_graph, field_name)
            prediction_values = extract_scalar(pred_graph, field_name)
            residual_errors = prediction_values - target_values
            absolute_errors = np.abs(residual_errors)
            signed_relative_errors = _compute_error(target_values, prediction_values, "signed_relative")
            edge_index = to_numpy(true_graph.edge_index).astype(np.int64)

            target_per_step.append(target_values)
            prediction_per_step.append(prediction_values)
            residual_errors_per_step.append(residual_errors)
            absolute_errors_per_step.append(absolute_errors)
            signed_relative_errors_per_step.append(signed_relative_errors)
            edge_error_i.append(residual_errors[edge_index[0]])
            edge_error_j.append(residual_errors[edge_index[1]])

        errors_by_mode = {
            "absolute": absolute_errors_per_step,
            "signed_relative": signed_relative_errors_per_step,
        }
        summary["fields"][field_name] = {
            "label": field_label,
            "target_values": np.concatenate(target_per_step),
            "prediction_values": np.concatenate(prediction_per_step),
            "edge_error_i": np.concatenate(edge_error_i),
            "edge_error_j": np.concatenate(edge_error_j),
            "errors_by_mode": errors_by_mode,
            "limits_by_mode": {
                error_mode: _compute_limits(np.concatenate(error_values), error_mode)
                for error_mode, error_values in errors_by_mode.items()
            },
        }

    return summary


def _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path, split_name, num_timesteps):
    """
    Args:
        rollout_summary: dict
            Cached rollout data shared by all validation plots.
        field_name: str
            Scalar field identifier.
        error_mode: str
            Supported mode: `signed_relative`.
        output_path: Path
            Destination GIF path.
        return: None
            Saves the animated error mesh to disk.
    """
    field_summary = rollout_summary["fields"][field_name]
    field_label = field_summary["label"]
    coordinates_per_step = rollout_summary["coordinates_per_step"]
    error_per_step = field_summary["errors_by_mode"][error_mode]
    faces = rollout_summary["faces"]
    coordinate_min = rollout_summary["coordinate_min"]
    coordinate_max = rollout_summary["coordinate_max"]
    coordinate_span = rollout_summary["coordinate_span"]
    padding = rollout_summary["padding"]
    vmin, vmax = field_summary["limits_by_mode"][error_mode]
    cmap = plt.cm.coolwarm if error_mode == "signed_relative" else plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.12)
    if error_mode != "signed_relative":
        raise ValueError(f"Unsupported rollout heatmap mode: {error_mode}")
    colorbar.set_label(f"{field_label} signed relative error [%, clipped at \u00b1500%]")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:+.0f}%"))
    ax.set_xlim(coordinate_min[0] - padding[0], coordinate_max[0] + padding[0])
    ax.set_ylim(coordinate_min[1] - padding[1], coordinate_max[1] + padding[1])
    ax.set_zlim(coordinate_min[2] - padding[2], coordinate_max[2] + padding[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(tuple(coordinate_span + padding))
    ax.view_init(elev=30, azim=45)

    poly_collection = None
    frames = []
    for timestep, (coordinates, error_values) in enumerate(zip(coordinates_per_step, error_per_step)):
        error_values = np.clip(error_values, vmin, vmax)
        poly_collection = update_surface_collection(ax, faces, coordinates, error_values, cmap, norm, poly_collection)
        ax.set_title(f"{field_label} Signed Relative Error [%, clipped \u00b1500%], step {timestep}\n[{split_name}, "
                     f"1 sample, {num_timesteps} timesteps]")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=4, loop=0)


def plot_prediction_target_scatter(field_summary, output_path, split_name, num_timesteps):
    """
    Args:
        field_summary: dict
            Cached values for one scalar field.
        output_path: Path
            Destination image path.
        split_name: str
            Split label for the title.
        num_timesteps: int
            Number of timesteps for the title.
        return: None
            Saves the scatter plot to disk with sklearn.metrics.r2_score(target, prediction) annotated.
    """
    target_values = field_summary["target_values"]
    prediction_values = field_summary["prediction_values"]
    min_value = min(target_values.min(), prediction_values.min())
    max_value = max(target_values.max(), prediction_values.max())

    r2 = r2_prediction_vs_target(target_values, prediction_values)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(target_values, prediction_values, s=10, alpha=0.15)
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{field_summary['label']} Prediction vs Target\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.text(0.03, 0.97, format_r2_text(r2), transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_edge_error_correlation(field_summary, output_path, split_name, num_timesteps):
    """
    Args:
        field_summary: dict
            Cached values for one scalar field.
        output_path: Path
            Destination image path.
        split_name: str
            Split label for the title.
        num_timesteps: int
            Number of timesteps for the title.
        return: None
            Saves the hexbin plot to disk; annotates in-sample R^2 of OLS predicting error_j from error_i (sklearn).
    """
    error_i = field_summary["edge_error_i"]
    error_j = field_summary["edge_error_j"]
    if len(error_i) == 0 or np.std(error_i) == 0.0 or np.std(error_j) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(error_i, error_j)[0, 1])
    r2_lin = r2_linear_fit_x_to_y(error_i, error_j)

    fig, ax = plt.subplots(figsize=(8, 6))
    hexbin = ax.hexbin(error_i, error_j, gridsize=60, cmap="viridis", mincnt=1)
    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Edge count")
    ax.set_xlabel("error_i")
    ax.set_ylabel("error_j")
    ax.set_title(
        f"{field_summary['label']} Edge Error Correlation (corr={correlation:.4f})\n"
        f"[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.text(0.03, 0.97, format_r2_text(r2_lin), transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_roll_err_pos_vs_time(true_rollout, pred_rollout, output_path, split_name):
    """
    Plot position MAE at each rollout timestep.

    Args:
        true_rollout: list
            Ground-truth rollout graphs (each has x_pos_t).
        pred_rollout: list
            Predicted rollout graphs (each has x_pos_t).
        output_path: Path
            Destination image path.
        return: None
    """
    print("Plot rollout position error vs timestep...")
    errors = []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        step_mae = torch.mean(torch.abs(true_graph.x_pos_t - pred_graph.x_pos_t)).item()
        errors.append(step_mae)

    timesteps = np.arange(len(errors))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, errors, linewidth=2.0, marker="o", markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position MAE")
    ax.set_title(f"Rollout Position Error vs Timestep\n[{split_name}, 1 sample, {len(errors)} timesteps]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_err_heatmap_rollout(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
    """
    Plot rollout of the ground truth mesh (evolving over time, so a gif), with on every node of the mesh a heatmap
    which corresponds to the signed relative error
    - signed_relative (percentage, clipped to +/-500%)

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Directory where diagnostic artifacts will be saved.
        filename_prefix: str
            Prefix used for saved artifact names.
        return: None
            Saves all requested validation plots.
    """
    output_dir = Path(output_dir)
    rollout_summary = build_rollout_summary(true_rollout, pred_rollout)
    print("Printing validation rollout...")
    for field_name, field_label in TARGET_FIELDS:
        for error_mode in ["signed_relative"]:
            output_path = output_dir / f"{filename_prefix}_{field_name}_{error_mode}_error.gif"
            _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path, split_name, len(true_rollout))

def plot_per_trajectory_bar(traj_indices, values, ylabel, title, output_path, color, true_values):
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


def plot_scatter_edge(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
    """
    Plot
    1. scatter plots: prediction vs target, where every p[oint is one node
    2. edge error correlation:

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Directory where diagnostic artifacts will be saved.
        filename_prefix: str
            Prefix used for saved artifact names.
    return: None
        Saves all requested validation plots.
    """
    print("Printing validation scatter plots and edge correlation...")
    output_dir = Path(output_dir)
    rollout_summary = build_rollout_summary(true_rollout, pred_rollout)
    num_timesteps = len(true_rollout)
    for field_name, field_label in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"
        correlation_path = output_dir / f"{filename_prefix}_{field_name}_edge_error_correlation.png"
        field_summary = rollout_summary["fields"][field_name]
        plot_prediction_target_scatter(field_summary, scatter_path, split_name, num_timesteps)
        plot_edge_error_correlation(field_summary, correlation_path, split_name, num_timesteps)