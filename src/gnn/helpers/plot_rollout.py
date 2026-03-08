import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from src.gnn.helpers.mesh_build import (compute_coordinate_bounds, extract_surface_faces, extract_tetra_connectivity,
                                        to_numpy, update_surface_collection)
from pathlib import Path
from mesh_build import volume

def plot_volume_change(pred_rollout, true_rollout, path):
    """
    Plot volume change over time for predicted vs ground truth.

    Args:
        pred_rollout: list[FIXME: TYPE]
            list of predictions (each element attributes = targets)
        true_rollout: list[FIXME: TYPE]
            list of ground truth values (each element has attributes = targets)
        path: Path
    """
    fig, ax = plt.subplots()
    true_volumes_list, pred_volumes_list = [], []

    for pred, true in zip(pred_rollout, true_rollout):
        x_true = true.x_pos_t
        x_pred = pred.x_pos_t
        cells = torch.as_tensor(extract_tetra_connectivity(true.x_element_connectivity), dtype=torch.long)
        true_volumes = volume(x_true, cells)
        pred_volumes = volume(x_pred, cells)

        V_true = np.abs(true_volumes).sum()
        V_pred = np.abs(pred_volumes).sum()

        true_volumes_list.append(V_true)
        pred_volumes_list.append(V_pred)

    ax.plot(true_volumes_list, label="True Volumes")
    ax.plot(pred_volumes_list, label="Predicted Volumes")
    plt.xlabel("Time Step")
    plt.ylabel("Volume")
    ax.legend()
    plt.savefig(path)
    plt.close()

def make_beam_comparison_gif(pred_rollout, true_rollout, L, W, D, out_gif, fps, bbox_only_on_true, verbose):
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
        out_gif: Path
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
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"GIF saved => {out_gif}")

TARGET_FIELDS = [
    ("acc_x", "Acceleration X"),
    ("acc_y", "Acceleration Y"),
    ("acc_z", "Acceleration Z"),
    ("von_mises", "von Mises"),
]


def _extract_scalar(graph, field_name):
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
            One of `absolute`, `relative`, `signed`.
        return: np.ndarray
            Nodewise error values for plotting.
    """
    signed_error = pred_values - true_values
    if error_mode == "absolute":
        return np.abs(signed_error)
    if error_mode == "relative":
        denominator = np.maximum(np.abs(true_values), 1e-12)
        return np.abs(signed_error) / denominator
    if error_mode == "signed":
        return signed_error
    raise ValueError(f"Unsupported error mode: {error_mode}")


def _compute_limits(all_values, error_mode):
    """
    Args:
        all_values: np.ndarray
            Values aggregated across the full rollout.
        error_mode: str
            One of `absolute`, `relative`, `signed`.
        return: tuple
            `(vmin, vmax)` for a stable color scale.
    """
    if error_mode == "signed":
        max_abs = float(np.max(np.abs(all_values)))
        if max_abs == 0.0:
            max_abs = 1e-12
        return -max_abs, max_abs

    vmax = float(np.max(all_values))
    if vmax == 0.0:
        vmax = 1e-12
    return 0.0, vmax


def _build_rollout_summary(true_rollout, pred_rollout):
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
        signed_errors_per_step = []
        absolute_errors_per_step = []
        relative_errors_per_step = []
        edge_error_i = []
        edge_error_j = []

        for true_graph, pred_graph in zip(true_rollout, pred_rollout):
            target_values = _extract_scalar(true_graph, field_name)
            prediction_values = _extract_scalar(pred_graph, field_name)
            signed_errors = _compute_error(target_values, prediction_values, "signed")
            absolute_errors = np.abs(signed_errors)
            relative_errors = _compute_error(target_values, prediction_values, "relative")
            edge_index = to_numpy(true_graph.edge_index).astype(np.int64)

            target_per_step.append(target_values)
            prediction_per_step.append(prediction_values)
            signed_errors_per_step.append(signed_errors)
            absolute_errors_per_step.append(absolute_errors)
            relative_errors_per_step.append(relative_errors)
            edge_error_i.append(signed_errors[edge_index[0]])
            edge_error_j.append(signed_errors[edge_index[1]])

        errors_by_mode = {
            "absolute": absolute_errors_per_step,
            "relative": relative_errors_per_step,
            "signed": signed_errors_per_step,
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


def _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path):
    """
    Args:
        rollout_summary: dict
            Cached rollout data shared by all validation plots.
        field_name: str
            Scalar field identifier.
        error_mode: str
            One of `absolute`, `relative`, `signed`.
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
    cmap = plt.cm.coolwarm if error_mode == "signed" else plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.12)
    colorbar.set_label(f"{field_label} {error_mode} error")
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
        poly_collection = update_surface_collection(ax, faces, coordinates, error_values, cmap, norm, poly_collection)
        ax.set_title(f"{field_label} {error_mode.title()} Error, step {timestep}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=4, loop=0)


def _plot_prediction_target_scatter(field_summary, output_path):
    """
    Args:
        field_summary: dict
            Cached values for one scalar field.
        output_path: Path
            Destination image path.
        return: None
            Saves the scatter plot to disk.
    """
    target_values = field_summary["target_values"]
    prediction_values = field_summary["prediction_values"]
    min_value = min(target_values.min(), prediction_values.min())
    max_value = max(target_values.max(), prediction_values.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(target_values, prediction_values, s=10, alpha=0.15)
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{field_summary['label']} Prediction vs Target")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_edge_error_correlation(field_summary, output_path):
    """
    Args:
        field_summary: dict
            Cached values for one scalar field.
        output_path: Path
            Destination image path.
        return: None
            Saves the edge error correlation plot to disk.
    """
    error_i = field_summary["edge_error_i"]
    error_j = field_summary["edge_error_j"]
    if len(error_i) == 0 or np.std(error_i) == 0.0 or np.std(error_j) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(error_i, error_j)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    hexbin = ax.hexbin(error_i, error_j, gridsize=60, cmap="viridis", mincnt=1)
    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Edge count")
    ax.set_xlabel("error_i")
    ax.set_ylabel("error_j")
    ax.set_title(f"{field_summary['label']} Edge Error Correlation (corr={correlation:.4f})")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_validation_diagnostics(true_rollout, pred_rollout, output_dir, output_prefix):
    """
    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Directory where diagnostic artifacts will be saved.
        output_prefix: str
            Prefix used for saved artifact names.
        return: None
            Saves all requested validation plots.
    """
    output_dir = Path(output_dir)
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)
    for field_name, field_label in TARGET_FIELDS:
        for error_mode in ["absolute", "relative", "signed"]:
            output_path = output_dir / f"{output_prefix}_{field_name}_{error_mode}_error.gif"
            _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path)

        scatter_path = output_dir / f"{output_prefix}_{field_name}_prediction_vs_target.png"
        correlation_path = output_dir / f"{output_prefix}_{field_name}_edge_error_correlation.png"
        field_summary = rollout_summary["fields"][field_name]
        _plot_prediction_target_scatter(field_summary, scatter_path)
        _plot_edge_error_correlation(field_summary, correlation_path)