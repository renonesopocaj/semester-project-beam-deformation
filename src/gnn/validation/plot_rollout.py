import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from matplotlib.ticker import FuncFormatter
from src.gnn.validation.mesh_build_numerical import (compute_coordinate_bounds, extract_surface_faces,
                                                     extract_tetra_connectivity, to_numpy,
                                                     update_surface_collection, volume,
                                                     compute_shortest_path_distances,
                                                     extract_boundary_node_set,
                                                     compute_discrete_mean_curvature,
                                                     compute_dirichlet_energy,
                                                     compute_graph_laplacian_spectrum)
from pathlib import Path

RELATIVE_ERROR_CLIP = 5.0  # 500 %
TARGET_FIELDS = [
    ("acc_x", "Acceleration X"),
    ("acc_y", "Acceleration Y"),
    ("acc_z", "Acceleration Z"),
    ("von_mises", "von Mises"),
]


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

def make_beam_comparison_gif(pred_rollout, true_rollout, L, W, D, gif_path, fps, bbox_only_on_true, verbose):
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


def make_ground_truth_gif(trajectory_graphs, L, W, D, gif_path, fps, verbose):
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
        ax.set_title(f"Ground Truth, step {i}")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"\t\tGIF saved => {gif_path}")


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
        residual_errors_per_step = []
        absolute_errors_per_step = []
        signed_relative_errors_per_step = []
        edge_error_i = []
        edge_error_j = []

        for true_graph, pred_graph in zip(true_rollout, pred_rollout):
            target_values = _extract_scalar(true_graph, field_name)
            prediction_values = _extract_scalar(pred_graph, field_name)
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


def _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path):
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
        ax.set_title(f"{field_label} Signed Relative Error [%, clipped \u00b1500%], step {timestep}")
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

def plot_rollout_error_vs_timestep(true_rollout, pred_rollout, output_path):
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
    errors = []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        step_mae = torch.mean(torch.abs(true_graph.x_pos_t - pred_graph.x_pos_t)).item()
        errors.append(step_mae)

    timesteps = np.arange(len(errors))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, errors, linewidth=2.0, marker="o", markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position MAE")
    ax.set_title("Rollout Position Error vs Timestep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_validation_diagnostics_rollout(true_rollout, pred_rollout, output_dir, filename_prefix):
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
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)
    for field_name, field_label in TARGET_FIELDS:
        for error_mode in ["signed_relative"]:
            output_path = output_dir / f"{filename_prefix}_{field_name}_{error_mode}_error.gif"
            _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path)


# ---------------------------------------------------------------------------
#  Mesh diagnostics (5.1, 5.2, 7.1, 7.2, 8.3, 8.4)
# ---------------------------------------------------------------------------

def _plot_neighbor_abs_error(true_values, pred_values, edge_index, field_label, output_path):
    """
    Diagnostic 5.1 -- scatter of absolute error on neighboring nodes.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    abs_error = np.abs(pred_values - true_values)
    a_i = abs_error[edge_index[0]]
    a_j = abs_error[edge_index[1]]

    if len(a_i) == 0 or np.std(a_i) == 0.0 or np.std(a_j) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(a_i, a_j)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 7))
    hexbin = ax.hexbin(a_i, a_j, gridsize=60, cmap="viridis", mincnt=1)
    fig.colorbar(hexbin, ax=ax, label="Edge count")
    ax.set_xlabel("Absolute error node i")
    ax.set_ylabel("Absolute error node j")
    ax.set_title(f"{field_label} Neighbor Abs-Error (corr={correlation:.4f})")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_distance_from_worst(true_values, pred_values, edge_index, num_nodes, field_label, output_path):
    """
    Diagnostic 5.2 -- scatter of absolute error vs hop-distance from the
    worst-error node.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        num_nodes: int
            Total number of nodes.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    abs_error = np.abs(pred_values - true_values)
    worst_node = int(np.argmax(abs_error))
    distances = compute_shortest_path_distances(edge_index, [worst_node], num_nodes)
    reachable = distances >= 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distances[reachable], abs_error[reachable], s=8, alpha=0.3)
    ax.set_xlabel(f"Hop distance from worst node (node {worst_node})")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"{field_label} Error vs Distance from Worst Node")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_dirichlet_energy_bar(true_rollout_graph, pred_rollout_graph, output_path):
    """
    Diagnostic 7.1 -- grouped bar chart of Dirichlet energy E(y) vs E(y_hat)
    for all target fields, computed at a single timestep.

    Args:
        true_rollout_graph: Data
            Single ground-truth rollout graph.
        pred_rollout_graph: Data
            Single predicted rollout graph.
        output_path: Path
            Destination image path.
    """
    edge_index = to_numpy(true_rollout_graph.edge_index).astype(np.int64)
    field_labels = []
    energy_true_list = []
    energy_pred_list = []

    for field_name, field_label in TARGET_FIELDS:
        true_vals = _extract_scalar(true_rollout_graph, field_name)
        pred_vals = _extract_scalar(pred_rollout_graph, field_name)
        energy_true_list.append(compute_dirichlet_energy(true_vals, edge_index))
        energy_pred_list.append(compute_dirichlet_energy(pred_vals, edge_index))
        field_labels.append(field_label)

    x_pos = np.arange(len(field_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - bar_width / 2, energy_true_list, bar_width, label="True E(y)")
    ax.bar(x_pos + bar_width / 2, energy_pred_list, bar_width, label=r"Predicted E($\hat{y}$)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(field_labels, rotation=15, ha="right")
    ax.set_ylabel("Dirichlet Energy")
    ax.set_title("Dirichlet Energy: True vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_spectral_comparison(true_rollout_graph, pred_rollout_graph, output_path, k):
    """
    Diagnostic 7.2 -- line plots comparing spectral content (graph Laplacian)
    of true vs predicted fields. One subplot per target field.

    Args:
        true_rollout_graph: Data
            Single ground-truth rollout graph.
        pred_rollout_graph: Data
            Single predicted rollout graph.
        output_path: Path
            Destination image path.
        k: int
            Number of Laplacian eigenvalues to compute.
    """
    edge_index = to_numpy(true_rollout_graph.edge_index).astype(np.int64)
    num_nodes = int(to_numpy(true_rollout_graph.x_pos_t).shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, (field_name, field_label) in enumerate(TARGET_FIELDS):
        ax = axes[idx]
        true_signal = _extract_scalar(true_rollout_graph, field_name)
        pred_signal = _extract_scalar(pred_rollout_graph, field_name)

        result_true = compute_graph_laplacian_spectrum(edge_index, true_signal, num_nodes, k)
        result_pred = compute_graph_laplacian_spectrum(edge_index, pred_signal, num_nodes, k)

        if result_true is None or result_pred is None:
            ax.text(0.5, 0.5, "Spectral analysis unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(field_label)
            continue

        eig_true, coeff_true = result_true
        eig_pred, coeff_pred = result_pred

        ax.plot(np.arange(len(coeff_true)), coeff_true, label="True", linewidth=1.5)
        ax.plot(np.arange(len(coeff_pred)), coeff_pred, label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(r"$|c_k|^2$")
        ax.set_title(f"{field_label} Spectral Content")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log", nonpositive="clip")

    fig.suptitle("Low-Frequency vs High-Frequency Content", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_curvature(true_values, pred_values, positions, element_connectivity, field_label, output_path):
    """
    Diagnostic 8.3 -- scatter of absolute error vs mean curvature at
    surface (boundary) nodes.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        positions: np.ndarray
            (N, 3) node coordinates (true mesh).
        element_connectivity: object
            Tetrahedral connectivity from the graph.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    boundary_faces = extract_surface_faces(element_connectivity)
    boundary_nodes = np.unique(boundary_faces.ravel())
    node_indices, curvature = compute_discrete_mean_curvature(positions, boundary_faces, boundary_nodes)

    abs_error = np.abs(pred_values - true_values)
    error_at_boundary = abs_error[node_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(curvature, error_at_boundary, s=8, alpha=0.3)
    ax.set_xlabel("Mean Curvature")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"{field_label} Error vs Curvature (surface nodes)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_boundary_distance(true_values, pred_values, edge_index, num_nodes, element_connectivity,
                                     field_label, output_path):
    """
    Diagnostic 8.4 -- scatter of absolute error vs hop-distance to the
    nearest surface (boundary) node.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        num_nodes: int
            Total number of nodes.
        element_connectivity: object
            Tetrahedral connectivity from the graph.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    boundary_nodes = extract_boundary_node_set(element_connectivity)
    distances = compute_shortest_path_distances(edge_index, boundary_nodes, num_nodes)
    reachable = distances >= 0

    abs_error = np.abs(pred_values - true_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distances[reachable], abs_error[reachable], s=8, alpha=0.3)
    ax.set_xlabel("Hop distance to boundary")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"{field_label} Error vs Distance to Boundary")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_mesh_diagnostics(true_rollout, pred_rollout, output_dir, filename_prefix):
    """
    Orchestrator that runs all six mesh diagnostics (5.1, 5.2, 7.1, 7.2,
    8.3, 8.4) on the last rollout timestep.

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Parent directory for output; a ``diagnostics/`` subdirectory is created.
        filename_prefix: str
            Prefix used for saved file names.
    """
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    true_graph = true_rollout[-1]
    pred_graph = pred_rollout[-1]
    edge_index = to_numpy(true_graph.edge_index).astype(np.int64)
    positions = to_numpy(true_graph.x_pos_t)
    num_nodes = positions.shape[0]

    for field_name, field_label in TARGET_FIELDS:
        true_vals = _extract_scalar(true_graph, field_name)
        pred_vals = _extract_scalar(pred_graph, field_name)

        # 5.1 Neighbor absolute error
        print(f"\t\tDiag 5.1 ({field_name})...")
        _plot_neighbor_abs_error(true_vals, pred_vals, edge_index, field_label,
                                 diag_dir / f"{filename_prefix}_{field_name}_neighbor_abs_error.png")

        # 5.2 Error vs distance from worst node
        print(f"\t\tDiag 5.2 ({field_name})...")
        _plot_error_vs_distance_from_worst(true_vals, pred_vals, edge_index, num_nodes, field_label,
                                           diag_dir / f"{filename_prefix}_{field_name}_error_vs_dist_worst.png")

        # 8.3 Error vs curvature
        print(f"\t\tDiag 8.3 ({field_name})...")
        _plot_error_vs_curvature(true_vals, pred_vals, positions, true_graph.x_element_connectivity,
                                 field_label,
                                 diag_dir / f"{filename_prefix}_{field_name}_error_vs_curvature.png")

        # 8.4 Error vs boundary distance
        print(f"\t\tDiag 8.4 ({field_name})...")
        _plot_error_vs_boundary_distance(true_vals, pred_vals, edge_index, num_nodes,
                                         true_graph.x_element_connectivity, field_label,
                                         diag_dir / f"{filename_prefix}_{field_name}_error_vs_boundary_dist.png")

    # 7.1 Dirichlet energy (single combined plot)
    print("\t\tDiag 7.1 (Dirichlet energy)...")
    _plot_dirichlet_energy_bar(true_graph, pred_graph,
                               diag_dir / f"{filename_prefix}_dirichlet_energy.png")

    # 7.2 Spectral comparison (single combined figure)
    print("\t\tDiag 7.2 (spectral comparison)...")
    _plot_spectral_comparison(true_graph, pred_graph,
                              diag_dir / f"{filename_prefix}_spectral_comparison.png", k=50)