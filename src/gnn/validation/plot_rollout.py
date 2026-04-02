"""
plot_rollout.py
Contains all files that plot metrics or diagnostics for rollout, in depth for each trajectory.
Main access points: in_depth_plots()
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from matplotlib.ticker import FuncFormatter
from src.gnn.validation.plot_diagnostic import (extract_surface_faces, extract_scalar,
                                                plot_mesh_diagnostics)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.gnn.mesh_numerical_helpers.compute_volume import volume
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import to_numpy_tetra_connectivity, to_numpy
from src.gnn.misc_helpers.compute_r2 import (format_r2_text, r2_linear_fit_x_to_y,
                                             r2_prediction_vs_target)
from pathlib import Path

RELATIVE_ERROR_CLIP = 5.0  # 500 %
TARGET_FIELDS = [("acc_x", "Acceleration X"),
                 ("acc_y", "Acceleration Y"),
                 ("acc_z", "Acceleration Z"),
                 ("von_mises", "von Mises")]

def _finite_coordinate_bounds_from_stack(all_coordinates):
    """
    Compute axis-aligned bounds using only rows with finite XYZ.

    Args:
        all_coordinates: np.ndarray
            Stacked node coordinates, shape (N, 3).
    return: tuple
        (coordinate_min, coordinate_max, coordinate_span, padding) all shape-(3,) arrays.
    """
    mask = np.isfinite(all_coordinates).all(axis=1)
    n_row = int(all_coordinates.shape[0])
    n_finite_row = int(mask.sum())
    if n_finite_row < n_row:
        print("[WARN] _finite_coordinate_bounds_from_stack: %d of %d node rows have non-finite XYZ; "
              "bounds use only finite rows." % (n_row - n_finite_row, n_row))
    if not np.any(mask):
        print("[WARN] _finite_coordinate_bounds_from_stack: no finite coordinates; using fallback bounds.")
        coordinate_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        coordinate_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        finite_coords = all_coordinates[mask]
        coordinate_min = finite_coords.min(axis=0).astype(np.float64)
        coordinate_max = finite_coords.max(axis=0).astype(np.float64)
    eps = 1e-9
    for axis in range(3):
        if not np.isfinite(coordinate_min[axis]) or not np.isfinite(coordinate_max[axis]):
            coordinate_min[axis] = -1.0
            coordinate_max[axis] = 1.0
        elif coordinate_max[axis] <= coordinate_min[axis]:
            coordinate_max[axis] = coordinate_min[axis] + eps
    coordinate_span = np.maximum(coordinate_max - coordinate_min, eps)
    padding = 0.05 * coordinate_span
    if not (np.isfinite(coordinate_min).all() and np.isfinite(coordinate_max).all()
            and np.isfinite(coordinate_span).all() and np.isfinite(padding).all()):
        print("[WARN] _finite_coordinate_bounds_from_stack: non-finite min/max/span/padding after fix; "
              "forcing symmetric fallback box.")
        coordinate_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        coordinate_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        coordinate_span = np.maximum(coordinate_max - coordinate_min, eps)
        padding = 0.05 * coordinate_span
    return coordinate_min, coordinate_max, coordinate_span, padding


def _finite_scalar_max_from_arrays(arrays):
    """
    Maximum over finite entries of multiple 1D-array-like objects.

    Args:
        arrays: list
            List of array-like scalars per node (von Mises, etc.).
    return: float
        Strictly positive scale for color normalization; 1.0 if nothing finite.
    """
    max_val = None
    n_nonfinite = 0
    n_totalElts = 0
    for arr in arrays:
        flat = np.asarray(arr).reshape(-1)
        n_totalElts += int(flat.size)
        n_nonfinite += int((~np.isfinite(flat)).sum())
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            continue
        local_max = float(np.max(finite))
        if max_val is None or local_max > max_val:
            max_val = local_max
    if n_nonfinite > 0:
        print("[WARN] _finite_scalar_max_from_arrays: %d non-finite of %d scalar samples ignored for vmax."
              % (n_nonfinite, n_totalElts))
    if max_val is None or not np.isfinite(max_val) or max_val <= 0.0:
        print("[WARN] _finite_scalar_max_from_arrays: no positive finite vmax; using 1.0 for color scale.")
        return 1.0
    return max_val


def _plot_volume_change(pred_rollout, true_rollout, path, split_name):
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
        cells = torch.as_tensor(to_numpy_tetra_connectivity(true.x_element_connectivity), dtype=torch.long)
        true_volumes = volume(x_true, cells)
        pred_volumes = volume(x_pred, cells)
        # sum the absolute value of all volumes
        V_true = torch.abs(true_volumes).sum().detach().cpu().item()
        V_pred = torch.abs(pred_volumes).sum().detach().cpu().item()
        if not np.isfinite(V_true) or not np.isfinite(V_pred):
            print("[WARN] _plot_volume_change: timestep non-finite aggregate volume (V_true=%s, V_pred=%s); "
                  "plot may be misleading." % (V_true, V_pred))

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

def update_surface_collection(ax, faces, coordinates, node_values, cmap, norm, poly_collection):
    """
    Helper function to beam renderings.

    Args:
        ax: matplotlib.axes._subplots.Axes3DSubplot
            Target 3D axis.
        faces: np.ndarray
            Boundary faces with shape `(num_faces, 3)`.
        coordinates: np.ndarray
            Node coordinates with shape `(num_nodes, 3)`.
        node_values: np.ndarray
            Scalar values defined per node.
        cmap: matplotlib.colors.Colormap
            Colormap used for rendering.
        norm: matplotlib.colors.Normalize
            Normalization used for rendering.
        poly_collection: Poly3DCollection or None
            Existing collection to update.
    return: Poly3DCollection
        Created or updated collection.
    """
    if faces.size == 0:
        return poly_collection
    valid_nodes = np.isfinite(coordinates).all(axis=1) & np.isfinite(node_values)
    n_bad_nodes = int((~valid_nodes).sum())
    if n_bad_nodes > 0 and poly_collection is None:
        print("[WARN] update_surface_collection (first draw): %d nodes have non-finite position or nodal scalar; "
              "boundary faces incident on them are dropped in this view." % n_bad_nodes)
    use_faces = faces[valid_nodes[faces].all(axis=1)]
    if use_faces.size < faces.size and poly_collection is None:
        print("[WARN] update_surface_collection (first draw): using %d of %d boundary faces (others touch omitted nodes)."
              % (use_faces.size, faces.size))
    if use_faces.size == 0:
        print("[WARN] update_surface_collection: no faces left after dropping non-finite nodes; frame may be empty.")
        if poly_collection is not None:
            try:
                poly_collection.set_verts(np.zeros((0, 3, 3), dtype=np.float64))
            except Exception:
                pass
        return poly_collection

    polygons = coordinates[use_faces]
    face_values = np.mean(node_values[use_faces], axis=1)
    finite_face = np.isfinite(face_values)
    if not np.all(finite_face):
        n_bad_face = int((~finite_face).sum())
        print("[WARN] update_surface_collection: %d faces have non-finite mean nodal value; "
              "replacing with 0.0 for coloring." % n_bad_face)
        face_values = np.where(finite_face, face_values, 0.0)
    face_colors = cmap(norm(face_values))

    if poly_collection is None:
        poly_collection = Poly3DCollection(polygons)
        poly_collection.set_facecolors(face_colors)
        poly_collection.set_edgecolor("none")
        ax.add_collection3d(poly_collection)
        return poly_collection

    poly_collection.set_verts(polygons)
    poly_collection.set_facecolors(face_colors)
    return poly_collection

def _make_beam_comparison_gif(pred_rollout, true_rollout, L, W, D, gif_path, fps, bbox_only_on_true, verbose, split_name):
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
    vm_arrays = []

    for g_pred, g_true in zip(pred_rollout, true_rollout):
        coords_pred = to_numpy(g_pred.x_pos_t)
        coords_true = to_numpy(g_true.x_pos_t)
        von_mises_pred = to_numpy(g_pred.von_mises).reshape(-1)
        von_mises_true = to_numpy(g_true.von_mises).reshape(-1)

        vm_arrays.append(von_mises_pred)
        vm_arrays.append(von_mises_true)
        coordinates_for_bounds.append(coords_true)

        if bbox_only_on_true:
            continue
        coordinates_for_bounds.append(coords_pred)

    coordinate_min, coordinate_max, _, _ = compute_coordinate_bounds(coordinates_for_bounds)
    vm_min = 0.0
    vm_max = _finite_scalar_max_from_arrays(vm_arrays)

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
    if len(frames) == 0:
        print(f"[WARN] _make_beam_comparison_gif: no frames; skipping {gif_path}.")
        return
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"\tGIF saved => {gif_path}")


def _format_traj_metadata_text(traj_metadata):
    """
    Build a human-readable one-line subtitle from trajectory metadata.

    Args:
        traj_metadata: dict
            Trajectory parameters from trajectory_params.json.
    return: str
        Formatted text for the GIF subtitle.
    """
    fx = traj_metadata.get("fx", "?")
    fy = traj_metadata.get("fy", "?")
    fz = traj_metadata.get("fz", "?")
    beam_L = traj_metadata.get("L", "?")
    beam_W = traj_metadata.get("W", "?")
    ood = traj_metadata.get("ood_features", [])
    ood_str = ", ".join(ood) if ood else "none"
    return (f"fx={fx}  fy={fy}  fz={fz}  |  L={beam_L}  W={beam_W}"
            f"  |  OOD: {ood_str}")


def make_ground_truth_gif(trajectory_graphs, L, W, D, gif_path, fps, verbose, split_name, traj_idx,
                          traj_metadata=None):
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
        traj_metadata: dict or None
            trajectory parameters from trajectory_params.json, shown as subtitle
    """
    frames = []
    faces = extract_surface_faces(trajectory_graphs[0].x_element_connectivity)

    coordinates_for_bounds = []
    vm_arrays = []
    for graph in trajectory_graphs:
        coords = to_numpy(graph.x_pos_t)
        vm = to_numpy(graph.von_mises).reshape(-1)
        vm_arrays.append(vm)
        coordinates_for_bounds.append(coords)

    coordinate_min, coordinate_max, _, _ = compute_coordinate_bounds(coordinates_for_bounds)
    vm_min = 0.0
    vm_max = _finite_scalar_max_from_arrays(vm_arrays)

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

    # Add static metadata subtitle below the 3D plot.
    if traj_metadata is not None:
        subtitle = _format_traj_metadata_text(traj_metadata)
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})

    poly_collection = None
    for i, graph in enumerate(trajectory_graphs):
        coords = to_numpy(graph.x_pos_t)
        vm = to_numpy(graph.von_mises).reshape(-1)
        poly_collection = update_surface_collection(ax, faces, coords, vm, cmap, norm, poly_collection)
        ax.set_title(f"Ground Truth, step {i}\n[{split_name}, traj {traj_idx}, {len(trajectory_graphs)} timesteps]")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    if len(frames) == 0:
        print(f"[WARN] make_ground_truth_gif: no frames; skipping {gif_path}.")
        return
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    if verbose:
        print(f"\t\tGIF saved => {gif_path}")


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
    flat = np.asarray(all_values).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        print("[WARN] _compute_limits: no finite error values; using tiny symmetric range.")
        tiny = 1e-12
        if error_mode == "signed_relative":
            return -tiny, tiny
        return 0.0, tiny

    if error_mode == "signed_relative":
        max_abs = min(float(np.max(np.abs(finite))), RELATIVE_ERROR_CLIP)
        if max_abs == 0.0:
            max_abs = 1e-12
        return -max_abs, max_abs

    vmax = float(np.max(finite))
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
        Rollout data shared by all validation plots.
    """
    coordinates_per_step = []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        coordinates_per_step.append(to_numpy(true_graph.x_pos_t))

    coordinate_min, coordinate_max, coordinate_span, padding = compute_coordinate_bounds(coordinates_per_step)
    summary = {"coordinates_per_step": coordinates_per_step,
               "faces": extract_surface_faces(true_rollout[0].x_element_connectivity),
               "coordinate_min": coordinate_min,
               "coordinate_max": coordinate_max,
               "coordinate_span": coordinate_span,
               "padding": padding,
               "fields": {}}

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

        errors_by_mode = {"absolute": absolute_errors_per_step,
                          "signed_relative": signed_relative_errors_per_step    }
        summary["fields"][field_name] = {
            "label": field_label,
            "target_values": np.concatenate(target_per_step),
            "prediction_values": np.concatenate(prediction_per_step),
            "edge_error_i": np.concatenate(edge_error_i),
            "edge_error_j": np.concatenate(edge_error_j),
            "errors_by_mode": errors_by_mode,
            "limits_by_mode": {error_mode: _compute_limits(np.concatenate(error_values), error_mode)
                               for error_mode, error_values in errors_by_mode.items()}}

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
    n_err_sanitized = 0
    for timestep, (coordinates, error_values) in enumerate(zip(coordinates_per_step, error_per_step)):
        err = np.asarray(error_values, dtype=np.float64)
        n_bad = int((~np.isfinite(err)).sum())
        if n_bad > 0:
            n_err_sanitized += n_bad
        err = np.where(np.isfinite(err), err, 0.0)
        error_values = np.clip(err, vmin, vmax)
        poly_collection = update_surface_collection(ax, faces, coordinates, error_values, cmap, norm, poly_collection)
        ax.set_title(f"{field_label} Signed Relative Error [%, clipped \u00b1500%], step {timestep}\n[{split_name}, "
                     f"1 sample, {num_timesteps} timesteps]")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    if n_err_sanitized > 0:
        print("[WARN] _render_error_mesh_gif (%s): replaced %d non-finite nodal error values with 0.0 before clip."
              % (field_name, n_err_sanitized))
    if len(frames) == 0:
        print(f"[WARN] _render_error_mesh_gif: no frames for {output_path}; skipping GIF.")
        return
    imageio.mimsave(output_path, frames, fps=4, loop=0)


def _plot_prediction_target_scatter(field_summary, output_path, split_name, num_timesteps):
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
    pair_mask = np.isfinite(target_values) & np.isfinite(prediction_values)
    n_tot = int(pair_mask.size)
    n_ok = int(pair_mask.sum())
    if n_ok < n_tot:
        print("[WARN] _plot_prediction_target_scatter (%s): omitting %d non-finite of %d nodewise pairs from plot."
              % (field_summary["label"], n_tot - n_ok, n_tot))
    tgt_f = target_values[pair_mask]
    pred_f = prediction_values[pair_mask]
    if tgt_f.size == 0:
        print("[WARN] _plot_prediction_target_scatter: no finite target/prediction pairs; skipping scatter.")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{field_summary['label']} Prediction vs Target\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    min_value = float(min(tgt_f.min(), pred_f.min()))
    max_value = float(max(tgt_f.max(), pred_f.max()))

    r2 = r2_prediction_vs_target(tgt_f, pred_f)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(tgt_f, pred_f, s=10, alpha=0.15)
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


def _plot_edge_error_correlation(field_summary, output_path, split_name, num_timesteps):
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
    pair_mask = np.isfinite(error_i) & np.isfinite(error_j)
    n_tot = int(pair_mask.size)
    n_ok = int(pair_mask.sum())
    if n_ok < n_tot:
        print("[WARN] _plot_edge_error_correlation (%s): omitting %d non-finite of %d edge pairs from hexbin."
              % (field_summary["label"], n_tot - n_ok, n_tot))
    ei = error_i[pair_mask]
    ej = error_j[pair_mask]
    if len(ei) == 0 or np.std(ei) == 0.0 or np.std(ej) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(ei, ej)[0, 1])
    r2_lin = r2_linear_fit_x_to_y(ei, ej)

    fig, ax = plt.subplots(figsize=(8, 6))
    if ei.size == 0:
        ax.text(0.5, 0.5, "No finite edge errors", ha="center", va="center", transform=ax.transAxes)
    else:
        hexbin = ax.hexbin(ei, ej, gridsize=60, cmap="viridis", mincnt=1)
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

def _plot_roll_err_pos_vs_time(true_rollout, pred_rollout, output_path, split_name):
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
        diff = torch.abs(true_graph.x_pos_t - pred_graph.x_pos_t)
        if torch.isfinite(diff).all():
            step_mae = torch.mean(diff).item()
        else:
            n_bad = int((~torch.isfinite(diff)).sum().item())
            n_all = int(diff.numel())
            print("[WARN] _plot_roll_err_pos_vs_time: timestep %s has %d non-finite |pos_diff| of %d nodes; "
                  "MAE uses finite entries only." % (len(errors), n_bad, n_all))
            finite = diff[torch.isfinite(diff)]
            step_mae = float(torch.mean(finite).item()) if finite.numel() > 0 else float("nan")
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


def _plot_err_heatmap_rollout(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
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
    print("Printing validation rollout...")
    for field_name, field_label in TARGET_FIELDS:
        for error_mode in ["signed_relative"]:
            output_path = output_dir / f"{filename_prefix}_{field_name}_{error_mode}_error.gif"
            _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path, split_name, len(true_rollout))


def _plot_scatter_edge_corr(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
    """
    Plot
    1. scatter plots: prediction vs target, where every point is one node

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
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)
    num_timesteps = len(true_rollout)
    for field_name, field_label in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"
        field_summary = rollout_summary["fields"][field_name]
        _plot_prediction_target_scatter(field_summary, scatter_path, split_name, num_timesteps)
        # correlation_path = output_dir / f"{filename_prefix}_{field_name}_edge_error_correlation.png"
        # plot_edge_error_correlation(field_summary, correlation_path, split_name, num_timesteps)

def compute_coordinate_bounds(coordinates_per_step):
    """
    Args:
        coordinates_per_step: list
            List of coordinate arrays with shape `(num_nodes, 3)`.
        return: tuple
            `(coordinate_min, coordinate_max, coordinate_span, padding)`.
    """
    if len(coordinates_per_step) == 0:
        return _finite_coordinate_bounds_from_stack(np.zeros((0, 3), dtype=np.float64))
    all_coordinates = np.concatenate(coordinates_per_step, axis=0)
    return _finite_coordinate_bounds_from_stack(all_coordinates)

def in_depth_plots(true_rollout, pred_rollout, output_dir, traj_split_name, filename_prefix,
                   plot_diagnostics_advanced):
    """
    Generates all in-depth diagnostic plots for a single trajectory rollout.
    Scatter and hexbin figures that compare or relate two quantities include an R^2 annotation
    computed with scikit-learn (see plot_rollout / diagnostic_plot_helpers).

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Directory where all plots for this trajectory are saved.
        traj_split_name: str
            Label used in plot titles, e.g. "traj0, test".
        filename_prefix: str
            Prefix for saved filenames, e.g. "traj0_test".
        return: dict
            Nested dict with keys "images" and "videos", each mapping a plot name to its Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths (for _scatter generated inside the function itself)
    gif_path = output_dir / f"{filename_prefix}_beam_comparison.gif"
    volume_png_path = output_dir / f"{filename_prefix}_volume_change.png"
    rollout_error_path = output_dir / f"{filename_prefix}_rollout_error_vs_timestep.png"

    # Plots
    _plot_volume_change(pred_rollout, true_rollout, path=volume_png_path, split_name=traj_split_name)
    _make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout, L=1.0, W=0.1, D=0.04,
                              gif_path=gif_path, fps=4, bbox_only_on_true=False, verbose=False,
                              split_name=traj_split_name)
    _plot_roll_err_pos_vs_time(true_rollout, pred_rollout, rollout_error_path, split_name=traj_split_name)
    _plot_err_heatmap_rollout(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=output_dir,
                              filename_prefix=filename_prefix, split_name=traj_split_name)
    _plot_scatter_edge_corr(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=output_dir,
                            filename_prefix=filename_prefix, split_name=traj_split_name)

    if plot_diagnostics_advanced:
        try:
            plot_mesh_diagnostics(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=output_dir,
                                  filename_prefix=filename_prefix, split_name=traj_split_name)
        except Exception as e:
            print(f"Error while plotting mesh diagnostics: {e}")

    paths = {"images": {}, "videos": {}}
    paths["videos"]["beam_comparison_gif"] = gif_path
    paths["images"]["volume_change"] = volume_png_path
    paths["images"]["rollout_error_vs_timestep"] = rollout_error_path
    for field_name, _ in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"
        if scatter_path.exists():
            paths["images"][f"{field_name}_prediction_vs_target"] = scatter_path
        error_gif_path = output_dir / f"{filename_prefix}_{field_name}_signed_relative_error.gif"
        if error_gif_path.exists():
            paths["videos"][f"{field_name}_signed_relative_error"] = error_gif_path
        if plot_diagnostics_advanced:
            corr_path = output_dir / f"{filename_prefix}_{field_name}_edge_error_correlation.png"
            if corr_path.exists():
                paths["images"][f"{field_name}_edge_error_correlation"] = corr_path

    if plot_diagnostics_advanced:
        diag_dir = output_dir / "diagnostics"
        if diag_dir.exists():
            for field_name, _ in TARGET_FIELDS:
                for suffix in ["neighbor_abs_error", "error_vs_dist_worst",
                               "error_vs_curvature", "error_vs_boundary_dist"]:
                    diag_path = diag_dir / f"{filename_prefix}_{field_name}_{suffix}.png"
                    if diag_path.exists():
                        paths["images"][f"diag/{field_name}_{suffix}"] = diag_path
            dirichlet_path = diag_dir / f"{filename_prefix}_dirichlet_energy.png"
            if dirichlet_path.exists():
                paths["images"]["diag/dirichlet_energy"] = dirichlet_path
            spectral_path = diag_dir / f"{filename_prefix}_spectral_comparison.png"
            if spectral_path.exists():
                paths["images"]["diag/spectral_comparison"] = spectral_path
    return paths