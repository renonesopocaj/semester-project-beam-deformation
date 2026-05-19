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
from src.gnn.plots.plot_diagnostic import (extract_surface_faces, extract_scalar,
                                           plot_mesh_diagnostics)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  
from src.gnn.mesh_numerical_helpers.compute_volume import volume, rollout_total_absolute_volumes  
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import to_numpy_tetra_connectivity, to_numpy  
from src.gnn.misc_helpers.compute_r2 import (format_r2_text, r2_linear_fit_x_to_y,
                                             r2_prediction_vs_target)  
from pathlib import Path  

RELATIVE_ERROR_CLIP = 5.0  # 500 %; caps signed relative error color scale and clipping for mesh error GIFs.
TARGET_FIELDS = [("acc_x", "Acceleration X"),  # Field key + human label for titles/colorbars/scatter.
                 ("acc_y", "Acceleration Y"),  # Same pattern: extract_scalar key + plot label string.
                 ("acc_z", "Acceleration Z"),
                 ("von_mises", "von Mises")]  # von Mises stress: key matches graph attribute naming.

def _finite_coordinate_bounds_from_stack(all_coordinates):
    """
    Compute axis-aligned bounds using only rows with finite XYZ.

    Args:
        all_coordinates: np.ndarray
            Stacked node coordinates, shape (N, 3).
    return: tuple
        (coordinate_min, coordinate_max, coordinate_span, padding) all shape-(3,) arrays.
    """
    mask = np.isfinite(all_coordinates).all(axis=1)  # Rows usable for 3D axis limits (finite XYZ per node).
    n_row = int(all_coordinates.shape[0])  # Total stacked nodes across timesteps (for warn counts).
    n_finite_row = int(mask.sum())  # How many rows contribute to min/max for the plot box.
    if n_finite_row < n_row:
        print("[WARN] _finite_coordinate_bounds_from_stack: %d of %d node rows have non-finite XYZ; "
              "bounds use only finite rows." % (n_row - n_finite_row, n_row))
    if not np.any(mask):
        print("[WARN] _finite_coordinate_bounds_from_stack: no finite coordinates; using fallback bounds.")
        coordinate_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)  # Default cube low corner for empty plot.
        coordinate_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # Default cube high corner for empty plot.
    else:
        finite_coords = all_coordinates[mask]  # Only finite positions fed into axis limit computation.
        coordinate_min = finite_coords.min(axis=0).astype(np.float64)  # Per-axis min for set_xlim/ylim/zlim lower.
        coordinate_max = finite_coords.max(axis=0).astype(np.float64)  # Per-axis max for set_xlim/ylim/zlim upper.
    eps = 1e-10  # Tiny extent so degenerate axis range still draws in 3D.
    for axis in range(3):
        if not np.isfinite(coordinate_min[axis]) or not np.isfinite(coordinate_max[axis]):
            coordinate_min[axis] = -1.0  # Replace bad axis with symmetric default for stable camera box.
            coordinate_max[axis] = 1.0
        elif coordinate_max[axis] <= coordinate_min[axis]:
            coordinate_max[axis] = coordinate_min[axis] + eps  # Enforce positive thickness on each axis span.
    coordinate_span = np.maximum(coordinate_max - coordinate_min, eps)  # Used later for set_box_aspect padding math.
    padding = 0.05 * coordinate_span  # Extra margin around mesh for error-mesh GIF axis limits.
    if not (np.isfinite(coordinate_min).all() and np.isfinite(coordinate_max).all()
            and np.isfinite(coordinate_span).all() and np.isfinite(padding).all()):
        print("[WARN] _finite_coordinate_bounds_from_stack: non-finite min/max/span/padding after fix; "
              "forcing symmetric fallback box.")
        coordinate_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)  # Reset plot box to unit cube low.
        coordinate_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # Reset plot box to unit cube high.
        coordinate_span = np.maximum(coordinate_max - coordinate_min, eps)  # Recompute span for aspect.
        padding = 0.05 * coordinate_span  # Recompute margin for error mesh plots.
    return coordinate_min, coordinate_max, coordinate_span, padding  # Drive 3D lims, aspect, and padded error views.


def _finite_scalar_max_from_arrays(arrays):
    """
    Maximum over finite entries of multiple 1D-array-like objects.

    Args:
        arrays: list
            List of array-like scalars per node (von Mises, etc.).
    return: float
        Strictly positive scale for color normalization; 1.0 if nothing finite.
    """
    max_val = None  # Running global max of finite scalars across all arrays (von Mises etc.).
    n_nonfinite = 0  # Count of NaN/inf samples skipped when building color normalization vmax.
    n_totalElts = 0  # Total scalar samples seen (for warning denominator).
    for arr in arrays:
        flat = np.asarray(arr).reshape(-1)  # 1D view of nodal scalars for one timestep / branch.
        n_totalElts += int(flat.size)  # Accumulate sample count across pred/true stacks.
        n_nonfinite += int((~np.isfinite(flat)).sum())  # Track bad values excluded from vmax.
        finite = flat[np.isfinite(flat)]  # Only finite entries participate in max for colormap stretch.
        if finite.size == 0:
            continue  # Skip this array if it contributes no finite vmax.
        local_max = float(np.max(finite))  # Peak stress (etc.) on this slice for colorbar upper bound.
        if max_val is None or local_max > max_val:
            max_val = local_max  # Keep widest range so Normalize(vmin, vmax) covers all GIF frames.
    if n_nonfinite > 0:
        print("[WARN] _finite_scalar_max_from_arrays: %d non-finite of %d scalar samples ignored for vmax."
              % (n_nonfinite, n_totalElts))
    if max_val is None or not np.isfinite(max_val) or max_val <= 0.0:
        print("[WARN] _finite_scalar_max_from_arrays: no positive finite vmax; using 1.0 for color scale.")
        return 1.0  # Safe default vmax so plt.Normalize and facecolors never divide by zero.
    return max_val  # Upper end of jet colormap for von Mises surface GIFs.


def _plot_volume_change(pred_rollout, true_rollout, path, split_name, traj_metadata):
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
        traj_metadata: dict or None
            Trajectory metadata for a footnote textbox; skipped when None.
    """
    fig, ax = plt.subplots()  # Single 2D axes: x = timestep index, y = total mesh volume scalar.
    true_volumes_list, pred_volumes_list = [], []  # Series accumulated for two ax.plot lines.

    for pred, true in zip(pred_rollout, true_rollout):
        x_true = true.x_pos_t  # Ground-truth node positions at this timestep (for tet volume integrals).
        x_pred = pred.x_pos_t  # Predicted node positions at same timestep (same connectivity).
        # Obtain all cells as tensors of (N, 4) i.e. 4 nodes for everyone of the N tetrahedrons
        cells = torch.as_tensor(to_numpy_tetra_connectivity(true.x_element_connectivity), dtype=torch.long,
                                device=x_true.device)
        true_volumes = volume(x_true, cells)  # Per-tet signed volume from true geometry.
        pred_volumes = volume(x_pred, cells)  # Per-tet signed volume from predicted geometry.
        # sum the absolute value of all volumes
        V_true = torch.abs(true_volumes).sum().detach().cpu().item()  # Scalar total “absolute” mesh volume, true.
        V_pred = torch.abs(pred_volumes).sum().detach().cpu().item()  # Scalar total “absolute” mesh volume, pred.
        if not np.isfinite(V_true) or not np.isfinite(V_pred):
            print("[WARN] _plot_volume_change: timestep non-finite aggregate volume (V_true=%s, V_pred=%s); "
                  "plot may be misleading." % (V_true, V_pred))

        true_volumes_list.append(V_true)  # One y-sample per timestep for blue/orange true curve.
        pred_volumes_list.append(V_pred)  # One y-sample per timestep for predicted curve.

    ax.plot(true_volumes_list, label="True Volumes")  # Line plot: x implicit 0..T-1, y = true total volume.
    ax.plot(pred_volumes_list, label="Predicted Volumes")  # Second line on same axes for comparison.
    plt.xlabel("Time Step")  # Horizontal axis = rollout index (0-based integer steps).
    plt.ylabel("Volume")  # Vertical axis = summed absolute tet volumes (model units).
    ax.set_title(f"Volume Change\n[{split_name}, 1 sample, {len(pred_rollout)} timesteps]")  # Title encodes split + T.
    ax.legend()  # Distinguish true vs predicted traces.
    if traj_metadata is not None:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})
    plt.savefig(path)  # Write static PNG to validation folder.
    plt.close()  # Release figure backend memory.

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
        return poly_collection  # No boundary triangles to draw; leave axes unchanged.
    valid_nodes = np.isfinite(coordinates).all(axis=1) & np.isfinite(node_values)  # Mask nodes safe for indexing/color.
    n_bad_nodes = int((~valid_nodes).sum())  # Count dropped nodes (warn once on first draw).
    if n_bad_nodes > 0 and poly_collection is None:
        print("[WARN] update_surface_collection (first draw): %d nodes have non-finite position or nodal scalar; "
              "boundary faces incident on them are dropped in this view." % n_bad_nodes)
    use_faces = faces[valid_nodes[faces].all(axis=1)]  # Keep only triangles whose three corner nodes are valid.
    if use_faces.size < faces.size and poly_collection is None:
        print("[WARN] update_surface_collection (first draw): using %d of %d boundary faces (others touch omitted nodes)."
              % (use_faces.size, faces.size))
    if use_faces.size == 0:
        print("[WARN] update_surface_collection: no faces left after dropping non-finite nodes; frame may be empty.")
        if poly_collection is not None:
            try:
                poly_collection.set_verts(np.zeros((0, 3, 3), dtype=np.float64))  # Empty mesh: clear prior frame triangles.
            except Exception:
                pass
        return poly_collection

    polygons = coordinates[use_faces]  # Shape (F,3,3): each row is one triangle’s 3 vertex XYZ tuples for Poly3D.
    face_values = np.mean(node_values[use_faces], axis=1)  # Per-triangle scalar = mean of its 3 nodal values (color).
    finite_face = np.isfinite(face_values)  # Mask faces whose mean scalar is plottable.
    if not np.all(finite_face):
        n_bad_face = int((~finite_face).sum())
        print("[WARN] update_surface_collection: %d faces have non-finite mean nodal value; "
              "replacing with 0.0 for coloring." % n_bad_face)
        face_values = np.where(finite_face, face_values, 0.0)  # Map bad means to colormap vmin (via norm) safely.
    face_colors = cmap(norm(face_values))  # RGBA row per face: scalar → normalized → colormap lookup.

    if poly_collection is None:
        poly_collection = Poly3DCollection(polygons)  # First frame: allocate 3D triangle artist on ax.
        poly_collection.set_facecolors(face_colors)  # Paint each triangle face from per-face RGBA.
        poly_collection.set_edgecolor("none")  # Hide triangle edges for smooth shaded surface look.
        ax.add_collection3d(poly_collection)  # Register artist with mplot3d so it renders in 3D scene.
        return poly_collection

    poly_collection.set_verts(polygons)  # Animation path: update vertex positions without recreating artist.
    poly_collection.set_facecolors(face_colors)  # Refresh colors after verts (e.g. stress or error field changed).
    return poly_collection  # Same object reused next GIF frame.

def _make_beam_comparison_gif(pred_rollout, true_rollout, L, W, D, gif_path, fps, bbox_only_on_true, verbose, split_name,
                              traj_metadata):
    """
    Create side-by-side comparison GIF of predicted vs ground truth.
    Each frame shows the timestep index as t = i / last (0-based).

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
        split_name: str
            Split label for titles.
        traj_metadata: dict or None
            Trajectory metadata for a static footnote textbox; skipped when None.

    """
    assert len(pred_rollout) == len(true_rollout), ("Mismatch in predicted and ground truth frames.")

    frames = []  # List of H×W×3 uint8 RGB images passed to imageio for GIF encoding.

    faces = extract_surface_faces(true_rollout[0].x_element_connectivity)  # Fixed surface triangulation for all frames.
    coordinates_for_bounds = []  # Stack of XYZ arrays used once to fix shared camera box for both panels.
    vm_arrays = []  # All von Mises samples across time/branches to set one global colorbar vmax.

    for g_pred, g_true in zip(pred_rollout, true_rollout):
        coords_pred = to_numpy(g_pred.x_pos_t)  # Predicted node XYZ for bounds (optional) and per-frame draw.
        coords_true = to_numpy(g_true.x_pos_t)  # True node XYZ for bounds and left panel mesh.
        von_mises_pred = to_numpy(g_pred.von_mises).reshape(-1)  # Nodal stress for right panel coloring.
        von_mises_true = to_numpy(g_true.von_mises).reshape(-1)  # Nodal stress for left panel coloring.

        vm_arrays.append(von_mises_pred)  # Extend global max search to predicted stress.
        vm_arrays.append(von_mises_true)  # Extend global max search to true stress.
        coordinates_for_bounds.append(coords_true)  # Always include true geometry in axis limit union.

        if bbox_only_on_true:
            continue  # Skip pred coords for limits: tighter box following GT only.
        coordinates_for_bounds.append(coords_pred)  # Include pred coords so both meshes stay in view.

    coordinate_min, coordinate_max, _, _ = compute_coordinate_bounds(coordinates_for_bounds)  # Shared 3D xlim/ylim/zlim.
    vm_min = 0.0  # Colorbar starts at zero stress (jet scale anchored at 0).
    vm_max = _finite_scalar_max_from_arrays(vm_arrays)  # Colorbar top = max finite von Mises over rollout.

    # Create figure
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"})  # Side-by-side 3D subplots.

    cmap = plt.cm.jet  # Pseudo-color map for von Mises on both meshes (same mapping).
    norm = plt.Normalize(vmin=vm_min, vmax=vm_max)  # Shared normalization: identical colors mean identical stress MPa-scale.

    # Create colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Tall narrow axes at right for vertical colorbar strip.
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)  # Draw scalar→color legend for fig.
    cbar.set_label("von Mises Stress")  # Colorbar axis title explaining mapped quantity.
    fig.suptitle(f"Beam Comparison\n[{split_name}, 1 sample, {len(pred_rollout)} timesteps]", fontsize=13)  # Overall title.

    num_steps = len(pred_rollout)  # Number of animation frames equals rollout length.
    last_timestep = num_steps - 1  # Denominator for timestep overlay text (0-based last index).
    if traj_metadata is not None:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})
        timestep_y = 0.09
    else:
        timestep_y = 0.03
    timestep_label = fig.text(
        0.5, timestep_y, "", ha="center", va="bottom", fontsize=12, fontweight="bold")  # Bottom-center footer updated per frame.

    polycoll_true = None  # Lazily created Poly3DCollection for left subplot; reused across frames.
    polycoll_pred = None  # Same for right subplot predicted mesh.

    # Configure axes
    for ax, title in zip(axarr, ["Ground Truth", "Predicted MeshGraphNet"]):
        ax.set_xlim(coordinate_min[0], coordinate_max[0])  # Fix X extent so mesh does not jump scale between frames.
        ax.set_ylim(coordinate_min[1], coordinate_max[1])  # Fix Y extent (shared both panels).
        ax.set_zlim(coordinate_min[2], coordinate_max[2])  # Fix Z extent (shared both panels).
        ax.set_xlabel("X")  # Axis label for 3D plot readability.
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((L * 0.1, 0.8 * W, 2.5 * D))  # Physical-ish aspect ratios so beam shape looks elongated.
        ax.view_init(elev=30, azim=45)  # Fixed camera angle for entire GIF.
        ax.set_title(title)  # Distinguish GT vs prediction panel.

    # Iterate over time steps
    for i, (graph_pred, graph_true) in enumerate(zip(pred_rollout, true_rollout)):
        if verbose:
            print(f"Rendering frame {i}/{len(pred_rollout) - 1}")

        coords_pred = to_numpy(graph_pred.x_pos_t)  # Current predicted vertex positions for right mesh.
        coords_true = to_numpy(graph_true.x_pos_t)  # Current true vertex positions for left mesh.
        vm_pred = to_numpy(graph_pred.von_mises).reshape(-1)  # Per-node stress for right face colors.
        vm_true = to_numpy(graph_true.von_mises).reshape(-1)  # Per-node stress for left face colors.

        # Update both subplots
        polycoll_true = update_surface_collection(axarr[0], faces, coords_true, vm_true, cmap, norm, polycoll_true)
        polycoll_pred = update_surface_collection(axarr[1], faces, coords_pred, vm_pred, cmap, norm, polycoll_pred)

        timestep_label.set_text("t = %d / %d" % (i, last_timestep))  # Show rollout progress on figure.

        # Render to numpy array
        fig.canvas.draw()  # Rasterize current figure to backend buffer (needed before buffer_rgba).
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # RGBA→RGB uint8 snapshot of whole twin plot.
        frames.append(frame)  # Stack frame for GIF writer.

    plt.close(fig)  # Tear down figure after capturing all frames.

    # Create GIF
    if len(frames) == 0:
        print(f"[WARN] _make_beam_comparison_gif: no frames; skipping {gif_path}.")
        return
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)  # Write animated GIF; loop=0 means repeat forever in viewers.
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
    fx = _format_metadata_scalar(traj_metadata.get("fx", "?"))  # Load/format force x for subtitle string under 3D plot.
    fy = _format_metadata_scalar(traj_metadata.get("fy", "?"))  # Load/format force y component text.
    fz = _format_metadata_scalar(traj_metadata.get("fz", "?"))  # Load/format force z component text.
    beam_L = _format_metadata_scalar(traj_metadata.get("L", "?"))  # Load/format beam length label in subtitle.
    beam_W = _format_metadata_scalar(traj_metadata.get("W", "?"))  # Load/format beam width label in subtitle.
    beam_D = _format_metadata_scalar(traj_metadata.get("D", "?"))  # Load/format beam depth/thickness label in subtitle.
    ood = traj_metadata.get("ood_features", [])  # List of OOD tags to summarize on figure.
    ood_str = ", ".join(ood) if ood else "none"  # Comma-separated OOD list or literal “none” for display.
    return (f"fx={fx}  fy={fy}  fz={fz}  |  L={beam_L}  W={beam_W}  D={beam_D}"
            f"  |  OOD: {ood_str}")  # One string drawn with fig.text (not axis title).


def _format_metadata_scalar(value):
    """
    Format metadata scalars shown in plot footnote textboxes.

    Args:
        value: Any
            Scalar-like metadata value (number/string/placeholder).
    return: str
        Readable scalar string with bounded float precision.
    """
    if isinstance(value, bool):
        return str(value)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(scalar):
        return str(value)
    formatted = f"{scalar:.4f}".rstrip("0").rstrip(".")
    if formatted in {"", "-0"}:
        return "0"
    return formatted


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
    frames = []  # RGB frames for single-panel ground-truth animation.
    faces = extract_surface_faces(trajectory_graphs[0].x_element_connectivity)  # Boundary faces from first graph (fixed topo).

    coordinates_for_bounds = []  # Collect all timesteps’ coords to set fixed axis limits for stable GIF.
    vm_arrays = []  # All von Mises values to set one colorbar range across frames.
    for graph in trajectory_graphs:
        coords = to_numpy(graph.x_pos_t)  # Node positions at this timestep for bounds + drawing.
        vm = to_numpy(graph.von_mises).reshape(-1)  # Nodal stress samples for vmax and coloring.
        vm_arrays.append(vm)  # Feed into global finite max for Normalize vmax.
        coordinates_for_bounds.append(coords)  # Feed into compute_coordinate_bounds union.

    coordinate_min, coordinate_max, _, _ = compute_coordinate_bounds(coordinates_for_bounds)  # Static xlim/ylim/zlim.
    vm_min = 0.0  # Stress colormap lower bound.
    vm_max = _finite_scalar_max_from_arrays(vm_arrays)  # Upper stress bound for jet mapping.

    fig = plt.figure(figsize=(10, 8))  # Taller canvas for single 3D view + subtitle margin.
    ax = fig.add_subplot(111, projection="3d")  # One 3D axes occupying the figure.

    cmap = plt.cm.jet  # Colormap for von Mises surface.
    norm = plt.Normalize(vmin=vm_min, vmax=vm_max)  # Maps scalar stress to [0,1] before cmap.

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.1)  # Colorbar beside 3D ax.
    cbar.set_label("von Mises Stress")  # Explains color strip units/quantity.

    ax.set_xlim(coordinate_min[0], coordinate_max[0])  # Freeze X range for animation stability.
    ax.set_ylim(coordinate_min[1], coordinate_max[1])  # Freeze Y range.
    ax.set_zlim(coordinate_min[2], coordinate_max[2])  # Freeze Z range.
    ax.set_xlabel("X")  # Axis labels on 3D plot.
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((L * 0.1, 0.8 * W, 2.5 * D))  # Match beam proportions visually.
    ax.view_init(elev=30, azim=45)  # Fixed viewpoint for all frames.

    # Add static metadata subtitle below the 3D plot.
    if traj_metadata is not None:
        subtitle = _format_traj_metadata_text(traj_metadata)  # Build one-line param summary for fig.text.
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})  # Wheat box under plot area.

    poly_collection = None  # Reused Poly3DCollection updated each timestep.
    for i, graph in enumerate(trajectory_graphs):
        coords = to_numpy(graph.x_pos_t)  # Vertices for current animation frame.
        vm = to_numpy(graph.von_mises).reshape(-1)  # Per-node colors input (averaged per face inside helper).
        poly_collection = update_surface_collection(ax, faces, coords, vm, cmap, norm, poly_collection)  # Redraw surface.
        ax.set_title(f"Ground Truth, step {i}\n[{split_name}, traj {traj_idx}, {len(trajectory_graphs)} timesteps]")  # Dynamic title.

        fig.canvas.draw()  # Flush render pipeline before grabbing pixels.
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())  # Snapshot RGB for GIF.

    plt.close(fig)  # Free resources after frame list built.
    if len(frames) == 0:
        print(f"[WARN] make_ground_truth_gif: no frames; skipping {gif_path}.")
        return
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)  # Encode GT-only stress animation.
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
    signed_error = pred_values - true_values  # Raw residual pred−target at each node (for derived error maps).
    if error_mode == "absolute":
        return np.abs(signed_error)  # Nodal |error| for absolute-error heatmaps (not used in current GIF path).
    if error_mode == "relative":
        denominator = np.maximum(np.abs(true_values), 1e-12)  # Avoid divide-by-zero in relative error.
        return np.abs(signed_error) / denominator  # Nodal |rel error| capped denominator.
    if error_mode == "signed_relative":
        denominator = np.maximum(np.abs(true_values), 1e-12)  # Same safe denominator for signed % error.
        return signed_error / denominator  # Signed fractional error → later shown as % on colorbar.
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
    flat = np.asarray(all_values).reshape(-1)  # All nodal errors concatenated over time for vmin/vmax stats.
    finite = flat[np.isfinite(flat)]  # Ignore NaN/inf when setting color scale limits.
    if finite.size == 0:
        print("[WARN] _compute_limits: no finite error values; using tiny symmetric range.")
        tiny = 1e-12
        if error_mode == "signed_relative":
            return -tiny, tiny  # Symmetric diverging colormap limits around zero when no data.
        return 0.0, tiny  # One-sided scale floor for absolute/relative modes.

    if error_mode == "signed_relative":
        max_abs = min(float(np.max(np.abs(finite))), RELATIVE_ERROR_CLIP)  # Clip symmetric limit at ±500% (5.0 fraction).
        if max_abs == 0.0:
            max_abs = 1e-12  # Avoid vmin==vmax degeneracy in Normalize.
        return -max_abs, max_abs  # coolwarm center at 0, saturated at ±clip for mesh error GIF.

    vmax = float(np.max(finite))  # Upper positive limit for non-signed error modes.
    if error_mode == "relative":
        vmax = min(vmax, RELATIVE_ERROR_CLIP)  # Cap relative error color scale same as signed clip magnitude.
    if vmax == 0.0:
        vmax = 1e-12  # Prevent zero-width normalization.
    return 0.0, vmax  # inferno-style lower bound at 0 for positive-only error display.


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
    coordinates_per_step = []  # List of (N,3) arrays: true mesh geometry per timestep for 3D error rendering.
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        coordinates_per_step.append(to_numpy(true_graph.x_pos_t))  # True positions drive surface vertex locations in GIFs.

    coordinate_min, coordinate_max, coordinate_span, padding = compute_coordinate_bounds(coordinates_per_step)  # Shared 3D box + pad.
    summary = {"coordinates_per_step": coordinates_per_step,  # Per-step coords consumed by error mesh animator.
               "faces": extract_surface_faces(true_rollout[0].x_element_connectivity),  # Fixed boundary tris for all error frames.
               "coordinate_min": coordinate_min,  # Cached for axis limits in error GIF.
               "coordinate_max": coordinate_max,
               "coordinate_span": coordinate_span,  # Used for set_box_aspect on error mesh plot.
               "padding": padding,  # Margin added to lims so mesh does not touch axes.
               "fields": {}}  # Per-physical-field bundles for scatter, hexbin, GIF.

    for field_name, field_label in TARGET_FIELDS:
        target_per_step = []  # Per-timestep nodal target scalars for this field (acc component or von Mises).
        prediction_per_step = []  # Matching predictions for scatter / error computation.
        residual_errors_per_step = []  # pred−target per node per step (edge correlation uses these).
        absolute_errors_per_step = []  # |residual| per node per step (limits precompute).
        signed_relative_errors_per_step = []  # (pred−target)/|target| per node per step for signed error mesh GIF.
        edge_error_i = []  # Residual at edge source nodes (flattened later for hexbin x-axis).
        edge_error_j = []  # Residual at edge target nodes (hexbin y-axis).

        for true_graph, pred_graph in zip(true_rollout, pred_rollout):
            target_values = extract_scalar(true_graph, field_name)  # Ground-truth field on nodes at this step.
            prediction_values = extract_scalar(pred_graph, field_name)  # Model output on same nodes.
            residual_errors = prediction_values - target_values  # Nodal residual for this step.
            absolute_errors = np.abs(residual_errors)  # Magnitude time series for limits dict.
            signed_relative_errors = _compute_error(target_values, prediction_values, "signed_relative")  # Fractional signed error.
            edge_index = to_numpy(true_graph.edge_index).astype(np.int64)  # Graph edges: rows source, target node ids.

            target_per_step.append(target_values)  # Stack for global scatter x-axis pool.
            prediction_per_step.append(prediction_values)  # Stack for global scatter y-axis pool.
            residual_errors_per_step.append(residual_errors)  # Kept for potential extended diagnostics.
            absolute_errors_per_step.append(absolute_errors)  # Feeds absolute error color limits if enabled.
            signed_relative_errors_per_step.append(signed_relative_errors)  # Per-step arrays animated on mesh surface.
            edge_error_i.append(residual_errors[edge_index[0]])  # Sample residual at each edge’s tail node.
            edge_error_j.append(residual_errors[edge_index[1]])  # Sample residual at each edge’s head node.

        errors_by_mode = {"absolute": absolute_errors_per_step,  # Bundle keyed by how we might color errors.
                          "signed_relative": signed_relative_errors_per_step    }
        summary["fields"][field_name] = {
            "label": field_label,  # Human-readable name in plot titles/colorbars.
            "target_values": np.concatenate(target_per_step),  # All timesteps, all nodes: scatter “Target” axis.
            "prediction_values": np.concatenate(prediction_per_step),  # Scatter “Prediction” axis.
            "edge_error_i": np.concatenate(edge_error_i),  # Hexbin horizontal coordinates.
            "edge_error_j": np.concatenate(edge_error_j),  # Hexbin vertical coordinates.
            "errors_by_mode": errors_by_mode,  # Per-mode time stacks passed to GIF renderer.
            "limits_by_mode": {error_mode: _compute_limits(np.concatenate(error_values), error_mode)
                               for error_mode, error_values in errors_by_mode.items()}}  # Precomputed vmin/vmax per mode.

    return summary  # Single structure reused to avoid recomputing concat per plot type.


def _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path, split_name, num_timesteps, traj_metadata):
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
        split_name: str
            Split label for titles.
        num_timesteps: int
            Rollout length for titles.
        traj_metadata: dict or None
            Trajectory metadata for a static footnote textbox; skipped when None.
        return: None
            Saves the animated error mesh to disk.
    """
    field_summary = rollout_summary["fields"][field_name]  # Pre-baked arrays/limits for this scalar field.
    field_label = field_summary["label"]  # Title/colorbar text (e.g. “Acceleration X”).
    coordinates_per_step = rollout_summary["coordinates_per_step"]  # True mesh XYZ per timestep (surface moves).
    error_per_step = field_summary["errors_by_mode"][error_mode]  # Nodal error time series to paint on surface.
    faces = rollout_summary["faces"]  # Boundary triangle indices (fixed).
    coordinate_min = rollout_summary["coordinate_min"]  # Base axis limits before padding.
    coordinate_max = rollout_summary["coordinate_max"]
    coordinate_span = rollout_summary["coordinate_span"]  # Diagonal extent for aspect tuple.
    padding = rollout_summary["padding"]  # Per-axis margin added outside min/max for framing.
    vmin, vmax = field_summary["limits_by_mode"][error_mode]  # Symmetric ± for signed_relative (clipped globally).
    cmap = plt.cm.coolwarm if error_mode == "signed_relative" else plt.cm.inferno  # Diverging blue–white–red for ± error.
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # Maps clipped error fractions to [0,1] for cmap.

    fig = plt.figure(figsize=(8, 6))  # Canvas for one 3D error mesh view.
    ax = fig.add_subplot(111, projection="3d")  # 3D axes for colored surface.
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.12)  # Vertical legend.
    if error_mode != "signed_relative":
        raise ValueError(f"Unsupported rollout heatmap mode: {error_mode}")
    colorbar.set_label(f"{field_label} signed relative error [%, clipped at \u00b1500%]")  # Describes mapped quantity + clip.
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:+.0f}%"))  # Tick labels as signed whole percents.
    ax.set_xlim(coordinate_min[0] - padding[0], coordinate_max[0] + padding[0])  # Padded X extent so mesh floats inside box.
    ax.set_ylim(coordinate_min[1] - padding[1], coordinate_max[1] + padding[1])  # Padded Y extent.
    ax.set_zlim(coordinate_min[2] - padding[2], coordinate_max[2] + padding[2])  # Padded Z extent.
    ax.set_xlabel("X")  # Axis labels for orientation.
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(tuple(coordinate_span + padding))  # Physical proportions from span+pad (matches padded lims).
    ax.view_init(elev=30, azim=45)  # Fixed camera for GIF coherence.
    if traj_metadata is not None:
        fig.subplots_adjust(bottom=0.14)
        fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})

    poly_collection = None  # Mutable 3D surface artist updated each timestep.
    frames = []  # RGB buffers for GIF writer.
    n_err_sanitized = 0  # Count NaN/inf nodes replaced before clipping (warn at end).
    for timestep, (coordinates, error_values) in enumerate(zip(coordinates_per_step, error_per_step)):
        err = np.asarray(error_values, dtype=np.float64)  # Ensure float64 for stable clip/norm.
        n_bad = int((~np.isfinite(err)).sum())  # How many bad nodes this frame.
        if n_bad > 0:
            n_err_sanitized += n_bad  # Accumulate for summary warning.
        err = np.where(np.isfinite(err), err, 0.0)  # Replace non-finite with 0 before norm (maps near center of coolwarm).
        error_values = np.clip(err, vmin, vmax)  # Clamp to ±RELATIVE_ERROR_CLIP so colors don’t saturate past ±500%.
        poly_collection = update_surface_collection(ax, faces, coordinates, error_values, cmap, norm, poly_collection)  # Color surface.
        ax.set_title(f"{field_label} Signed Relative Error [%, clipped \u00b1500%], step {timestep}\n[{split_name}, "
                     f"1 sample, {num_timesteps} timesteps]")  # Frame title with timestep and split context.
        fig.canvas.draw()  # Render before pixel readback.
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())  # Capture RGB frame.

    plt.close(fig)  # Dispose figure after frames collected.
    if n_err_sanitized > 0:
        print("[WARN] _render_error_mesh_gif (%s): replaced %d non-finite nodal error values with 0.0 before clip."
              % (field_name, n_err_sanitized))
    if len(frames) == 0:
        print(f"[WARN] _render_error_mesh_gif: no frames for {output_path}; skipping GIF.")
        return
    imageio.mimsave(output_path, frames, fps=4, loop=0)  # Save error-on-mesh animation at fixed 4 fps.


def _plot_prediction_target_scatter(field_summary, output_path, split_name, num_timesteps, traj_metadata):
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
        traj_metadata: dict or None
            Trajectory metadata for a footnote textbox; skipped when None.
        return: None
            Saves the scatter plot to disk with sklearn.metrics.r2_score(target, prediction) annotated.
    """
    target_values = field_summary["target_values"]  # All nodal targets concatenated over rollout (x-axis candidates).
    prediction_values = field_summary["prediction_values"]  # Matching predictions (y-axis candidates).
    pair_mask = np.isfinite(target_values) & np.isfinite(prediction_values)  # Keep only finite (x,y) scatter points.
    n_tot = int(pair_mask.size)  # Total nodal pairs before filtering.
    n_ok = int(pair_mask.sum())  # Count of drawable points.
    if n_ok < n_tot:
        print("[WARN] _plot_prediction_target_scatter (%s): omitting %d non-finite of %d nodewise pairs from plot."
              % (field_summary["label"], n_tot - n_ok, n_tot))
    tgt_f = target_values[pair_mask]  # Cleaned target coordinates for scatter.
    pred_f = prediction_values[pair_mask]  # Cleaned prediction coordinates for scatter.
    if tgt_f.size == 0:
        print("[WARN] _plot_prediction_target_scatter: no finite target/prediction pairs; skipping scatter.")
        fig, ax = plt.subplots(figsize=(7, 7))  # Empty square figure for placeholder message.
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)  # Centered notice in axes coords.
        ax.set_title(f"{field_summary['label']} Prediction vs Target\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
        if traj_metadata is not None:
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                     bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})
        fig.tight_layout()  # Margins for title/text.
        fig.savefig(output_path)  # Still emit a PNG so downstream path.exists checks succeed.
        plt.close(fig)
        return
    min_value = float(min(tgt_f.min(), pred_f.min()))  # Lower end of y=x reference line span.
    max_value = float(max(tgt_f.max(), pred_f.max()))  # Upper end of perfect-prediction diagonal.

    r2 = r2_prediction_vs_target(tgt_f, pred_f)  # Scalar R² for annotation box (sklearn-backed helper).
    fig, ax = plt.subplots(figsize=(7, 7))  # Square axes so x and y scales visually comparable.
    ax.scatter(tgt_f, pred_f, s=10, alpha=0.15)  # Each point = one node across all timesteps; light transparency for density.
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.5)  # Identity reference line.
    ax.set_xlabel("Target")  # Horizontal axis = ground truth scalar.
    ax.set_ylabel("Prediction")  # Vertical axis = model output.
    ax.set_title(f"{field_summary['label']} Prediction vs Target\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.text(0.03, 0.97, format_r2_text(r2), transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))  # Upper-left R² label in axes-relative coords.
    ax.grid(True, alpha=0.3)  # Light grid behind dense scatter for readability.
    if traj_metadata is not None:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})
    fig.tight_layout()  # Fit subplots within figure before save.
    fig.savefig(output_path)  # Write PNG scatter diagnostic.
    plt.close(fig)


SIG6_COMPONENT_LABELS = ("sxx", "syy", "szz", "sxy", "sxz", "syz")


def _plot_fullstress_prediction_vs_target(true_rollout, pred_rollout, output_path, split_name,
                                          num_timesteps, traj_metadata):
    """
    Single 2x3 figure: one prediction-vs-target scatter per stress component on the 6-component packed
    stress tensor saved at rollout time (only available in target_mode = full_stress). Each subplot mirrors
    `_plot_prediction_target_scatter` (identity reference line, R^2 annotation, finite-pair masking).

    Args:
        true_rollout: list
            Ground-truth rollout graphs; each must have `sig6` (N, 6) attached (see do_rollout).
        pred_rollout: list
            Predicted rollout graphs; each must have `sig6` (N, 6) attached.
        output_path: Path
            Destination PNG path.
        split_name: str
            Split label for the title.
        num_timesteps: int
            Number of rollout timesteps for the title.
        traj_metadata: dict or None
            Optional metadata footnote; skipped when None.
    return: None
        Saves the figure to disk; does nothing if `sig6` is not available on any rollout graph.
    """
    if len(true_rollout) == 0 or len(pred_rollout) == 0:
        return
    if not (hasattr(true_rollout[0], "sig6") and hasattr(pred_rollout[0], "sig6")):
        return
    target_per_step, prediction_per_step = [], []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        target_per_step.append(to_numpy(true_graph.sig6))
        prediction_per_step.append(to_numpy(pred_graph.sig6))
    target_all = np.concatenate(target_per_step, axis=0)  # (T*N, 6)
    prediction_all = np.concatenate(prediction_per_step, axis=0)  # (T*N, 6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for component_idx, label in enumerate(SIG6_COMPONENT_LABELS):
        ax = axes.flat[component_idx]
        target_values = target_all[:, component_idx]
        prediction_values = prediction_all[:, component_idx]
        pair_mask = np.isfinite(target_values) & np.isfinite(prediction_values)
        target_finite = target_values[pair_mask]
        prediction_finite = prediction_values[pair_mask]
        if target_finite.size == 0:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label} (no data)")
            continue
        min_value = float(min(target_finite.min(), prediction_finite.min()))
        max_value = float(max(target_finite.max(), prediction_finite.max()))
        r2 = r2_prediction_vs_target(target_finite, prediction_finite)
        ax.scatter(target_finite, prediction_finite, s=10, alpha=0.15)
        ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.5)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.set_title(label)
        ax.text(0.03, 0.97, format_r2_text(r2), transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Full Stress Prediction vs Target\n[{split_name}, 1 sample, {num_timesteps} timesteps]",
                 fontsize=14)
    if traj_metadata is not None:
        fig.subplots_adjust(bottom=0.12)
        fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})
    fig.tight_layout(rect=(0, 0.04 if traj_metadata is not None else 0.0, 1, 0.96))
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
    error_i = field_summary["edge_error_i"]  # Residual at tail node of each directed edge (flattened rollout).
    error_j = field_summary["edge_error_j"]  # Residual at head node of each directed edge.
    pair_mask = np.isfinite(error_i) & np.isfinite(error_j)  # Valid (x,y) pairs for hexbin density.
    n_tot = int(pair_mask.size)  # Total edge samples before drop.
    n_ok = int(pair_mask.sum())  # Edges used in density plot.
    if n_ok < n_tot:
        print("[WARN] _plot_edge_error_correlation (%s): omitting %d non-finite of %d edge pairs from hexbin."
              % (field_summary["label"], n_tot - n_ok, n_tot))
    ei = error_i[pair_mask]  # x coordinates for hexbin = error at source endpoint.
    ej = error_j[pair_mask]  # y coordinates for hexbin = error at sink endpoint.
    if len(ei) == 0 or np.std(ei) == 0.0 or np.std(ej) == 0.0:
        correlation = np.nan  # Pearson undefined → title shows nan.
    else:
        correlation = float(np.corrcoef(ei, ej)[0, 1])  # Linear correlation for title annotation.
    r2_lin = r2_linear_fit_x_to_y(ei, ej)  # R² of regressing error_j on error_i for text box.

    fig, ax = plt.subplots(figsize=(8, 6))  # Landscape canvas for hexagonal 2D histogram.
    if ei.size == 0:
        ax.text(0.5, 0.5, "No finite edge errors", ha="center", va="center", transform=ax.transAxes)  # Empty-state message.
    else:
        hexbin = ax.hexbin(ei, ej, gridsize=60, cmap="viridis", mincnt=1)  # 60×60 hex grid; drop empty cells; color = count.
        colorbar = fig.colorbar(hexbin, ax=ax)  # Legend mapping viridis intensity to number of edges per bin.
        colorbar.set_label("Edge count")  # Colorbar axis title.
    ax.set_xlabel("error_i")  # Horizontal: residual on edge source nodes.
    ax.set_ylabel("error_j")  # Vertical: residual on edge target nodes.
    ax.set_title(
        f"{field_summary['label']} Edge Error Correlation (corr={correlation:.4f})\n"
        f"[{split_name}, 1 sample, {num_timesteps} timesteps]")  # Two-line title with Pearson in first line.
    ax.text(0.03, 0.97, format_r2_text(r2_lin), transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))  # OLS R² annotation box.
    ax.grid(True, alpha=0.2)  # Faint Cartesian grid under hex cells.
    fig.tight_layout()  # Adjust margins before rasterizing.
    fig.savefig(output_path)  # Persist hexbin PNG.
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
    # print("Plot rollout position error vs timestep...")
    errors = []  # One scalar MAE per rollout step for the y-axis series.
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        diff = torch.abs(true_graph.x_pos_t - pred_graph.x_pos_t)  # Per-node L1 position error at this timestep.
        if torch.isfinite(diff).all():
            step_mae = torch.mean(diff).item()  # Mean over all nodes and coords → single curve height.
        else:
            n_bad = int((~torch.isfinite(diff)).sum().item())
            n_all = int(diff.numel())
            print("[WARN] _plot_roll_err_pos_vs_time: timestep %s has %d non-finite |pos_diff| of %d nodes; "
                  "MAE uses finite entries only." % (len(errors), n_bad, n_all))
            finite = diff[torch.isfinite(diff)]  # Mask out NaN/inf before averaging.
            step_mae = float(torch.mean(finite).item()) if finite.numel() > 0 else float("nan")  # Robust per-step MAE.
        errors.append(step_mae)  # Append point to time series.

    timesteps = np.arange(len(errors))  # x-coordinates 0..T-1 for line plot.
    fig, ax = plt.subplots(figsize=(10, 6))  # Wide figure for readable timestep axis.
    ax.plot(timesteps, errors, linewidth=2.0, marker="o", markersize=3)  # Line + small circles at each step.
    ax.set_xlabel("Timestep")  # Horizontal: rollout index.
    ax.set_ylabel("Position MAE")  # Vertical: mean absolute nodal position error (tensor space units).
    ax.set_title(f"Rollout Position Error vs Timestep\n[{split_name}, 1 sample, {len(errors)} timesteps]")
    ax.grid(True, alpha=0.3)  # Light grid to read MAE values.
    fig.tight_layout()  # Fit layout before save.
    fig.savefig(output_path)  # Write PNG curve.
    plt.close(fig)


def _position_error_series(true_rollout, pred_rollout):
    """
    Computes position MAE at each rollout timestep.

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
    return: list[float]
        One absolute position MAE value per timestep.
    """
    errors = []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        diff = torch.abs(true_graph.x_pos_t - pred_graph.x_pos_t)
        finite = diff[torch.isfinite(diff)]
        if finite.numel() == 0:
            errors.append(float("nan"))
        else:
            errors.append(float(torch.mean(finite).detach().cpu().item()))
    return errors


def _plot_rollout_pos_volume_error_vs_time(true_rollout, pred_rollout, output_path, split_name, traj_metadata):
    """
    Plot position MAE and absolute volume error at each rollout timestep.

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_path: Path
            Destination image path.
        split_name: str
            Split and trajectory label used in the title.
        traj_metadata: dict or None
            Trajectory metadata for a footnote textbox; skipped when None.
    """
    print("Plot rollout position and volume error vs timestep...")
    position_errors = _position_error_series(true_rollout, pred_rollout)
    true_volumes = rollout_total_absolute_volumes(true_rollout)
    pred_volumes = rollout_total_absolute_volumes(pred_rollout)
    n_steps = min(len(position_errors), len(true_volumes), len(pred_volumes))
    timesteps = np.arange(n_steps)
    volume_errors = []
    for step_idx in range(n_steps):
        volume_errors.append(abs(pred_volumes[step_idx] - true_volumes[step_idx]))

    fig, ax_pos = plt.subplots(figsize=(10, 6))
    pos_line = ax_pos.plot(timesteps, position_errors[:n_steps], color="tab:blue", linewidth=2.0,
                           marker="o", markersize=3, label="Position MAE")
    ax_pos.set_xlabel("Timestep")
    ax_pos.set_ylabel("Position MAE", color="tab:blue")
    ax_pos.tick_params(axis="y", labelcolor="tab:blue")
    ax_pos.grid(True, alpha=0.3)

    ax_vol = ax_pos.twinx()
    vol_line = ax_vol.plot(timesteps, volume_errors, color="tab:orange", linewidth=2.0,
                           marker="s", markersize=3, label="Absolute Volume Error")
    ax_vol.set_ylabel("Absolute Volume Error", color="tab:orange")
    ax_vol.tick_params(axis="y", labelcolor="tab:orange")

    lines = pos_line + vol_line
    labels = [line.get_label() for line in lines]
    ax_pos.legend(lines, labels, loc="best")
    ax_pos.set_title(f"Rollout Position and Volume Error vs Timestep\n[{split_name}, 1 sample, {n_steps} timesteps]")
    if traj_metadata is not None:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, _format_traj_metadata_text(traj_metadata), ha="center", va="bottom", fontsize=9,
                 bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8})
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_err_heatmap_rollout(true_rollout, pred_rollout, output_dir, filename_prefix, split_name, traj_metadata):
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
        split_name: str
            Split label for titles.
        traj_metadata: dict or None
            Trajectory metadata for error-mesh GIF footnotes; skipped when None.
        return: None
            Saves all requested validation plots.
    """
    output_dir = Path(output_dir)  # Normalize directory for GIF paths.
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)  # Shared geometry + per-field errors for all GIFs.
    print("Printing validation rollout...")
    for field_name, field_label in TARGET_FIELDS:
        for error_mode in ["signed_relative"]:
            output_path = output_dir / f"{filename_prefix}_{field_name}_{error_mode}_error.gif"  # One GIF per field/mode.
            _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path, split_name, len(true_rollout),
                                   traj_metadata)  # Render+save.


def _plot_scatter_edge_corr(true_rollout, pred_rollout, output_dir, filename_prefix, split_name, traj_metadata):
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
        split_name: str
            Split label for titles.
        traj_metadata: dict or None
            Trajectory metadata for scatter PNG footnotes; skipped when None.
    return: None
        Saves all requested validation plots.
    """
    output_dir = Path(output_dir)  # Ensure pathlib for filename joins.
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)  # Same cache as error GIFs (concat nodal data).
    num_timesteps = len(true_rollout)  # Title/context for scatter PNGs.
    for field_name, field_label in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"  # Output scatter per field.
        field_summary = rollout_summary["fields"][field_name]  # Slice precomputed targets/preds for this quantity.
        _plot_prediction_target_scatter(field_summary, scatter_path, split_name, num_timesteps, traj_metadata)  # Save nodal scatter PNG.
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
        return _finite_coordinate_bounds_from_stack(np.zeros((0, 3), dtype=np.float64))  # Empty list → fallback plot box.
    all_coordinates = np.concatenate(coordinates_per_step, axis=0)  # Stack all nodes from all timesteps into one (M,3) array.
    return _finite_coordinate_bounds_from_stack(all_coordinates)  # Min/max/padding for 3D axis setup in GIFs.

def in_depth_plots(true_rollout, pred_rollout, output_dir, traj_split_name, filename_prefix,
                   plot_diagnostics_advanced, traj_metadata, target_mode):
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
        plot_diagnostics_advanced: boo;
            boolean to plot more complex diagnostics
        traj_metadata: dict or None
            Per-trajectory metadata (fx, fy, fz, L, W, D, ood_features) for footnote textboxes; skipped when None.
        target_mode: str
            "von_mises" or "full_stress"; controls whether the 2x3 fullstress scatter mega-plot is generated
            (only available in full_stress mode where rollout graphs carry the `sig6` attribute).
    return: dict
        Nested dict with keys "images" and "videos", each mapping a plot name to its Path.
    """
    output_dir = Path(output_dir)  # Root folder for this trajectory’s validation artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create hierarchy before any savefig/mimsave.

    # Paths (for _scatter generated inside the function itself)
    gif_path = output_dir / f"{filename_prefix}_beam_comparison.gif"  # Twin-panel von Mises animation path.
    volume_png_path = output_dir / f"{filename_prefix}_volume_change.png"  # Static volume vs time PNG path.
    rollout_error_path = output_dir / f"{filename_prefix}_rollout_error_vs_timestep.png"  # MAE vs timestep PNG path.

    # Plots
    _plot_volume_change(pred_rollout, true_rollout, path=volume_png_path, split_name=traj_split_name,
                        traj_metadata=traj_metadata)  # 2D volume curves.
    _make_beam_comparison_gif(pred_rollout=pred_rollout, true_rollout=true_rollout, L=1.0, W=0.1, D=0.04,
                              gif_path=gif_path, fps=4, bbox_only_on_true=False, verbose=False,
                              split_name=traj_split_name, traj_metadata=traj_metadata)  # Side-by-side 3D stress GIF.
    _plot_roll_err_pos_vs_time(true_rollout, pred_rollout, rollout_error_path, split_name=traj_split_name)  # Position MAE curve.
    _plot_scatter_edge_corr(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=output_dir,
                            filename_prefix=filename_prefix, split_name=traj_split_name,
                            traj_metadata=traj_metadata)  # Per-field scatter PNGs.

    # Optional full-stress mega-scatter: only generated in full_stress mode and when do_rollout attached
    # `sig6` to every stored rollout graph.
    fullstress_scatter_path = output_dir / f"{filename_prefix}_fullstress_prediction_vs_target.png"
    if target_mode == "full_stress":
        _plot_fullstress_prediction_vs_target(true_rollout=true_rollout, pred_rollout=pred_rollout,
                                              output_path=fullstress_scatter_path,
                                              split_name=traj_split_name,
                                              num_timesteps=len(true_rollout),
                                              traj_metadata=traj_metadata)

    if plot_diagnostics_advanced:
        try:
            plot_mesh_diagnostics(true_rollout=true_rollout, pred_rollout=pred_rollout, output_dir=output_dir,
                                  filename_prefix=filename_prefix, split_name=traj_split_name)  # Extra PNGs under diagnostics/.
        except Exception as e:
            print(f"Error while plotting mesh diagnostics: {e}")

    paths = {"images": {}, "videos": {}}  # Collect paths for trainer/logging (may include diag entries below).
    paths["videos"]["beam_comparison_gif"] = gif_path  # Register main comparison animation.
    paths["images"]["volume_change"] = volume_png_path  # Register volume plot.
    paths["images"]["rollout_error_vs_timestep"] = rollout_error_path  # Register MAE curve.
    for field_name, _ in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"
        if scatter_path.exists():
            paths["images"][f"{field_name}_prediction_vs_target"] = scatter_path  # Add scatter if writer succeeded.
        if plot_diagnostics_advanced:
            corr_path = output_dir / f"{filename_prefix}_{field_name}_edge_error_correlation.png"
            if corr_path.exists():
                paths["images"][f"{field_name}_edge_error_correlation"] = corr_path  # Hexbin correlation (if enabled upstream).
    if target_mode == "full_stress" and fullstress_scatter_path.exists():
        paths["images"]["fullstress_prediction_vs_target"] = fullstress_scatter_path

    if plot_diagnostics_advanced:
        diag_dir = output_dir / "diagnostics"  # Subfolder populated by plot_mesh_diagnostics.
        if diag_dir.exists():
            for field_name, _ in TARGET_FIELDS:
                for suffix in ["neighbor_abs_error", "error_vs_dist_worst",
                               "error_vs_curvature", "error_vs_boundary_dist"]:
                    diag_path = diag_dir / f"{filename_prefix}_{field_name}_{suffix}.png"
                    if diag_path.exists():
                        paths["images"][f"diag/{field_name}_{suffix}"] = diag_path  # Register per-field diagnostic PNGs.
            dirichlet_path = diag_dir / f"{filename_prefix}_dirichlet_energy.png"
            if dirichlet_path.exists():
                paths["images"]["diag/dirichlet_energy"] = dirichlet_path  # Scalar energy diagnostic.
            spectral_path = diag_dir / f"{filename_prefix}_spectral_comparison.png"
            if spectral_path.exists():
                paths["images"]["diag/spectral_comparison"] = spectral_path  # Spectral comparison figure.
    return paths  # Caller merges these into run-level artifact lists.