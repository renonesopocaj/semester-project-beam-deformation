import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch

from src.gnn.helpers.plots.mesh import compute_coordinate_bounds, extract_surface_faces, extract_tetra_connectivity, to_numpy, update_surface_collection

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

def volume(V, T):
    """
    Calculate volumes of tetrahedra.

    Args:
        V: torch.Tensor
            Vertex positions (N, 3)
        T: torch.Tensor
            Tetrahedra connectivity (M, 4)

    Returns: torch.Tensor
        Volumes for each tetrahedron (M,)
    """
    a = V[T[:, 0], :]
    b = V[T[:, 1], :]
    c = V[T[:, 2], :]
    d = V[T[:, 3], :]
    vols = -torch.sum(torch.mul(a - d, torch.cross(b - c, c - d, dim=1)), dim=1) / 6.0
    return vols