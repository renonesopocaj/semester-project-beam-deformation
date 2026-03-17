"""
visualization_fea.py
This file contains helper functions for visualizing FEA results.
"""

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import numpy as np
import shutil
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import dolfin as df

def _compute_von_mises_3d(stress_9):
    """
    Compute the 3D von Mises stress from flattened 3x3 stress tensors.

    Args:
        stress_9: np.ndarray
            Array of shape (num_nodes, 9) containing flattened stress tensors
            in row-major order:
            [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz].
        return: np.ndarray
            Array of shape (num_nodes,) containing the von Mises stress.
    """
    # Extract tensor components.
    sxx = stress_9[:, 0]
    sxy = stress_9[:, 1]
    sxz = stress_9[:, 2]
    syx = stress_9[:, 3]
    syy = stress_9[:, 4]
    syz = stress_9[:, 5]
    szx = stress_9[:, 6]
    szy = stress_9[:, 7]
    szz = stress_9[:, 8]

    # Standard 3D von Mises stress formula.
    # Note that syx, szx, szy are read for completeness but not needed explicitly
    # because symmetry is assumed in the compact formula below.
    vm = 0.5 * (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
        + 6.0 * (sxy**2 + syz**2 + sxz**2)
    )

    return np.sqrt(vm)


def find_boundary_triangles(tetra_connectivity):
    """
    Find the exterior triangular faces of a tetrahedral mesh.

    A triangular face is on the boundary if it appears in exactly one tetrahedron.

    Args:
        tetra_connectivity: np.ndarray
            Array of shape (num_tetra, 4) containing tetrahedral node indices.
        return: np.ndarray
            Array of shape (num_boundary_faces, 3) containing boundary triangle
            node indices.
    """
    # Count how many times each triangular face appears.
    face_count = defaultdict(int)

    for tet in tetra_connectivity:
        # Each tetrahedron has four triangular faces.
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for f in faces:
            face_count[f] += 1

    # Faces appearing only once belong to the outer surface.
    boundary_faces = [list(face) for face, count in face_count.items() if count == 1]

    return np.array(boundary_faces, dtype=int)


def make_beam_gif(lst_graphs, L, W, D, out_gif, fps, verbose):
    """
    Render an animated GIF of the deformed beam surface colored by von Mises stress.

    Args:
        lst_graphs: list
            List of PyTorch Geometric Data objects. Each graph must contain:
            - x: node positions of shape (n, 3)
            - y_sig_t: flattened stress tensors of shape (n, 9)
            - x_element_connectivity: tetrahedral connectivity of shape (m, 4)
        L: float
            Beam length. Included for API compatibility, not directly used.
        W: float
            Beam width. Included for API compatibility, not directly used.
        D: float
            Beam depth. Included for API compatibility, not directly used.
        out_gif: str
            Output filename for the generated GIF.
        fps: int
            Frames per second of the GIF.
        return: None
            Saves the animation to disk.
    """
    # Temporary directory storing rendered PNG frames.
    temp_dir = "temp_frames"

    # Remove any old frames from previous runs.
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    # Use the first graph to recover the tetrahedral connectivity.
    first_graph = lst_graphs[0]
    cells = first_graph.x_element_connectivity
    if hasattr(cells, "cpu"):
        cells = cells.cpu().numpy()

    # Compute the surface triangulation once because mesh topology is constant.
    boundary_triangles = find_boundary_triangles(cells)

    # Collect all coordinates across all frames to enforce fixed axis limits.
    all_coords = np.concatenate([g.x.cpu().numpy() for g in lst_graphs], axis=0)
    x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    z_min, z_max = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])

    # Compute global stress range so the colormap stays consistent across frames.
    vm_min, vm_max = float("inf"), 0.0
    for g in lst_graphs:
        stress_9 = g.y_sig_t.cpu().numpy()
        vm = _compute_von_mises_3d(stress_9)
        vm_min = min(vm_min, np.min(vm))
        vm_max = max(vm_max, np.max(vm))

    for i, graph in enumerate(lst_graphs):
        if verbose:
            print(f"Rendering frame {i}/{len(lst_graphs) - 1}")

        # Extract node coordinates and stress values for the current frame.
        coords = graph.x.cpu().numpy()
        stress_9 = graph.y_sig_t.cpu().numpy()
        vm = _compute_von_mises_3d(stress_9)

        # Build one polygon per boundary triangle and assign one scalar value
        # per face as the average nodal von Mises stress.
        polys = []
        face_values = []

        for tri in boundary_triangles:
            tri_coords = coords[tri]
            polys.append(tri_coords)
            face_values.append(np.mean(vm[tri]))

        face_values = np.array(face_values)

        # Create a 3D figure.
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Map scalar face values to colors.
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=vm_min, vmax=vm_max)
        face_colors = cmap(norm(face_values))

        # Add colored triangular surface patches.
        polycoll = Poly3DCollection(polys)
        polycoll.set_facecolors(face_colors)
        polycoll.set_edgecolor("none")
        ax.add_collection3d(polycoll)

        # Keep axes fixed across frames to avoid visual jitter.
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Fix camera viewpoint for all frames.
        ax.view_init(elev=30, azim=45)

        # Add title and colorbar.
        ax.set_title(f"Time step {i}")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_values)
        cb = plt.colorbar(mappable, ax=ax, shrink=0.6)
        cb.set_label("von Mises Stress")

        # Save frame to disk.
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100)
        plt.close(fig)

        # Read saved frame so it can be added to the GIF.
        frames.append(imageio.imread(frame_path))

    # Combine all frames into a looping GIF.
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    if verbose:
        print(f"GIF saved => {out_gif}")

    # Clean up temporary PNG files.
    shutil.rmtree(temp_dir)
    if verbose:
        print("Done.")

def plot_check(time, u_tip, energies, save_path, verbose):
    """
    Save a diagnostic plot showing tip displacement and energy evolution.

    Args:
        time: np.ndarray
            Time array of shape (num_steps + 1,).
        u_tip: np.ndarray
            Tip displacement history.
        energies: np.ndarray
            Energy history with columns:
            [elastic, kinetic, damping, total].
        save_path: str
            Output path for the saved PNG figure.
        return: None
            Saves a plot to disk on rank 0.
    """
    # Only the root MPI process should save plots.
    if df.MPI.comm_world.rank == 0:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # Plot tip displacement versus time.
        ax[0].plot(time, u_tip)
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Tip displacement")
        ax[0].set_title("Tip Displacement Evolution")

        # Plot all tracked energies.
        ax[1].plot(time, energies)
        ax[1].legend(("Elastic", "Kinetic", "Damping", "Total"))
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Energies")
        ax[1].set_title("Energies Evolution")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        if verbose:
            print("plot saved to :", save_path)