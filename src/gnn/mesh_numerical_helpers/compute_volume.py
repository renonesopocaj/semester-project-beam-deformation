"""
compute_volume.py
Functions to compute volume of tetrahedra.
"""

import torch
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import to_numpy_tetra_connectivity

def volume(V, T):
    """
    Calculate volumes of each tetrahedra. Used in plot_volume_change. Does it by
    1. Computes four vertex as vectors of a tetrahedron  a,b,c,d
    2. Computes volume by doing vol = -1/6 (a-d) dot ((b-c) x (c-d))

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
    # vol = -1/6 (a-d) dot ((b-c) x (c-d))
    vols = -torch.sum(torch.mul(a - d, torch.cross(b - c, c - d, dim=1)), dim=1) / 6.0
    return vols

def _compute_final_volumes(all_true_rollouts, all_pred_rollouts):
    """
    Computes the total mesh volume at the final rollout timestep for each trajectory.

    Args:
        all_true_rollouts: dict[int, list]
            Mapping of trajectory index to ground-truth rollout graphs.
        all_pred_rollouts: dict[int, list]
            Mapping of trajectory index to predicted rollout graphs.
        return: (list[float], list[float])
            Per-trajectory final true volume and final predicted volume.
    """
    true_volumes, pred_volumes = [], []
    # for all trajectories
    for traj_idx in sorted(all_true_rollouts.keys()):
        true_final_graph = all_true_rollouts[traj_idx][-1]
        pred_final_graph = all_pred_rollouts[traj_idx][-1]
        cells = torch.as_tensor(to_numpy_tetra_connectivity(true_final_graph.x_element_connectivity), dtype=torch.long)
        # Compute all signed volumes and take absolute values
        v_true = torch.abs(volume(true_final_graph.x_pos_t, cells)).sum().item()
        v_pred = torch.abs(volume(pred_final_graph.x_pos_t, cells)).sum().item()
        # Return list of final volume
        true_volumes.append(v_true)
        pred_volumes.append(v_pred)
    return true_volumes, pred_volumes