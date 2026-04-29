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


def total_absolute_volume(V, T):
    """
    Compute total absolute mesh volume from vertex positions and tetra connectivity.

    Args:
        V: torch.Tensor
            Vertex positions with shape (num_nodes, 3).
        T: torch.Tensor
            Tetrahedra connectivity with shape (num_tets, 4).
    return: torch.Tensor
        Scalar tensor equal to sum(abs(signed_tetra_volumes)).
    """
    return torch.abs(volume(V, T)).sum()


def batched_total_absolute_volumes(V, T_batched, batch_ptr, tet_slices):
    """
    Compute one total absolute volume value per graph for a PyG-collated batch.

    Args:
        V: torch.Tensor
            Batched vertex positions with shape (total_nodes, 3).
        T_batched: torch.Tensor
            Concatenated tetra indices from all graphs, shape (total_tets, 4), local per-graph indexing.
        batch_ptr: torch.Tensor
            Node pointer vector with shape (num_graphs + 1,), where ptr[i] is first node index of graph i.
        tet_slices: torch.Tensor
            Pointer vector for tetra connectivity with shape (num_graphs + 1,).
    return: torch.Tensor
        Per-graph total absolute volumes with shape (num_graphs,).
    """
    volumes_per_graph = []
    num_graphs = int(batch_ptr.numel()) - 1
    for graph_idx in range(num_graphs):
        node_offset = int(batch_ptr[graph_idx].item())
        tet_start = int(tet_slices[graph_idx].item())
        tet_end = int(tet_slices[graph_idx + 1].item())
        local_tets = T_batched[tet_start:tet_end]
        global_tets = local_tets + node_offset
        volumes_per_graph.append(total_absolute_volume(V, global_tets))
    if len(volumes_per_graph) == 0:
        return V.new_zeros((0,))
    return torch.stack(volumes_per_graph, dim=0)

def compute_final_volumes(all_true_rollouts, all_pred_rollouts):
    """
    Computes the total mesh volume at the final rollout timestep for each trajectory.
    Used in trainer.

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