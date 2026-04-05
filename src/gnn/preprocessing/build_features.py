"""
build_features.py
Builds features for the GNN and physical quantities.
"""

import torch

def build_node_features(x, v, rest_pos, node_force, enable_node_norms, enable_stretch_node):
    """
    Build node features: [position, velocity, rest_position, node_force, displacement]

    Args:
        x: torch.Tensor (N, 3)
        v: torch.Tensor (N, 3)
        rest_pos: torch.Tensor (N, 3)
        node_force: torch.Tensor (N, 3)
        enable_node_norms: torch.Tensor (N, 3)
    return: Tensor
        Concatenated features with shape (N, 15).
    """
    device = x.device
    feat_list = []
    feat_list.append(x)
    # rest position
    rest_pos = rest_pos.to(device)
    feat_list.append(rest_pos)
    # velocty
    v = v.to(device)
    feat_list.append(v)
    # displacement field
    u = x - rest_pos
    feat_list.append(u)
    # node force
    feat_list.append(node_force)
    # stretch
    if enable_stretch_node:
        stretch_node = u / rest_pos
        feat_list.append(u / rest_pos)
        if enable_node_norms:
            feat_list.append(torch.norm(stretch_node, dim=1).unsqueeze(1))
    # node norms
    if enable_node_norms:
        feat_list.append(torch.norm(x, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(u, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(node_force, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(v, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(rest_pos, dim=1).unsqueeze(1))
    return torch.concat(feat_list, dim=1)


def build_edge_features(x, u, edge_index, v, node_force, enable_velocity_edge, enable_force_edge, enable_stretch_edge):
    """
    Build edge features from node positions and rest positions.
    Features: [xij, uij, xij-uij, ||xij||, ||uij||, ||xij-uij||]

    Args:
        x: torch.Tensor
            absolute position, from which I compute the relative one and the norm
        u: torch.Tensor
            rest absolute position
        edge_index: torch.Tensor (2, #edges)
            Pairwise graph edges `(i,j)` for message passing
        v: torch.Tensor
            velocity
        enable_velocity_edge: bool
            controls if add v_ij = v_i - v_j to the edge features
    return: torch.Tensor (E, 12)
    """
    src, dst = edge_index
    feat_list = []
    # abs position
    xij = x[dst] - x[src] # (E, 3) relative position
    feat_list.append(xij)
    # rest pos
    rest_pos_ij = u[dst] - u[src] # (E, 3) rest relative position
    feat_list.append(rest_pos_ij)
    # abs position norm
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1) # (E, 1)
    feat_list.append(xij_norm)
    # rest pos norm
    rest_pos_ij_norm = torch.norm(rest_pos_ij, dim=1).unsqueeze(1) # (E, 1)
    feat_list.append(rest_pos_ij_norm)
    # displacement
    uij = xij - rest_pos_ij
    feat_list.append(uij)
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1)
    feat_list.append(uij_norm)
    if enable_velocity_edge:
        vij = v[dst] - v[src]
        feat_list.append(torch.norm(vij, dim=1).unsqueeze(1))
    if enable_stretch_edge:
        stretch_ij = uij / rest_pos_ij
        feat_list.append(stretch_ij)
        stretch_ij_norm = torch.norm(stretch_ij, dim=1).unsqueeze(1)
        feat_list.append(stretch_ij_norm)
    if enable_force_edge:
        force_ij = node_force[dst] - node_force[src]
        force_ij_norm = torch.norm(force_ij, dim=1).unsqueeze(1)
        feat_list.append(force_ij_norm)
        feat_list.append(force_ij)
    return torch.concat(feat_list, dim=1)


def compute_von_mises_3d(stress_9):
    """
    Compute von Mises stress from 9-component stress tensor.
    Args:
        stress_9: torch.Tensor (N, 9)
            full stress tensor

    :return
        vm: torch.Tensor (N,)
    """
    sxx = stress_9[:, 0] # (N, 9)
    sxy = stress_9[:, 1]
    sxz = stress_9[:, 2]
    syy = stress_9[:, 4]
    syz = stress_9[:, 5]
    szz = stress_9[:, 8]
    vm = 0.5 * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    )
    return torch.sqrt(vm) # (N,)