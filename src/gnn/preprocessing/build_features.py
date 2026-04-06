"""
build_features.py
Builds features for the GNN and physical quantities.
"""

import torch

EPS = 1e-8

def build_node_features(x, v, rest_pos, node_force, enable_node_norms):
    """
    Build node features: [position, velocity, rest_position, node_force, displacement]

    Args:
        x: torch.Tensor (N, 3)
        v: torch.Tensor (N, 3)
        rest_pos: torch.Tensor (N, 3)
        node_force: torch.Tensor (N, 3)
        enable_node_norms: bool
    return: Tensor
        Concatenated features with shape (N, 15).
    """
    device = x.device
    feat_list = []
    # absolute position 3
    feat_list.append(x)
    # rest position 6
    rest_pos = rest_pos.to(device)
    feat_list.append(rest_pos)
    # velocty 9
    v = v.to(device)
    feat_list.append(v)
    # displacement field 12
    u = x - rest_pos
    feat_list.append(u)
    # node force 15
    node_force = node_force.to(device)
    feat_list.append(node_force)
    split_idx_node = 15
    # node norms (16-21)
    if enable_node_norms:
        feat_list.append(torch.norm(x, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(u, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(node_force, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(v, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(rest_pos, dim=1).unsqueeze(1))
    # Finally, concatenate
    node_feat_tensor = torch.concat(feat_list, dim=1)
    return node_feat_tensor, split_idx_node


def build_edge_features(x, u, edge_index, v, node_force, enable_velocity_edge, enable_force_edge, enable_stretch_edge,
                        enable_rotation_angle):
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
    # abs position (3)
    xij = x[dst] - x[src] # (E, 3) relative position
    feat_list.append(xij)
    # rest pos (6)
    rest_pos_ij = u[dst] - u[src] # (E, 3) rest relative position
    feat_list.append(rest_pos_ij)
    # displacement (9)
    uij = xij - rest_pos_ij
    feat_list.append(uij)
    split_idx_edge = 12
    # velocity (12)
    if enable_velocity_edge:
        vij = v[dst] - v[src]
        feat_list.append(vij)
        split_idx_edge += 3
    # force (15)
    if enable_force_edge:
        force_ij = node_force[dst] - node_force[src]
        feat_list.append(force_ij)
        split_idx_edge += 3
    # --------------- NORMS ----------------
    # position norm (16)
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1) # (E, 1)
    feat_list.append(xij_norm)
    # rest position norm (17)
    rest_pos_ij_norm = torch.norm(rest_pos_ij, dim=1).unsqueeze(1) # (E, 1)
    feat_list.append(rest_pos_ij_norm)
    # displacement norm (18)
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1)
    feat_list.append(uij_norm)
    # force norm (19)
    if enable_force_edge:
        force_ij_norm = torch.norm(force_ij, dim=1).unsqueeze(1)
        feat_list.append(force_ij_norm)
    # velocity norm (20)
    if enable_velocity_edge:
        vij_norm = torch.norm(vij, dim=1).unsqueeze(1)
        feat_list.append(vij_norm)
    # stretch (norm-only-feature) (21)
    if enable_stretch_edge:
        rest_pos_ij_norm = torch.norm(rest_pos_ij, dim=1).unsqueeze(1)
        stretch_ij_norm = xij_norm / (rest_pos_ij_norm + EPS)
        feat_list.append(stretch_ij_norm)
    # rotation angle (norm-only-feature) (22)
    if enable_rotation_angle:
        e0 = rest_pos_ij / (rest_pos_ij_norm + EPS)
        et = xij / (xij_norm + EPS)
        direction_cosine = torch.sum(e0 * et, dim=1, keepdim=True)
        feat_list.append(direction_cosine)
    # Finally concatenate all features
    edge_feat_tensor = torch.concat(feat_list, dim=1)
    return edge_feat_tensor, split_idx_edge

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