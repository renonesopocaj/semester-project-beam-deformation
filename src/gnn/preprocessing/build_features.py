"""
build_features.py
Builds features for the GNN and physical quantities.
"""

import torch

EPS = 1e-8

def build_node_features(x, v, x_tm1_prev, rest_pos, node_force, preprocess_cfg):
    """
    Build node features: [position, velocity, rest_position, node_force, displacement] FOR ONE TIMESTEP

    Args:
        x: torch.Tensor (N, 3)
        v: torch.Tensor (N, 3)
        rest_pos: torch.Tensor (N, 3)
        node_force: torch.Tensor (N, 3)
        preprocess_cfg: PreprocessYaml
    return: Tensor
        Concatenated features with shape (N, ???).
    """
    # Now
    enable_past_pos_displ_node, enable_absolute_pos, enable_node_norms = (preprocess_cfg.enable_past_pos_displ,
        preprocess_cfg.enable_absolute_pos, preprocess_cfg.enable_node_norms)
    # Start preprocessing
    device = x.device
    feat_list = []
    # velocty v (3)
    v = v.to(device)
    feat_list.append(v)
    # displacement field u (6)
    u = x - rest_pos
    feat_list.append(u)
    # node force f (9)
    node_force = node_force.to(device)
    feat_list.append(node_force)
    split_idx_node = 9
    # past position absolute and displacement x[t-1], u[t-1] (max +6 --> 15)
    if enable_past_pos_displ_node:
        # include x[t-1]
        if enable_absolute_pos:
            feat_list.append(x_tm1_prev)
            split_idx_node += 3
        # include u[t-1]
        u_tm1 = x_tm1_prev - rest_pos
        feat_list.append(u_tm1)
        split_idx_node += 3
    # absolute position x, rest absolute position rest_pos (max +6 --> 21)
    feat_list.append(x)
    if enable_absolute_pos:
        # absolute position x[t]
        feat_list.append(x)
        split_idx_node += 3
        # rest position x[0]
        rest_pos = rest_pos.to(device)
        feat_list.append(rest_pos)
        split_idx_node += 3
    # node norms of each feature (max +7 --> 28)
    if enable_node_norms:
        feat_list.append(torch.norm(u, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(node_force, dim=1).unsqueeze(1))
        feat_list.append(torch.norm(v, dim=1).unsqueeze(1))
        if enable_absolute_pos:
            feat_list.append(torch.norm(x, dim=1).unsqueeze(1))
            feat_list.append(torch.norm(rest_pos, dim=1).unsqueeze(1))
        if enable_past_pos_displ_node:
            feat_list.append(torch.norm(u_tm1, dim=1).unsqueeze(1))
            if enable_absolute_pos:
                feat_list.append(torch.norm(x_tm1_prev, dim=1).unsqueeze(1))

    # Finally, concatenate
    node_feat_tensor = torch.concat(feat_list, dim=1)
    return node_feat_tensor, split_idx_node


def build_edge_features(x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index, preprocess_cfg):
    """
    Build edge features from node positions and rest positions FOR ONE TIMESTEP.
    Features: [xij, uij, xij-uij, ||xij||, ||uij||, ||xij-uij||]

    Args:
        x_t: torch.Tensor
            absolute position, from which I compute the relative one and the norm
        u: torch.Tensor
            rest absolute position
        edge_index: torch.Tensor (2, #edges)
            Pairwise graph edges `(i,j)` for message passing
        v_t: torch.Tensor
            velocity
        preprocess_cfg: PreprocessYaml
            enable/disable features
    return: torch.Tensor (E, ???)
    """
    # First
    (enable_velocity_edge, enable_force_edge, enable_stretch_edge, enable_rotation_angle, enable_absolute_pos,
     enable_past_pos_displ_edge) = (preprocess_cfg.enable_velocity_edge, preprocess_cfg.enable_force_edge,
                                    preprocess_cfg.enable_stretch_edge, preprocess_cfg.enable_rotation_angle,
                                    preprocess_cfg.enable_absolute_pos, preprocess_cfg.enable_past_pos_displ)
    # Start preprocessing
    src, dst = edge_index
    feat_list = []
    xij_t = x_t[dst] - x_t[src] # (E, 3) relative position
    x_tm1_past_ij = x_tm1_prev[dst] - x_tm1_prev[src]
    u_tm1_past = x_tm1_prev - rest_pos
    # current displacement u_ij[t] (3)
    u_t = x_t - rest_pos
    uij_t = u_t[dst] - u_t[src] # (E, 3) rest relative position
    feat_list.append(uij_t)
    split_idx_edge = 3
    # velocity v_ij (6)
    if enable_velocity_edge:
        vij = v_t[dst] - v_t[src]
        feat_list.append(vij)
        split_idx_edge += 3
    # force f_ij (+3 --> max 9)
    if enable_force_edge:
        force_ij = node_force_t[dst] - node_force_t[src]
        feat_list.append(force_ij)
        split_idx_edge += 3
    if enable_absolute_pos:
        # current abs position x_ij[t] (+3 --> max 12)
        feat_list.append(xij_t)
        split_idx_edge += 3
        # rest pos x_ij[0] (+3 --> max 15)
        rest_pos_ij = rest_pos[dst] - rest_pos[src]
        feat_list.append(rest_pos_ij)
        split_idx_edge += 3
    if enable_past_pos_displ_edge:
        # past t-1 displacement u_ij[t-1] (+3 --> max 18)
        u_tm1_past_ij = u_tm1_past[dst] - u_tm1_past[src]
        feat_list.append(u_tm1_past_ij)
        split_idx_edge += 3
        # past t-1 absolute position x_ij[t-1] (+3 --> max 21)
        if enable_absolute_pos:
            feat_list.append(x_tm1_past_ij)
            split_idx_edge += 3
    # ==============================================================
    #                                   NORMS
    # ===============================================================
    # position norm (+1)
    xij_norm = torch.norm(xij_t, dim=1).unsqueeze(1) # (E, 1)
    feat_list.append(xij_norm)
    # rest position norm (+1)
    rest_pos_ij_norm = torch.norm(rest_pos_ij, dim=1).unsqueeze(1) # (E, 1)
    feat_list.append(rest_pos_ij_norm)
    # displacement norm (+1)
    uij_norm = torch.norm(uij_t, dim=1).unsqueeze(1)
    feat_list.append(uij_norm)
    # force norm (+1)
    if enable_force_edge:
        force_ij_norm = torch.norm(force_ij, dim=1).unsqueeze(1)
        feat_list.append(force_ij_norm)
    # velocity norm (+1)
    if enable_velocity_edge:
        vij_norm = torch.norm(vij, dim=1).unsqueeze(1)
        feat_list.append(vij_norm)
    # stretch (norm-only-feature) (+1)
    if enable_stretch_edge:
        rest_pos_ij_norm = torch.norm(rest_pos_ij, dim=1).unsqueeze(1)
        stretch_ij_norm = xij_norm / (rest_pos_ij_norm + EPS)
        feat_list.append(stretch_ij_norm)
    # rotation angle (norm-only-feature) (+1)
    if enable_rotation_angle:
        _e0 = rest_pos_ij / (rest_pos_ij_norm + EPS)
        _et = xij_t / (xij_norm + EPS)
        direction_cosine = torch.sum(_e0 * _et, dim=1, keepdim=True)
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