"""
build_features.py

Builds features for the GNN and physical quantities.
- Computes von mises stress from stress tensor
- Build edge and node features computed from the input ones using also jit compiling to make it faster
"""

import torch

EPS = 1e-8

@torch.jit.script
def _build_node_features_core(x, v, x_tm1_prev, rest_pos, node_force, enable_past_pos_displ: bool,
                              enable_absolute_pos: bool, enable_node_norms: bool):
    """
    TorchScript-compatible node feature builder with explicit feature flags.
    """
    feat_list = []
    feat_list.append(v)
    u = x - rest_pos
    feat_list.append(u)
    feat_list.append(node_force)
    split_idx_node = 9
    u_tm1 = x_tm1_prev - rest_pos
    if enable_past_pos_displ:
        if enable_absolute_pos:
            feat_list.append(x_tm1_prev)
            split_idx_node += 3
        feat_list.append(u_tm1)
        split_idx_node += 3
    if enable_absolute_pos:
        feat_list.append(x)
        split_idx_node += 3
        feat_list.append(rest_pos)
        split_idx_node += 3
    if enable_node_norms:
        feat_list.append(torch.norm(u, dim=1, keepdim=True))
        feat_list.append(torch.norm(node_force, dim=1, keepdim=True))
        feat_list.append(torch.norm(v, dim=1, keepdim=True))
        if enable_absolute_pos:
            feat_list.append(torch.norm(x, dim=1, keepdim=True))
            feat_list.append(torch.norm(rest_pos, dim=1, keepdim=True))
        if enable_past_pos_displ:
            feat_list.append(torch.norm(u_tm1, dim=1, keepdim=True))
            if enable_absolute_pos:
                feat_list.append(torch.norm(x_tm1_prev, dim=1, keepdim=True))
    node_feat_tensor = torch.cat(feat_list, dim=1)
    return node_feat_tensor, split_idx_node


@torch.jit.script
def _build_edge_features_core(x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index, enable_edge_velocity: bool,
                              enable_force_edge: bool, enable_stretch_edge: bool, enable_rotation_angle: bool,
                              enable_absolute_pos: bool, enable_past_pos_displ: bool):
    """
    TorchScript-compatible edge feature builder with explicit feature flags.
    """
    src = edge_index[0]
    dst = edge_index[1]
    feat_list = []
    eps = 1e-8

    xij_t = x_t[dst] - x_t[src]
    x_tm1_past_ij = x_tm1_prev[dst] - x_tm1_prev[src]
    u_tm1_past = x_tm1_prev - rest_pos
    rest_pos_ij = rest_pos[dst] - rest_pos[src]
    u_t = x_t - rest_pos
    uij_t = u_t[dst] - u_t[src]
    feat_list.append(uij_t)
    split_idx_edge = 3

    force_ij = node_force_t[dst] - node_force_t[src]
    vij = v_t[dst] - v_t[src]

    if enable_edge_velocity:
        feat_list.append(vij)
        split_idx_edge += 3
    if enable_force_edge:
        feat_list.append(force_ij)
        split_idx_edge += 3
    if enable_absolute_pos:
        feat_list.append(xij_t)
        split_idx_edge += 3
        feat_list.append(rest_pos_ij)
        split_idx_edge += 3
    if enable_past_pos_displ:
        u_tm1_past_ij = u_tm1_past[dst] - u_tm1_past[src]
        feat_list.append(u_tm1_past_ij)
        split_idx_edge += 3
        if enable_absolute_pos:
            feat_list.append(x_tm1_past_ij)
            split_idx_edge += 3

    xij_norm = torch.norm(xij_t, dim=1, keepdim=True)
    feat_list.append(xij_norm)
    rest_pos_ij_norm = torch.norm(rest_pos_ij, dim=1, keepdim=True)
    feat_list.append(rest_pos_ij_norm)
    uij_norm = torch.norm(uij_t, dim=1, keepdim=True)
    feat_list.append(uij_norm)
    if enable_force_edge:
        feat_list.append(torch.norm(force_ij, dim=1, keepdim=True))
    if enable_edge_velocity:
        feat_list.append(torch.norm(vij, dim=1, keepdim=True))
    if enable_stretch_edge:
        stretch_ij_norm = xij_norm / (rest_pos_ij_norm + eps)
        feat_list.append(stretch_ij_norm)
    if enable_rotation_angle:
        _e0 = rest_pos_ij / (rest_pos_ij_norm + eps)
        _et = xij_t / (xij_norm + eps)
        direction_cosine = torch.sum(_e0 * _et, dim=1, keepdim=True)
        feat_list.append(direction_cosine)
    edge_feat_tensor = torch.cat(feat_list, dim=1)
    return edge_feat_tensor, split_idx_edge


def make_scripted_feature_builders(preprocess_cfg):
    """
    Creates node/edge scripted feature builders with config flags baked in once.

    Args:
        preprocess_cfg: PreprocessYaml
    return: tuple
        (node_builder, edge_builder)
    """
    enable_past_pos_displ = preprocess_cfg.enable_past_pos_displ
    enable_absolute_pos = preprocess_cfg.enable_absolute_pos
    enable_node_norms = preprocess_cfg.enable_node_norms
    enable_edge_velocity = preprocess_cfg.enable_edge_velocity
    enable_force_edge = preprocess_cfg.enable_force_edge
    enable_stretch_edge = preprocess_cfg.enable_stretch_edge
    enable_rotation_angle = preprocess_cfg.enable_rotation_angle

    def node_builder(x, v, x_tm1_prev, rest_pos, node_force):
        return _build_node_features_core(x, v, x_tm1_prev, rest_pos, node_force, enable_past_pos_displ,
                                         enable_absolute_pos, enable_node_norms)

    def edge_builder(x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index):
        return _build_edge_features_core(x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index,
                                         enable_edge_velocity, enable_force_edge, enable_stretch_edge,
                                         enable_rotation_angle, enable_absolute_pos, enable_past_pos_displ)

    return node_builder, edge_builder


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
    enable_past_pos_displ, enable_absolute_pos, enable_node_norms = (preprocess_cfg.enable_past_pos_displ,
        preprocess_cfg.enable_absolute_pos, preprocess_cfg.enable_node_norms)
    return _build_node_features_core(x, v, x_tm1_prev, rest_pos, node_force, enable_past_pos_displ,
                                     enable_absolute_pos, enable_node_norms)


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
    (enable_edge_velocity, enable_force_edge, enable_stretch_edge, enable_rotation_angle, enable_absolute_pos,
     enable_past_pos_displ) = (preprocess_cfg.enable_edge_velocity, preprocess_cfg.enable_force_edge,
                               preprocess_cfg.enable_stretch_edge, preprocess_cfg.enable_rotation_angle,
                               preprocess_cfg.enable_absolute_pos, preprocess_cfg.enable_past_pos_displ)
    return _build_edge_features_core(x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index,
                                     enable_edge_velocity, enable_force_edge, enable_stretch_edge,
                                     enable_rotation_angle, enable_absolute_pos, enable_past_pos_displ)


def probe_feature_layout(preprocess_cfg, device):
    """
    Run build_node_features / build_edge_features on minimal tensors to infer channel counts and split indices.

    Args:
        preprocess_cfg: object
            Preprocess configuration (e.g. PreprocessYaml) with the same feature toggle flags as training.
        device: torch.device
            Device for probe tensors.

    return: tuple
        (node_channels, edge_channels, split_idx_node, split_idx_edge)
            node_channels is 1 (boundary flag) plus the width of valued node features from build_node_features.
            edge_channels is the width of build_edge_features output.
            split_idx_node and split_idx_edge are the isotropic / anisotropic split indices used by the normalizer.

    Resulting tensor shapes from the builders (here n=2 nodes, E=1 edge):
        node_feat: (n, F_node) with F_node = valued node feature width; only shape[1] is used.
        edge_feat: (E, F_edge) with F_edge = edge feature width; only shape[1] is used.
        F_node and F_edge depend on preprocess_cfg toggles, not on n or E, as long as the graph is valid.
    """
    # At least two nodes so we can form one edge (0 -> 1); edge features index src/dst and need distinct endpoints.
    n = 2
    # Per-node kinematics / forces: (num_nodes, 3) for (x, y, z).
    x = torch.zeros(n, 3, device=device, dtype=torch.float32)
    v = torch.zeros(n, 3, device=device, dtype=torch.float32)
    x_tm1_prev = torch.zeros(n, 3, device=device, dtype=torch.float32)
    rest_pos = torch.zeros(n, 3, device=device, dtype=torch.float32)
    node_force = torch.zeros(n, 3, device=device, dtype=torch.float32)
    # Single undirected edge in COO form: shape (2, num_edges) = (2, 1).
    edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    node_feat, split_idx_node = build_node_features(x, v, x_tm1_prev, rest_pos, node_force, preprocess_cfg)
    edge_feat, split_idx_edge = build_edge_features(x, v, x_tm1_prev, rest_pos, node_force, edge_index,
                                                    preprocess_cfg)
    node_channels = 1 + int(node_feat.shape[1])
    edge_channels = int(edge_feat.shape[1])
    return node_channels, edge_channels, split_idx_node, split_idx_edge


def compute_von_mises_3d(stress_9):
    """
    Compute von Mises stress from 9-component stress tensor.
    Args:
        stress_9: torch.Tensor (N, 9)
            full stress tensor

    :return
        vm: torch.Tensor (N,)
    """
    sxx = stress_9[:, 0]
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