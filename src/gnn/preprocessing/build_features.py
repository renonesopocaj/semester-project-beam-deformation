import torch

def calculate_kinematics(x, dt):
    """
    Calculate velocities and accelerations from positions. The idea is to fix t=0 velocity at 0.

    Args:
        x: Tensor (T, N, 3) = (#timesteps, #points in a graph, 3)
            Positions over time
        dt: float
            Time step size

    Returns: Tuple (v, a) of two tensors
        v: Velocities (T, N, 3)
        a: Accelerations (T-1, N, 3)
    """
    v0 = torch.zeros(*x[0].shape, device=x.device, dtype=x.dtype).unsqueeze(0) # (1, N, 3)
    dx = x[1:] - x[:-1]  # (T-1, N, 3) computes dx by shifting by 1, first excluding t=0 then excluding t=last
    v_steps = dx / dt  # (T-1, N, 3)
    v = torch.cat([v0, v_steps], dim=0) # (T, N, 3)
    dv = v[1:] - v[:-1] # same as velocity  (T-1, N, 3): but why isn't it brought back to (T, N, 3)?
    a = dv / dt
    return v, a


def build_valued_node_features(x, v, u, node_force, rest_minus_nonrest_pos):
    """
    Build valued node features: [position, velocity, rest_position, node_force]

    Args:
        x: Tensor
        v: Tensor
        u: Tensor
        node_force: Tensor
    :return Tensor
        concatenated features
    """
    if rest_minus_nonrest_pos:
        x_minus_u = x - u
        node_attr = torch.concat([x, v, u, node_force, x_minus_u], dim=1) # (15)
    else:
        node_attr = torch.concat([x, v, u, node_force], dim=1) # (12)
    return node_attr


def build_edge_features(x, u, edge_index, rest_minus_nonrest_pos_edge):
    """
    Build edge features from node positions and rest positions.
    Features: [relative_position, relative_position_norm,
               rest_relative_position, rest_relative_position_norm]
    Args:
        x: torch.Tensor
            absolute position, from which I compute the relative one and the norm
        u: torch.Tensor
            rest absolute position
        edge_index: torch.Tensor (2, #edges)
            Pairwise graph edges `(i,j)` for message passing, so it's a tensor of  i.e. a matrix with two columns/vecs
            of length #edges
    :return torch.Tensor (N, 8, 1)
        xij (N, 3), xij_norm (N, 3), uij (N, 1), uij_norm (N, 1)
    """
    src, dst = edge_index
    xij = x[dst] - x[src] # (N, 3) relative position while not in rest, from which I compute the relative one
    uij = u[dst] - u[src] # (N, 3) rest relative position
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1) # (N, 1) relative position in norm
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1) # (N, 1) rest_relative_position
    if rest_minus_nonrest_pos_edge: # 12 feat
        xij_minus_uij = xij - uij
        xij_minus_uij_norm = torch.norm(xij_minus_uij, dim=1).unsqueeze(1)
        # (N, 12, 1)
        edge_attr = torch.concat([xij, uij, xij_minus_uij, xij_norm, uij_norm, xij_minus_uij_norm], dim=1)
    else: # 8 feat
        edge_attr = torch.concat([xij, uij, xij_norm, uij_norm], dim=1) # (N, 8, 1)
    return edge_attr


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