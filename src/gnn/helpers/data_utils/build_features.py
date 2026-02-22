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


def build_valued_node_features(x, v, u, node_force):
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
    return torch.concat([x, v, u, node_force], dim=1)


def build_edge_features(x, u, edge_index):
    """
    Build edge features from node positions and rest positions.
    Features: [relative_position, relative_position_norm,
               rest_relative_position, rest_relative_position_norm]
    Args:
        x: Tensor
        u: Tensor
        edge_index: Tensor
    :return Tensor
    """
    src, dst = edge_index
    xij = x[dst] - x[src]
    uij = u[dst] - u[src]
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1)
    return torch.concat([xij, xij_norm, uij, uij_norm], dim=1)


def compute_von_mises_3d(stress_9):
    """
    Compute von Mises stress from 9-component stress tensor.
    Args:
        stress_9: full stress tensor
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
    return torch.sqrt(vm)


def volume(V, T):
    """
    FIXME: take a look at it and understand it
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