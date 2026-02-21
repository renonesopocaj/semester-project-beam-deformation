def calculate_kinematics(x: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """Calculate velocities and accelerations from positions.

    Args:
        x: Positions over time (T, N, 3)
        dt: Time step size

    Returns:
        v: Velocities (T, N, 3)
        a: Accelerations (T-1, N, 3)
    """
    v0 = torch.zeros(*x[0].shape, device=x.device, dtype=x.dtype).unsqueeze(0)
    dx = x[1:] - x[:-1]
    v_steps = dx / dt
    v = torch.cat([v0, v_steps], dim=0)
    dv = v[1:] - v[:-1]
    a = dv / dt
    return v, a


def build_valued_node_features(
        x: Tensor, v: Tensor, u: Tensor, node_force: Tensor
) -> Tensor:
    """Build valued node features: [position, velocity, rest_position, node_force]"""
    return torch.concat([x, v, u, node_force], dim=1)


def build_edge_features(x: Tensor, u: Tensor, edge_index: Tensor) -> Tensor:
    """Build edge features from node positions and rest positions.

    Features: [relative_position, relative_position_norm,
               rest_relative_position, rest_relative_position_norm]
    """
    src, dst = edge_index
    xij = x[dst] - x[src]
    uij = u[dst] - u[src]
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1)
    return torch.concat([xij, xij_norm, uij, uij_norm], dim=1)


def compute_von_mises_3d(stress_9):
    """Compute von Mises stress from 9-component stress tensor."""
    sxx = stress_9[:, 0]
    sxy = stress_9[:, 1]
    sxz = stress_9[:, 2]
    syx = stress_9[:, 3]
    syy = stress_9[:, 4]
    syz = stress_9[:, 5]
    szx = stress_9[:, 6]
    szy = stress_9[:, 7]
    szz = stress_9[:, 8]
    vm = 0.5 * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    )
    return torch.sqrt(vm)


def volume(V: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Calculate volumes of tetrahedra.

    Args:
        V: Vertex positions (N, 3)
        T: Tetrahedra connectivity (M, 4)

    Returns:
        Volumes for each tetrahedron (M,)
    """
    a = V[T[:, 0], :]
    b = V[T[:, 1], :]
    c = V[T[:, 2], :]
    d = V[T[:, 3], :]
    vols = -torch.sum(torch.mul(a - d, torch.cross(b - c, c - d, dim=1)), dim=1) / 6.0
    return vols