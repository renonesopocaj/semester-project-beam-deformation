"""
build_features.py
Builds features for the GNN and physical quantities.
"""

import torch

def calculate_kinematics(x, dt):
    """
    Calculate velocities and accelerations from positions. The idea is to fix t=0 velocity at 0.
    The velocity is integrated via a backwards difference of position, acceleration via a forward difference of
    velocity, so acceleration is a second order centered difference wrt position.

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
    # first order backwards
    dx = x[1:] - x[:-1]  # (T-1, N, 3) computes dx by shifting by 1, first excluding t=0 then excluding t=last
    v_steps = dx / dt  # (T-1, N, 3)
    v = torch.cat([v0, v_steps], dim=0) # (T, N, 3)
    # NOTE: but why isn't it brought back to (T, N, 3)? Because then anyway I save single timesteps so no need to do
    # it now. Note also this is a forward difference wrt v --> second order centered difference wrt x
    dv = v[1:] - v[:-1] # same as velocity  (T-1, N, 3)
    a = dv / dt
    return v, a


def build_valued_node_features(x, v, rest_pos, node_force):
    """
    Build valued node features: [position, velocity, rest_position, node_force, displacement]

    Args:
        x: Tensor
        v: Tensor
        rest_pos: Tensor
        node_force: Tensor
    return: Tensor
        Concatenated features with shape (N, 15).
    """
    device = x.device
    rest_pos = rest_pos.to(device)
    v = v.to(device)
    x_minus_u = x - rest_pos
    return torch.concat([x, v, rest_pos, node_force, x_minus_u], dim=1)


def build_edge_features(x, u, edge_index):
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
    return: torch.Tensor (E, 12)
    """
    src, dst = edge_index
    xij = x[dst] - x[src] # (E, 3) relative position
    uij = u[dst] - u[src] # (E, 3) rest relative position
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1) # (E, 1)
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1) # (E, 1)
    xij_minus_uij = xij - uij
    xij_minus_uij_norm = torch.norm(xij_minus_uij, dim=1).unsqueeze(1)
    return torch.concat([xij, uij, xij_minus_uij, xij_norm, uij_norm, xij_minus_uij_norm], dim=1)


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