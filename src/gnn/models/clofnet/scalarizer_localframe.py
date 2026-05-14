"""
scalarizer_localframe.py

Defines helper functions for ClofNet.
"""

import torch

def coord2localframe(edge_index, coord_xi, enable_norm_diff, crossdim1):
    """
    Construct the complete local frame per edge.
    This is the code counterpart of paper Eq. (2):
        a_ij = (x_i - x_j) / ||x_i - x_j||
        b_ij = (x_i x x_j) / ||x_i x x_j||
        c_ij = a_ij x b_ij

    Args:
        edge_index: tuple[torch.Tensor, torch.Tensor]
            Edge list as (row, col).
        coord_xi: torch.Tensor
            Coordinates of shape [N, 3].
        enable_norm_diff: bool
            normalize the difference
        crossdim1: bool
            distinguish between two different implementations
    return: (radial_rij, coord_diff, coord_cross, coord_vertical)
        radial_rij: squared edge lengths [E, 1]; the three frame directions each [E, 3].
    """
    EPS = 1e-5 # note =1 it's a heavy choice FIXME: try EPS=1e-5
    row, col = edge_index
    coord_diff_aij = coord_xi[row] - coord_xi[col] # unnormalized a_ij, shape [E, 3]
    radial_rij = torch.sum(coord_diff_aij ** 2, dim=1).unsqueeze(1)  # r_ij := norm^2(x_i-x_j), shape [E, 1]
    # Cross product of the endpoint position vectors.
    if crossdim1:
        coord_cross_bij = torch.cross(coord_xi[row], coord_xi[col]) # [E, 3]
    else:
        coord_cross_bij = torch.cross(coord_xi[row], coord_xi[col], dim=1) # [E, 3]
    if enable_norm_diff:
        norm = torch.sqrt(radial_rij) + EPS # [N, 3]
        coord_diff_aij = coord_diff_aij / norm # [N, 3]
        cross_norm = torch.sqrt(torch.sum(coord_cross_bij ** 2, dim=1).unsqueeze(1)) + EPS # [E, 1]
        coord_cross_bij = coord_cross_bij / cross_norm  # b_ij [E, 3]
    coord_vertical_cij = torch.cross(coord_diff_aij, coord_cross_bij, dim=1) # [E, 3]
    return radial_rij, coord_diff_aij, coord_cross_bij, coord_vertical_cij

def _compute_angle_feat(coord_frame_xi, coord_frame_xj, EPS):
    """
    Computes additional features representing the angle between two scalar coefficients.
    Helper for build_s_ij

    Args
        coff_i: torch.Tensor [E, 3]
        coff_j: torch.Tensor [E, 3]
        EPS: denominator eps
    :return: torch.Tensor [E, 2]
    """
    coords_ij_inner_prod = (coord_frame_xi * coord_frame_xj).sum(dim=-1, keepdim=True) # [E, 1]
    coff_i_norm = coord_frame_xi.norm(dim=-1, keepdim=True) # [E, 1]
    coff_j_norm = coord_frame_xj.norm(dim=-1, keepdim=True) # [E, 1]
    # Compute cosine similarity
    cosine_sim = coords_ij_inner_prod / (coff_i_norm * coff_j_norm + EPS) # [E, 1]
    # sin(x) = sqrt(1 - cos^2(x))
    sin_sim = torch.sqrt(torch.clamp(1 - cosine_sim ** 2, min=0.0)) # [E,1]
    pseudo_angle = torch.cat([sin_sim, cosine_sim], dim=-1) # [E, 2]
    return pseudo_angle # [E, 2]

def frame_basis_from_edges(coord_diff_e3, coord_cross_e3, coord_vertical_e3):
    """
    Stack three edge-local basis directions into a matrix suitable for torch.matmul.

    Args:
        coord_diff_e3: torch.Tensor
            First basis, shape [E, 3].
        coord_cross_e3: torch.Tensor
            Second basis, shape [E, 3].
        coord_vertical_e3: torch.Tensor
            Third basis, shape [E, 3].
    return: torch.Tensor
        Basis matrix of shape [E, 3, 3] (rows are a_ij, b_ij, c_ij per edge).
    """
    return torch.stack([coord_diff_e3, coord_cross_e3, coord_vertical_e3], dim=1)


def _scalarize_node_endpoints(edges, frame_basis_e33, vec_node_n3):
    """
    Project one per-node equivariant vector on both edge endpoints in the local frame. Helper for build_s_ij

    Args:
        edges: tuple[torch.Tensor, torch.Tensor]
            Directed edges (row, col).
        frame_basis_e33: torch.Tensor
            Frame matrix of shape [E, 3, 3].
        vec_node_n3: torch.Tensor
            Node vectors of shape [N, 3].
    return: torch.Tensor
        Concatenated coefficients [E, 6] = [coeff_i (3), coeff_j (3)].
    """
    row, col = edges
    vi = vec_node_n3[row]
    vj = vec_node_n3[col]
    coeff_i = torch.matmul(frame_basis_e33, vi.unsqueeze(-1)).squeeze(-1)
    coeff_j = torch.matmul(frame_basis_e33, vj.unsqueeze(-1)).squeeze(-1)
    return torch.cat([coeff_i, coeff_j], dim=-1)


def _scalarize_edge_vector(frame_basis_e33, vec_edge_e3):
    """
    Project one per-edge equivariant vector onto the local frame. Helper for build_s_ij

    Args:
        frame_basis_e33: torch.Tensor
            Frame matrix of shape [E, 3, 3].
        vec_edge_e3: torch.Tensor
            Edge vectors of shape [E, 3].
    return: torch.Tensor
        Coefficients of shape [E, 3].
    """
    return torch.matmul(frame_basis_e33, vec_edge_e3.unsqueeze(-1)).squeeze(-1)

def build_s_ij(a_i, node_vec_chunks, edge_vec_chunks, local_frame_basis_ij, edge_index_tuple):
    """
    Concatenate invariant scalarizations of all static node edge-vector triplets, acceleration, and edge vectors.

    Args:
        a_i: torch.Tensor
            Current acceleration estimate [N, 3] (used for endpoint projections and pseudo-angle).
        node_vec_chunks: torch.Tensor
            [N, K, 3] static vectors from node features.
        edge_vec_chunks: torch.Tensor
            [E, K, 3] static vectors from edge features.
        local_frame_basis_ij: torch.Tensor
            [E, 3, 3] basis matrix per edge.
        edge_index_tuple: tuple
            (row, col).
    return: torch.Tensor
        s_ij tensor [E, s_ij_dim] before concatenation with edge scalars.
    """
    eps = 1e-5
    parts = []
    num_k = int(node_vec_chunks.shape[1])
    for t in range(num_k):
        parts.append(_scalarize_node_endpoints(edge_index_tuple, local_frame_basis_ij, node_vec_chunks[:, t, :]))
    acc_end = _scalarize_node_endpoints(edge_index_tuple, local_frame_basis_ij, a_i)
    parts.append(acc_end)
    coeff_i_a = acc_end[:, 0:3]
    coeff_j_a = acc_end[:, 3:6]
    pseudo = _compute_angle_feat(coeff_i_a, coeff_j_a, eps)
    num_ek = int(edge_vec_chunks.shape[1])
    for t in range(num_ek):
        parts.append(_scalarize_edge_vector(local_frame_basis_ij, edge_vec_chunks[:, t, :]))
    rest = torch.cat(parts, dim=-1)
    return torch.cat([pseudo, rest], dim=-1)