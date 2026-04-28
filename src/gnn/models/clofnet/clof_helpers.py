"""
clof_helpers.py

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
    return: (coord_diff, coord_cross, coord_vertical)
        Edge-local basis tensors, each with shape [E, 3]. (then unsqueezed)
    """
    EPS = 1 # note =1 it's a heavy choice, TODO: try EPS=1e-5
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

def compute_angle_feat(coord_frame_xi, coord_frame_xj, EPS):
    """
    Computes additional features representing the angle between two scalar coefficients.

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

def split_node_features(x, split_idx_node):
    """
    Split PyG node tensor into boundary flag, stacked R^3-valued chunks, and remaining scalars.

    Args:
        x: torch.Tensor
            Full node features of shape [N, C] with layout [bc | vec (3k) | scalars].
        split_idx_node: int
            Width of the vector-valued block (must be divisible by 3); starts at column index 1 after bc.
    return: (bc, vec_chunks, scalars)
        bc: torch.Tensor [N, 1]
        vec_chunks: torch.Tensor [N, k, 3] with k = split_idx_node // 3
        scalars: torch.Tensor [N, C - 1 - split_idx_node]
    """
    bc = x[:, 0:1]
    vec_flat = x[:, 1 : 1 + split_idx_node]
    scalars = x[:, 1 + split_idx_node :]
    num_triplets = split_idx_node // 3
    vec_chunks = vec_flat.reshape(x.shape[0], num_triplets, 3)
    return bc, vec_chunks, scalars


def split_edge_features(edge_attr, split_idx_edge):
    """
    Split edge attribute tensor into stacked R^3-valued chunks and remaining scalars.

    Args:
        edge_attr: torch.Tensor
            Edge features of shape [E, C_e].
        split_idx_edge: int
            Leading width of vector triplets (must be divisible by 3).
    return: (vec_chunks, scalars)
        vec_chunks: torch.Tensor [E, k, 3] with k = split_idx_edge // 3
        scalars: torch.Tensor [E, C_e - split_idx_edge]
    """
    vec_flat = edge_attr[:, :split_idx_edge]
    scalars = edge_attr[:, split_idx_edge:]
    num_triplets = split_idx_edge // 3
    vec_chunks = vec_flat.reshape(edge_attr.shape[0], num_triplets, 3)
    return vec_chunks, scalars


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


def scalarize_node_endpoints(edges, frame_basis_e33, vec_node_n3):
    """
    Project one per-node equivariant vector on both edge endpoints in the local frame.

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


def scalarize_edge_vector(frame_basis_e33, vec_edge_e3):
    """
    Project one per-edge equivariant vector onto the local frame.

    Args:
        frame_basis_e33: torch.Tensor
            Frame matrix of shape [E, 3, 3].
        vec_edge_e3: torch.Tensor
            Edge vectors of shape [E, 3].
    return: torch.Tensor
        Coefficients of shape [E, 3].
    """
    return torch.matmul(frame_basis_e33, vec_edge_e3.unsqueeze(-1)).squeeze(-1)


def forward_pass_generic(model, x, h, n_nodes, edges_index, edge_attr_eij, vel, node_attr):
    """
    Run the ClofNet stack and return updated coordinates.

    Args:
        model: nn.Module
            model
        h: torch.Tensor
            Node features of shape [N, NODE_FEAT_DIM].
        x: torch.Tensor
            Coordinates of shape [N, 3].
        edges_index: tuple[torch.Tensor, torch.Tensor]
            Directed edge index tuple (row, col).
        vel: torch.Tensor
            Velocity tensor of shape [N, 3]. Included only for API compatibility with the other variants.
        edge_attr_eij: torch.Tensor
            Edge attributes of shape [E, EDGE_FEAT_DIM].
        node_attr: torch.Tensor or None
            Optional node-side attributes concatenated inside Clof_GCL.
        n_nodes: int
            Number of particles per graph in the batch. Needed to recover graphwise centroids after flattening.
    return: torch.Tensor
        Updated coordinates of shape [N, 3].
    """
    # Embed node and edges
    h = model.embedding_node_mlp(h) # [N, hidden_dim_node]
    # not mandatory: you can just embed everything via duse later
    # edge_attr = model.embedding_edge_mlp(edge_attr) # [E, hidden_dim_edge]
    # Translation handling: move every graph to its centroid-centered frame before constructing
    # complete local frames.
    x = x.reshape(-1, n_nodes, 3) # [1, N, 3]
    x_centroid = torch.mean(x, dim=1, keepdim=True) # [1, 3]
    x_centered = (x - x_centroid).reshape(-1, 3) # [N, 3]
    # coordinates of xi and xj projected on frame fij, and pseudo angles
    sij = model.scalarization(edges_index, x_centered, None) # [E, 8]
    # Embed edge only after fusing them with s_ij := (x_a_ij, x_b_ij, x_c_ij) invariant coordinates of x_i, x_j
    # after projecting on local frame basis (a_ij, b_ij, c_ij)
    eij_sij = torch.cat([edge_attr_eij, sij], dim=-1) # [E, EDGE_FEAT_DIM + 8]
    # embed the edge features
    eij_sij_embed = model.fuse_mlp_embed(eij_sij) # [E, FUSE_HIDD_DIM]
    # Pass through message passing layers
    for i in range(model.n_layers):
        h, x_centered, _ = model.graph_mp_block_list[i](h=h, edge_index=edges_index, coord_xi=x_centered, vel=vel,
                                                        eij_sij_embed=eij_sij_embed)
    # Restore the global translation.
    x = x_centered.reshape(-1, n_nodes, 3) + x_centroid # [-1, N, 3]
    x = x.reshape(-1, 3) # [N, 3]
    return x # [N, 3]