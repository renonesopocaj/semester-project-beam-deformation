"""
graph_ops.py

Graph-level operations that are computed once per trajectory and then cached
inside the TrajectorySample.extras dict.  Callers request results via the
helper functions below; if the result is not yet in extras it is computed and
stored on the spot.

All operations work on plain numpy / scipy objects (no torch dependency).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.linalg


# ─── hop-distance matrix ────────────────────────────────────────────────────

def hop_distance_matrix(traj):
    """
    Shortest-path hop distance between all pairs of nodes, cached in traj.extras.

    Uses scipy BFS (dijkstra with unweighted edges).  Only feasible for small
    meshes; caller is responsible for not passing huge graphs.

    Args:
        traj: TrajectorySample
    return: np.ndarray
        Shape (N, N), dtype int16.  Entry [i, j] = shortest-path hop count.
    """
    if "hop_dist" in traj.extras:
        return traj.extras["hop_dist"]

    n = traj.num_nodes
    src = traj.edge_index[0]
    dst = traj.edge_index[1]
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.csr_matrix((data, (src, dst)), shape=(n, n))
    # make symmetric
    adj = adj + adj.T
    adj.data[:] = 1.0
    dist = csgraph.shortest_path(adj, method="D", directed=False, unweighted=True)
    dist = dist.astype(np.int16)
    traj.extras["hop_dist"] = dist
    return dist


# ─── graph diameter ─────────────────────────────────────────────────────────

def graph_diameter(traj):
    """
    Graph diameter (maximum finite shortest-path distance).

    Args:
        traj: TrajectorySample
    return: int
    """
    if "diameter" in traj.extras:
        return traj.extras["diameter"]
    dist = hop_distance_matrix(traj)
    d = int(dist[np.isfinite(dist) & (dist > 0)].max())
    traj.extras["diameter"] = d
    return d


# ─── normalized Laplacian eigenvectors ──────────────────────────────────────

def laplacian_eigenvectors(traj):
    """
    Eigenvectors and eigenvalues of the normalized graph Laplacian, cached in
    traj.extras.

    L_norm = I - D^{-1/2} A D^{-1/2}

    Eigenvalues are in [0, 2] and sorted ascending.  The first eigenvector
    (eigenvalue ≈ 0) is the constant mode.

    Args:
        traj: TrajectorySample
    return: (eigenvalues, eigenvectors)
        eigenvalues: np.ndarray  shape (N,)
        eigenvectors: np.ndarray  shape (N, N), columns = eigenvectors
    """
    if "laplacian_eigs" in traj.extras:
        return traj.extras["laplacian_eigs"]

    n = traj.num_nodes
    src = traj.edge_index[0]
    dst = traj.edge_index[1]
    data = np.ones(len(src), dtype=np.float64)
    adj = sp.csr_matrix((data, (src, dst)), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1.0

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_norm = sp.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    L_dense = L_norm.toarray().astype(np.float64)

    eigenvalues, eigenvectors = scipy.linalg.eigh(L_dense)
    result = (eigenvalues, eigenvectors)
    traj.extras["laplacian_eigs"] = result
    return result


# ─── load-point BFS distance ────────────────────────────────────────────────

def load_distance_vector(traj):
    """
    Per-node BFS distance from the nearest load/clamped node, cached in
    traj.extras.

    The load mask is the complement of free_mask: nodes where free_mask=False
    are the source set (clamped / driven nodes).

    Args:
        traj: TrajectorySample
    return: np.ndarray
        Shape (N,), integer hop distance from the nearest source node.
        Source nodes themselves have distance 0.
    """
    if "load_dist" in traj.extras:
        return traj.extras["load_dist"]

    n = traj.num_nodes
    src = traj.edge_index[0]
    dst = traj.edge_index[1]
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.csr_matrix((data, (src, dst)), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1.0

    # BFS from all source nodes simultaneously: add a virtual super-source node
    # connected to every clamped node at zero cost.
    source_nodes = np.where(~traj.free_mask)[0]
    if len(source_nodes) == 0:
        # No clamped nodes: all distances = 0
        vec = np.zeros(n, dtype=np.int16)
        traj.extras["load_dist"] = vec
        return vec

    # Build augmented graph with one extra node (index n) connected to all sources
    extra_row = np.full(len(source_nodes), n, dtype=np.int32)
    extra_col = source_nodes.astype(np.int32)
    extra_data = np.ones(len(source_nodes), dtype=np.float32)
    rows = np.concatenate([src.astype(np.int32), dst.astype(np.int32), extra_row, extra_col])
    cols = np.concatenate([dst.astype(np.int32), src.astype(np.int32), extra_col, extra_row])
    vals = np.ones(len(rows), dtype=np.float32)
    aug_adj = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    dist_aug = csgraph.shortest_path(aug_adj, method="D", directed=False,
                                     unweighted=True, indices=n)
    vec = dist_aug[:n].astype(np.int16)
    # Subtract 1 because we routed through the virtual super-source
    vec = np.maximum(vec - 1, 0)
    traj.extras["load_dist"] = vec
    return vec
