from collections import Counter, defaultdict, deque
import torch
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def to_numpy(value):
    """
    Args:
        value: object
            Tensor-like or array-like object.
        return: np.ndarray
            Detached numpy array.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def extract_tetra_connectivity(connectivity):
    """
    Args:
        connectivity: object
            Mesh connectivity stored in a graph.
        return: np.ndarray
            Tetrahedral connectivity with shape `(num_tetrahedra, 4)`.
    """
    cells = to_numpy(connectivity)
    while cells.ndim > 2:
        cells = cells[0]
    return cells.astype(np.int64)


def extract_boundary_faces(cells):
    """
    Args:
        cells: np.ndarray
            Tetrahedral connectivity with shape `(num_tetrahedra, 4)`.
        return: np.ndarray
            Boundary triangle faces with shape `(num_faces, 3)`.
    """
    face_counter = Counter()
    face_lookup = {}
    for tetrahedron in cells:
        tetrahedron = tetrahedron.tolist()
        tetra_faces = [
            (tetrahedron[0], tetrahedron[1], tetrahedron[2]),
            (tetrahedron[0], tetrahedron[1], tetrahedron[3]),
            (tetrahedron[0], tetrahedron[2], tetrahedron[3]),
            (tetrahedron[1], tetrahedron[2], tetrahedron[3]),
        ]
        for face in tetra_faces:
            sorted_face = tuple(sorted(face))
            face_counter[sorted_face] += 1
            if sorted_face not in face_lookup:
                face_lookup[sorted_face] = face
    boundary_faces = [face_lookup[face] for face, count in face_counter.items() if count == 1]
    return np.asarray(boundary_faces, dtype=np.int64)


def extract_surface_faces(connectivity):
    """
    Args:
        connectivity: object
            Mesh connectivity stored in a graph.
        return: np.ndarray
            Boundary triangle faces with shape `(num_faces, 3)`.
    """
    return extract_boundary_faces(extract_tetra_connectivity(connectivity))


def _iter_face_edges(face):
    return [(int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))]


def _get_oriented_edge(face, sorted_edge):
    for edge in _iter_face_edges(face):
        if tuple(sorted(edge)) == sorted_edge:
            return edge
    raise ValueError(f"Edge {sorted_edge} not found in face {face}")


def _flip_face_orientation(face):
    return np.asarray([face[0], face[2], face[1]], dtype=np.int64)


def orient_boundary_faces_consistently(boundary_faces):
    """
    Args:
        boundary_faces: np.ndarray
            Boundary triangle faces with shape `(num_faces, 3)`.
        return: np.ndarray
            Faces re-oriented so adjacent triangles traverse shared edges in opposite directions.
    """
    if len(boundary_faces) == 0:
        return boundary_faces.astype(np.int64, copy=True)

    oriented_faces = boundary_faces.astype(np.int64, copy=True)
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(oriented_faces):
        for edge in _iter_face_edges(face):
            edge_to_faces[tuple(sorted(edge))].append(face_idx)

    visited = np.zeros(len(oriented_faces), dtype=bool)
    queue = deque()
    for start_idx in range(len(oriented_faces)):
        if visited[start_idx]:
            continue
        visited[start_idx] = True
        queue.append(start_idx)
        while queue:
            face_idx = queue.popleft()
            face = oriented_faces[face_idx]
            for edge in _iter_face_edges(face):
                sorted_edge = tuple(sorted(edge))
                for neighbor_idx in edge_to_faces[sorted_edge]:
                    if neighbor_idx == face_idx:
                        continue
                    if not visited[neighbor_idx]:
                        neighbor_face = oriented_faces[neighbor_idx]
                        neighbor_edge = _get_oriented_edge(neighbor_face, sorted_edge)
                        if neighbor_edge == edge:
                            oriented_faces[neighbor_idx] = _flip_face_orientation(neighbor_face)
                        visited[neighbor_idx] = True
                        queue.append(neighbor_idx)
    return oriented_faces


def extract_boundary_dihedral_triangle_pairs(connectivity):
    """
    Args:
        connectivity: object
            Mesh connectivity stored in a graph.
        return: (np.ndarray, np.ndarray)
            Two arrays with shape `(3, num_pairs)` containing oriented boundary triangle
            node indices for each interior boundary edge.
    """
    boundary_faces = extract_surface_faces(connectivity)
    oriented_faces = orient_boundary_faces_consistently(boundary_faces)
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(oriented_faces):
        for edge in _iter_face_edges(face):
            edge_to_faces[tuple(sorted(edge))].append(face_idx)

    left_triangles = []
    right_triangles = []
    for face_indices in edge_to_faces.values():
        if len(face_indices) != 2:
            continue
        left_triangles.append(oriented_faces[face_indices[0]])
        right_triangles.append(oriented_faces[face_indices[1]])

    if len(left_triangles) == 0:
        empty = np.zeros((3, 0), dtype=np.int64)
        return empty, empty.copy()

    left_triangles = np.asarray(left_triangles, dtype=np.int64).T
    right_triangles = np.asarray(right_triangles, dtype=np.int64).T
    return left_triangles, right_triangles


def compute_max_boundary_dihedral_angle(positions, left_triangles, right_triangles, eps):
    """
    Args:
        positions: np.ndarray
            Node coordinates with shape `(num_nodes, 3)`.
        left_triangles: np.ndarray
            Oriented triangle indices with shape `(3, num_pairs)`.
        right_triangles: np.ndarray
            Oriented triangle indices with shape `(3, num_pairs)`.
        eps: float
            Numerical stability epsilon used for normalization and clamping.
        return: float
            Maximum dihedral angle across all interior boundary edges.
    """
    if left_triangles.shape[1] == 0:
        return 0.0

    left_positions = positions[left_triangles.T]
    right_positions = positions[right_triangles.T]

    left_normals = np.cross(left_positions[:, 1] - left_positions[:, 0], left_positions[:, 2] - left_positions[:, 0])
    right_normals = np.cross(right_positions[:, 1] - right_positions[:, 0], right_positions[:, 2] - right_positions[:, 0])

    left_norms = np.linalg.norm(left_normals, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_normals, axis=1, keepdims=True)
    left_unit = left_normals / np.maximum(left_norms, eps)
    right_unit = right_normals / np.maximum(right_norms, eps)

    cosine = np.sum(left_unit * right_unit, axis=1)
    cosine = np.clip(cosine, -1.0 + eps, 1.0 - eps)
    angles = np.arccos(cosine)
    return float(np.max(angles))


def extract_boundary_dihedral_triangle_pairs_torch(connectivity, device):
    """
    Args:
        connectivity: object
            Mesh connectivity stored in a graph.
        device: torch.device or str
            Device where the output tensors should live.
        return: (torch.Tensor, torch.Tensor)
            Two tensors with shape `(3, num_pairs)` containing oriented boundary
            triangle node indices for each interior boundary edge.
    """
    left_triangles, right_triangles = extract_boundary_dihedral_triangle_pairs(connectivity)
    left_triangles = torch.as_tensor(left_triangles, dtype=torch.long, device=device)
    right_triangles = torch.as_tensor(right_triangles, dtype=torch.long, device=device)
    return left_triangles, right_triangles


def filter_all_fixed_boundary_dihedral_pairs(left_triangles, right_triangles, fixed_mask):
    """
    Args:
        left_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        right_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        fixed_mask: torch.Tensor
            Boolean tensor with shape `(num_nodes,)` where True means fixed.
        return: (torch.Tensor, torch.Tensor, torch.Tensor)
            Filtered left triangles, filtered right triangles, and the keep mask.
    """
    if left_triangles.numel() == 0:
        keep_mask = torch.zeros((0,), dtype=torch.bool, device=fixed_mask.device)
        return left_triangles, right_triangles, keep_mask

    fixed_mask = fixed_mask.to(dtype=torch.bool)
    left_fixed = fixed_mask[left_triangles]
    right_fixed = fixed_mask[right_triangles]
    all_fixed = torch.cat([left_fixed, right_fixed], dim=0).all(dim=0)
    keep_mask = ~all_fixed
    return left_triangles[:, keep_mask], right_triangles[:, keep_mask], keep_mask


def compute_boundary_dihedral_angles_torch(positions, left_triangles, right_triangles, eps):
    """
    Args:
        positions: torch.Tensor
            Node coordinates with shape `(num_nodes, 3)`.
        left_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        right_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        eps: float
            Numerical stability epsilon used for normalization and clamping.
        return: torch.Tensor
            Dihedral angles with shape `(num_pairs,)`.
    """
    if left_triangles.numel() == 0:
        return positions.new_zeros((0,))

    left_positions = positions[left_triangles.transpose(0, 1)]
    right_positions = positions[right_triangles.transpose(0, 1)]
    left_normals = torch.cross(left_positions[:, 1] - left_positions[:, 0],
                               left_positions[:, 2] - left_positions[:, 0], dim=1)
    right_normals = torch.cross(right_positions[:, 1] - right_positions[:, 0],
                                right_positions[:, 2] - right_positions[:, 0], dim=1)
    left_unit = left_normals / torch.clamp(torch.linalg.norm(left_normals, dim=1, keepdim=True), min=eps)
    right_unit = right_normals / torch.clamp(torch.linalg.norm(right_normals, dim=1, keepdim=True), min=eps)
    cosine = torch.sum(left_unit * right_unit, dim=1)
    cosine = torch.clamp(cosine, min=-1.0 + eps, max=1.0 - eps)
    return torch.arccos(cosine)


def compute_max_boundary_dihedral_angle_torch(positions, left_triangles, right_triangles, eps):
    """
    Args:
        positions: torch.Tensor
            Node coordinates with shape `(num_nodes, 3)`.
        left_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        right_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        eps: float
            Numerical stability epsilon used for normalization and clamping.
        return: torch.Tensor
            Scalar tensor with the maximum dihedral angle.
    """
    angles = compute_boundary_dihedral_angles_torch(positions, left_triangles, right_triangles, eps)
    if angles.numel() == 0:
        return positions.new_zeros(())
    return torch.max(angles)


def compute_coordinate_bounds(coordinates_per_step):
    """
    Args:
        coordinates_per_step: list
            List of coordinate arrays with shape `(num_nodes, 3)`.
        return: tuple
            `(coordinate_min, coordinate_max, coordinate_span, padding)`.
    """
    all_coordinates = np.concatenate(coordinates_per_step, axis=0)
    coordinate_min = all_coordinates.min(axis=0)
    coordinate_max = all_coordinates.max(axis=0)
    coordinate_span = np.maximum(coordinate_max - coordinate_min, 1e-9)
    padding = 0.05 * coordinate_span
    return coordinate_min, coordinate_max, coordinate_span, padding


def update_surface_collection(ax, faces, coordinates, node_values, cmap, norm, poly_collection):
    """
    Args:
        ax: matplotlib.axes._subplots.Axes3DSubplot
            Target 3D axis.
        faces: np.ndarray
            Boundary faces with shape `(num_faces, 3)`.
        coordinates: np.ndarray
            Node coordinates with shape `(num_nodes, 3)`.
        node_values: np.ndarray
            Scalar values defined per node.
        cmap: matplotlib.colors.Colormap
            Colormap used for rendering.
        norm: matplotlib.colors.Normalize
            Normalization used for rendering.
        poly_collection: Poly3DCollection or None
            Existing collection to update.
        return: Poly3DCollection
            Created or updated collection.
    """
    polygons = coordinates[faces]
    face_values = np.mean(node_values[faces], axis=1)
    face_colors = cmap(norm(face_values))

    if poly_collection is None:
        poly_collection = Poly3DCollection(polygons)
        poly_collection.set_facecolors(face_colors)
        poly_collection.set_edgecolor("none")
        ax.add_collection3d(poly_collection)
        return poly_collection

    poly_collection.set_verts(polygons)
    poly_collection.set_facecolors(face_colors)
    return poly_collection

def volume(V, T):
    """
    Calculate volumes of each tetrahedra. Used in plot_volume_change. Does it by
    1. Computes four vertex as vectors of a tetrahedron  a,b,c,d
    2. Computes volume by doing vol = -1/6 (a-d) dot ((b-c) x (c-d))

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
    # vol = -1/6 (a-d) dot ((b-c) x (c-d))
    vols = -torch.sum(torch.mul(a - d, torch.cross(b - c, c - d, dim=1)), dim=1) / 6.0
    return vols


# ---------------------------------------------------------------------------
#  Diagnostic helpers
# ---------------------------------------------------------------------------

def compute_shortest_path_distances(edge_index, source_nodes, num_nodes):
    """
    Multi-source BFS on an undirected graph.

    Args:
        edge_index: np.ndarray
            (2, E) array of edge endpoints.
        source_nodes: np.ndarray or list
            Node indices used as BFS sources.
        num_nodes: int
            Total number of nodes in the graph.
        return: np.ndarray
            Integer hop-distance from each node to the nearest source.
            Unreachable nodes get distance -1.
    """
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in zip(edge_index[0], edge_index[1]):
        adjacency[int(src)].append(int(dst))

    distances = np.full(num_nodes, -1, dtype=np.int64)
    queue = deque()
    for s in source_nodes:
        s = int(s)
        if distances[s] == -1:
            distances[s] = 0
            queue.append(s)

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


def extract_boundary_node_set(element_connectivity):
    """
    Args:
        element_connectivity: object
            Tetrahedral connectivity stored in a graph.
        return: np.ndarray
            Sorted array of unique node indices that lie on boundary faces.
    """
    boundary_faces = extract_surface_faces(element_connectivity)
    return np.unique(boundary_faces.ravel())


def compute_discrete_mean_curvature(positions, boundary_faces, boundary_nodes):
    """
    Cotangent-weight Laplace-Beltrami mean curvature on a triangulated surface.
    |H_i| = 0.5 * ||Laplacian(x)_i|| / A_mixed_i, approximated via the cotangent
    scheme.  Returns absolute mean curvature at each boundary node.

    Args:
        positions: np.ndarray
            (N, 3) node coordinates.
        boundary_faces: np.ndarray
            (F, 3) triangle faces on the surface.
        boundary_nodes: np.ndarray
            Sorted array of boundary node indices.
        return: (np.ndarray, np.ndarray)
            (boundary_node_indices, curvature_values) -- arrays of equal length.
    """
    boundary_set = set(boundary_nodes.tolist())
    node_to_idx = {int(n): i for i, n in enumerate(boundary_nodes)}
    num_boundary = len(boundary_nodes)

    laplacian_vec = np.zeros((num_boundary, 3), dtype=np.float64)
    area_mixed = np.zeros(num_boundary, dtype=np.float64)

    for face in boundary_faces:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        pi, pj, pk = positions[i], positions[j], positions[k]

        eij = pj - pi
        eik = pk - pi
        ejk = pk - pj
        eji = pi - pj
        eki = pi - pk
        ekj = pj - pk

        cross_area = np.linalg.norm(np.cross(eij, eik))
        triangle_area = 0.5 * max(cross_area, 1e-30)

        def _cot_angle(u, v):
            dot = np.dot(u, v)
            cross_mag = np.linalg.norm(np.cross(u, v))
            return dot / max(cross_mag, 1e-30)

        cot_i = _cot_angle(eij, eik)
        cot_j = _cot_angle(eji, ejk)
        cot_k = _cot_angle(eki, ekj)

        for node, opp_cot, edge_vec in [
            (i, cot_k, eij), (i, cot_j, eik),
            (j, cot_i, ejk), (j, cot_k, eji),
            (k, cot_j, eki), (k, cot_i, ekj),
        ]:
            if node in boundary_set:
                idx = node_to_idx[node]
                laplacian_vec[idx] += opp_cot * edge_vec
                area_mixed[idx] += triangle_area / 3.0

    area_mixed = np.maximum(area_mixed, 1e-30)
    laplacian_vec /= (2.0 * area_mixed[:, None])
    curvature = np.linalg.norm(laplacian_vec, axis=1)
    return boundary_nodes, curvature


def compute_dirichlet_energy(values, edge_index):
    """
    Graph Dirichlet energy: E(z) = sum_{(i,j) in E} ||z_i - z_j||^2.

    Args:
        values: np.ndarray
            (N,) or (N, D) signal on nodes.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        return: float
            Scalar energy value.
    """
    diff = values[edge_index[0]] - values[edge_index[1]]
    if diff.ndim == 1:
        return float(np.sum(diff ** 2))
    return float(np.sum(diff ** 2))


def compute_graph_laplacian_spectrum(edge_index, signal, num_nodes, k):
    """
    Project a node signal onto the first k graph Laplacian eigenvectors and
    return the spectral coefficients.

    Args:
        edge_index: np.ndarray
            (2, E) edge endpoints (undirected graph assumed).
        signal: np.ndarray
            (N,) scalar signal on nodes.
        num_nodes: int
            Number of nodes.
        k: int
            Number of smallest eigenvalues/vectors to compute.
        return: (np.ndarray, np.ndarray) or None
            (eigenvalues, squared_spectral_coefficients) of length k,
            or None if scipy is unavailable or k > num_nodes.
    """
    k = min(k, num_nodes - 2)
    if k < 2:
        return None

    row = np.concatenate([edge_index[0], edge_index[1]])
    col = np.concatenate([edge_index[1], edge_index[0]])
    data = np.ones(len(row), dtype=np.float64)
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj.eliminate_zeros()
    adj = (adj + adj.T) / 2.0

    degree = np.array(adj.sum(axis=1)).ravel()
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    laplacian_norm = sp.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
    laplacian_norm = (laplacian_norm + laplacian_norm.T) / 2.0

    try:
        eigenvalues, eigenvectors = spla.eigsh(laplacian_norm, k=k, which="SM")
    except Exception as e:
        print(f"Error computing spectral analysis: {e}")
        return None

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    coefficients = eigenvectors.T @ signal
    return eigenvalues, coefficients ** 2