import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import deque
from src.gnn.mesh_numerical_helpers.compute_angle import extract_surface_faces
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from pathlib import Path
from src.gnn.misc_helpers.misc import to_numpy
from src.gnn.validation.plot_rollout import TARGET_FIELDS, extract_scalar

# ---------------------------------------------------------------------------
#                                    Diagnostic helpers
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


def _plot_neighbor_abs_error(true_values, pred_values, edge_index, field_label, output_path, split_name, num_timesteps):
    """
    Diagnostic 5.1 -- scatter of absolute error on neighboring nodes.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    abs_error = np.abs(pred_values - true_values)
    a_i = abs_error[edge_index[0]]
    a_j = abs_error[edge_index[1]]

    if len(a_i) == 0 or np.std(a_i) == 0.0 or np.std(a_j) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(a_i, a_j)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 7))
    hexbin = ax.hexbin(a_i, a_j, gridsize=60, cmap="viridis", mincnt=1)
    fig.colorbar(hexbin, ax=ax, label="Edge count")
    ax.set_xlabel("Absolute error node i")
    ax.set_ylabel("Absolute error node j")
    ax.set_title(f"{field_label} Neighbor Abs-Error (corr={correlation:.4f})\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_distance_from_worst(true_values, pred_values, edge_index, num_nodes, field_label, output_path, split_name, num_timesteps):
    """
    Diagnostic 5.2 -- scatter of absolute error vs hop-distance from the
    worst-error node.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        num_nodes: int
            Total number of nodes.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    abs_error = np.abs(pred_values - true_values)
    worst_node = int(np.argmax(abs_error))
    distances = compute_shortest_path_distances(edge_index, [worst_node], num_nodes)
    reachable = distances >= 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distances[reachable], abs_error[reachable], s=8, alpha=0.3)
    ax.set_xlabel(f"Hop distance from worst node (node {worst_node})")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"{field_label} Error vs Distance from Worst Node\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_dirichlet_energy_bar(true_rollout_graph, pred_rollout_graph, output_path, split_name, num_timesteps):
    """
    Diagnostic 7.1 -- grouped bar chart of Dirichlet energy E(y) vs E(y_hat)
    for all target fields, computed at a single timestep.

    Args:
        true_rollout_graph: Data
            Single ground-truth rollout graph.
        pred_rollout_graph: Data
            Single predicted rollout graph.
        output_path: Path
            Destination image path.
    """
    edge_index = to_numpy(true_rollout_graph.edge_index).astype(np.int64)
    field_labels = []
    energy_true_list = []
    energy_pred_list = []

    for field_name, field_label in TARGET_FIELDS:
        true_vals = extract_scalar(true_rollout_graph, field_name)
        pred_vals = extract_scalar(pred_rollout_graph, field_name)
        energy_true_list.append(compute_dirichlet_energy(true_vals, edge_index))
        energy_pred_list.append(compute_dirichlet_energy(pred_vals, edge_index))
        field_labels.append(field_label)

    x_pos = np.arange(len(field_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - bar_width / 2, energy_true_list, bar_width, label="True E(y)")
    ax.bar(x_pos + bar_width / 2, energy_pred_list, bar_width, label=r"Predicted E($\hat{y}$)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(field_labels, rotation=15, ha="right")
    ax.set_ylabel("Dirichlet Energy")
    ax.set_title(f"Dirichlet Energy: True vs Predicted\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_spectral_comparison(true_rollout_graph, pred_rollout_graph, output_path, k, split_name, num_timesteps):
    """
    Diagnostic 7.2 -- line plots comparing spectral content (graph Laplacian)
    of true vs predicted fields. One subplot per target field.

    Args:
        true_rollout_graph: Data
            Single ground-truth rollout graph.
        pred_rollout_graph: Data
            Single predicted rollout graph.
        output_path: Path
            Destination image path.
        k: int
            Number of Laplacian eigenvalues to compute.
    """
    edge_index = to_numpy(true_rollout_graph.edge_index).astype(np.int64)
    num_nodes = int(to_numpy(true_rollout_graph.x_pos_t).shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, (field_name, field_label) in enumerate(TARGET_FIELDS):
        ax = axes[idx]
        true_signal = extract_scalar(true_rollout_graph, field_name)
        pred_signal = extract_scalar(pred_rollout_graph, field_name)

        result_true = compute_graph_laplacian_spectrum(edge_index, true_signal, num_nodes, k)
        result_pred = compute_graph_laplacian_spectrum(edge_index, pred_signal, num_nodes, k)

        if result_true is None or result_pred is None:
            ax.text(0.5, 0.5, "Spectral analysis unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(field_label)
            continue

        eig_true, coeff_true = result_true
        eig_pred, coeff_pred = result_pred

        ax.plot(np.arange(len(coeff_true)), coeff_true, label="True", linewidth=1.5)
        ax.plot(np.arange(len(coeff_pred)), coeff_pred, label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(r"$|c_k|^2$")
        ax.set_title(f"{field_label} Spectral Content")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log", nonpositive="clip")

    fig.suptitle(f"Low-Frequency vs High-Frequency Content\n[{split_name}, 1 sample, {num_timesteps} timesteps]", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_curvature(true_values, pred_values, positions, element_connectivity, field_label, output_path, split_name, num_timesteps):
    """
    Diagnostic 8.3 -- scatter of absolute error vs mean curvature at
    surface (boundary) nodes.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        positions: np.ndarray
            (N, 3) node coordinates (true mesh).
        element_connectivity: object
            Tetrahedral connectivity from the graph.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    boundary_faces = extract_surface_faces(element_connectivity)
    boundary_nodes = np.unique(boundary_faces.ravel())
    node_indices, curvature = compute_discrete_mean_curvature(positions, boundary_faces, boundary_nodes)

    abs_error = np.abs(pred_values - true_values)
    error_at_boundary = abs_error[node_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(curvature, error_at_boundary, s=8, alpha=0.3)
    ax.set_xlabel("Mean Curvature")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"{field_label} Error vs Curvature (surface nodes)\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_boundary_distance(true_values, pred_values, edge_index, num_nodes, element_connectivity,
                                     field_label, output_path, split_name, num_timesteps):
    """
    Diagnostic 8.4 -- scatter of absolute error vs hop-distance to the
    nearest surface (boundary) node.

    Args:
        true_values: np.ndarray
            (N,) ground-truth scalar field.
        pred_values: np.ndarray
            (N,) predicted scalar field.
        edge_index: np.ndarray
            (2, E) edge endpoints.
        num_nodes: int
            Total number of nodes.
        element_connectivity: object
            Tetrahedral connectivity from the graph.
        field_label: str
            Human-readable field name for the title.
        output_path: Path
            Destination image path.
    """
    boundary_nodes = extract_boundary_node_set(element_connectivity)
    distances = compute_shortest_path_distances(edge_index, boundary_nodes, num_nodes)
    reachable = distances >= 0

    abs_error = np.abs(pred_values - true_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distances[reachable], abs_error[reachable], s=8, alpha=0.3)
    ax.set_xlabel("Hop distance to boundary")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"{field_label} Error vs Distance to Boundary\n[{split_name}, 1 sample, {num_timesteps} timesteps]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_mesh_diagnostics(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
    """
    Orchestrator that runs all six mesh diagnostics (5.1, 5.2, 7.1, 7.2,
    8.3, 8.4) on the last rollout timestep.

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Parent directory for output; a ``diagnostics/`` subdirectory is created.
        filename_prefix: str
            Prefix used for saved file names.
    """
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    true_graph = true_rollout[-1]
    pred_graph = pred_rollout[-1]
    num_timesteps = len(true_rollout)
    edge_index = to_numpy(true_graph.edge_index).astype(np.int64)
    positions = to_numpy(true_graph.x_pos_t)
    num_nodes = positions.shape[0]
    print(f"Plotting mesh diagnostics for {split_name}...")

    for field_name, field_label in TARGET_FIELDS:
        true_vals = extract_scalar(true_graph, field_name)
        pred_vals = extract_scalar(pred_graph, field_name)

        # 5.1 Neighbor absolute error
        print(f"\t\tDiag 5.1 ({field_name})...")
        _plot_neighbor_abs_error(true_vals, pred_vals, edge_index, field_label,
                                 diag_dir / f"{filename_prefix}_{field_name}_neighbor_abs_error.png",
                                 split_name, num_timesteps)

        # 5.2 Error vs distance from worst node
        print(f"\t\tDiag 5.2 ({field_name})...")
        _plot_error_vs_distance_from_worst(true_vals, pred_vals, edge_index, num_nodes, field_label,
                                           diag_dir / f"{filename_prefix}_{field_name}_error_vs_dist_worst.png",
                                           split_name, num_timesteps)

        # 8.3 Error vs curvature
        print(f"\t\tDiag 8.3 ({field_name})...")
        _plot_error_vs_curvature(true_vals, pred_vals, positions, true_graph.x_element_connectivity,
                                 field_label,
                                 diag_dir / f"{filename_prefix}_{field_name}_error_vs_curvature.png",
                                 split_name, num_timesteps)

        # 8.4 Error vs boundary distance
        print(f"\t\tDiag 8.4 ({field_name})...")
        _plot_error_vs_boundary_distance(true_vals, pred_vals, edge_index, num_nodes,
                                         true_graph.x_element_connectivity, field_label,
                                         diag_dir / f"{filename_prefix}_{field_name}_error_vs_boundary_dist.png",
                                         split_name, num_timesteps)

    # 7.1 Dirichlet energy (single combined plot)
    print("\t\tDiag 7.1 (Dirichlet energy)...")
    _plot_dirichlet_energy_bar(true_graph, pred_graph,
                               diag_dir / f"{filename_prefix}_dirichlet_energy.png",
                               split_name, num_timesteps)

    # 7.2 Spectral comparison (single combined figure)
    print("\t\tDiag 7.2 (spectral comparison)...")
    _plot_spectral_comparison(true_graph, pred_graph,
                              diag_dir / f"{filename_prefix}_spectral_comparison.png", k=50,
                              split_name=split_name, num_timesteps=num_timesteps)