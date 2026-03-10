from collections import Counter
import torch
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
    Calculate volumes of tetrahedra. Used in plot_volume_change

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