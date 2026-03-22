"""
misc_numerical_helpers.py
Various helper functions for numerical mesh operations.
"""

import numpy as np
from src.gnn.misc_helpers.misc import to_numpy
from collections import Counter

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
        tetra_faces = [(tetrahedron[0], tetrahedron[1], tetrahedron[2]),
                       (tetrahedron[0], tetrahedron[1], tetrahedron[3]),
                       (tetrahedron[0], tetrahedron[2], tetrahedron[3]),
                       (tetrahedron[1], tetrahedron[2], tetrahedron[3])]
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