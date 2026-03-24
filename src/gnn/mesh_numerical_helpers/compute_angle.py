"""
compute_angle.py
Contains helper functions for computing angles.
The main access point should be two:
 - in preprocess to compute theta_max
 - in train to compute the angle physics informed residual.
"""

from collections import defaultdict, deque
import torch
import numpy as np
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import extract_surface_faces

EPS = 1e-9

def _iter_face_edges(face):
    """
    Given a list of nodes of a face, returns a list of edges that connect them 0 --> 1 --> 2 --> 0

    Args:
        face: list
    :return: list[tuple]
    """
    return [(int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))]


def _get_oriented_edge(face, sorted_edge):
    """
    Search an oriented edge in a face (as a list of edges) (sorted to impose an order) and return it.

    Args:
        face:
        sorted_edge:
    """
    for edge in _iter_face_edges(face):
        if tuple(sorted(edge)) == sorted_edge:
            return edge
    raise ValueError(f"Edge {sorted_edge} not found in face {face}")


def _flip_face_orientation(face):
    """
    Flips the orientation of a face, by returning 0 <-- 0', 1' <-- 2, 2' <-- 1.
    Helper for _orient_boundary_faces_consistently

    Args:
        face:
    """
    return np.asarray([face[0], face[2], face[1]], dtype=np.int64)

def _edge_to_faces(faces):
    """
    Create a dict of how edges (as key, as couples) map to the faces that share them. The value is a list of face idx

    Args:
        faces: list of faces
    """
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for edge in _iter_face_edges(face):
            edge_to_faces[tuple(sorted(edge))].append(face_idx)
    return edge_to_faces

def _orient_boundary_faces_consistently(boundary_faces):
    """
    Given boundary faces, orient them so that when we compute the normals they are consistent.

    Args:
        boundary_faces: np.ndarray
            Boundary triangle faces with shape `(num_faces, 3)`.

    return: np.ndarray
        Faces re-oriented so adjacent triangles traverse shared edges in opposite directions.
    """
    if len(boundary_faces) == 0:
        return boundary_faces.astype(np.int64, copy=True)

    oriented_faces = boundary_faces.astype(np.int64, copy=True)
    # create a dict of how edges (as key, as couples) map to the faces that share them. The value is a list of face idx
    edge_to_faces = _edge_to_faces(oriented_faces)

    # Use queue to do BFS and orient faces the same for all of the mesh
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


def _extract_surface_triangles(connectivity):
    """
    From mesh connectivity, obtain the boundary/surface faces and from that obtain the edges on the boundary.
    Flip orientation of faces to get coherent normal vectors later.
    Finally

    Args:
        connectivity: object
            Mesh connectivity stored in a graph.

    return: (np.ndarray, np.ndarray)
        Two arrays with shape (3, num_pairs_tri) containing oriented boundary triangle
        node indices for each interior boundary edge.
    """
    # From cell connectivity array of the whole mesh, extract a list (np.ndarray) of only the boundary/surface faces.
    boundary_faces = extract_surface_faces(connectivity)
    # Orient faces
    oriented_faces = _orient_boundary_faces_consistently(boundary_faces)
    # create a dict of how edges (as key, as couples) map to the faces that share them. The value is a list of face idx
    edge_to_faces = _edge_to_faces(oriented_faces)

    left_triangles, right_triangles = [], []
    # For each edge, if it is shared by two faces, we have a two surfaces that have an angle between them, and thus
    # we can compute their angle. Store the faces, one in left and one in right:
    for face_indices in edge_to_faces.values():
        if len(face_indices) != 2:
            continue
        left_triangles.append(oriented_faces[face_indices[0]])
        right_triangles.append(oriented_faces[face_indices[1]])
        # num_pairs = number of shared surface edges that have exactly two surface triangles around them

    if len(left_triangles) == 0:
        empty = np.zeros((3, 0), dtype=np.int64)
        return empty, empty.copy()

    left_triangles = np.asarray(left_triangles, dtype=np.int64).T
    right_triangles = np.asarray(right_triangles, dtype=np.int64).T
    return left_triangles, right_triangles


def extract_surface_triangles_torch(connectivity, device):
    """
    Wrapper of _extract_surface_triangles that returns tensors on the device.
    - From mesh connectivity, obtain the boundary (surface) faces and from that obtain the edges on the boundary
    Used by preprocess.py

    Args:
        connectivity: object
            Mesh connectivity stored in a graph.
        device: torch.device or str
            Device where the output tensors should live.
    return: (torch.Tensor, torch.Tensor)
        Two tensors with shape (3, num_pairs) containing oriented boundary triangle node indices for each
        interior boundary edge.
    """
    # From mesh connectivity, obtain the boundary (surface) faces and from that obtain the edges on the boundary
    left_triangles, right_triangles = _extract_surface_triangles(connectivity)
    # Convert to tensors
    left_triangles = torch.as_tensor(left_triangles, dtype=torch.long, device=device)
    right_triangles = torch.as_tensor(right_triangles, dtype=torch.long, device=device)
    return left_triangles, right_triangles


def filter_fixed_surface_triangles(left_triangles, right_triangles, fixed_mask):
    """
    Filter out all boundary triangle angles that are connected to fixed nodes.
    Used  by preprocess.py

    Args:
        left_triangles: torch.Tensor
            Oriented triangle indices with shape (3, num_pairs_tri).
        right_triangles: torch.Tensor
            Oriented triangle indices with shape (3, num_pairs_tri).
        fixed_mask: torch.Tensor
            Boolean tensor with shape (num_nodes,) where True (1) means fixed, i.e. clamped to wall.
    return: (torch.Tensor, torch.Tensor, torch.Tensor)
        Filtered left triangles, filtered right triangles, and the keep mask.
    """
    # edge case
    if left_triangles.numel() == 0:
        keep_mask = torch.zeros((0,), dtype=torch.bool, device=fixed_mask.device)
        return left_triangles, right_triangles, keep_mask
    # filter out all nodes clamped to wall
    fixed_mask = fixed_mask.to(dtype=torch.bool)
    left_fixed = fixed_mask[left_triangles]
    right_fixed = fixed_mask[right_triangles]
    all_fixed = torch.cat([left_fixed, right_fixed], dim=0).all(dim=0)
    keep_mask = ~all_fixed
    return left_triangles[:, keep_mask], right_triangles[:, keep_mask], keep_mask


def compute_surface_angles_torch(positions, left_triangles, right_triangles):
    """
    Return a tensor with all computed angles between couples of triangles on the surface.
    Used by preprocess.py and loss

    Args:
        positions: torch.Tensor
            Node coordinates with shape (num_nodes, 3).
        left_triangles: torch.Tensor
            Oriented triangle indices with shape (3, num_pairs).
        right_triangles: torch.Tensor
            Oriented triangle indices with shape (3, num_pairs).
    return: torch.Tensor
        Angles between two triangles with shape (num_pairs,).
    """
    if left_triangles.numel() == 0:
        return positions.new_zeros((0,))
    # Gather the 3 vertex positions for every triangle for every triangle
    # i.e. one "row" of left_positions is one triangle and mapped to (3 vertices for each, 3 coords for each vertex)
    # left_triangles.transpose(0, 1): (num_pairs, 3) AND left_positions: (num_pairs_tri, 3, 3)
    left_positions = positions[left_triangles.transpose(0, 1)] # (num_pairs_tri, 3, 3)
    right_positions = positions[right_triangles.transpose(0, 1)] # (num_pairs_tri, 3, 3)
    # Compute normals and unit vectors. left_positions[:, 1]: (num_pairs_tri, 3) -- same for others
    # left_positions[:, 1] = second vertex, left_positions[:, 0] = first vertex, left_positions[:, 2] = third vertex.
    # So it does b - a x c - a for all (a,b,c) faces. dim = 1 says the vectors are on 2nd axis since its (num_pairs_tri, 3)
    # Output shape left_normals: (num_pairs_tri, 3)
    left_normals = torch.cross(left_positions[:, 1] - left_positions[:, 0],
                               left_positions[:, 2] - left_positions[:, 0], dim=1)
    right_normals = torch.cross(right_positions[:, 1] - right_positions[:, 0],
                                right_positions[:, 2] - right_positions[:, 0], dim=1)
    left_unit = left_normals / torch.clamp(torch.linalg.norm(left_normals, dim=1, keepdim=True), min=EPS)
    right_unit = right_normals / torch.clamp(torch.linalg.norm(right_normals, dim=1, keepdim=True), min=EPS)
    # Obtain cosine of angle between normals: dot product sum(left * right) since left right is elementwise product,
    # and that's = cosine since we normalized normals
    cosine = torch.sum(left_unit * right_unit, dim=1)
    cosine = torch.clamp(cosine, min=-1.0 + EPS, max=1.0 - EPS)
    # Obtain angle (num_edges,)
    angle = torch.arccos(cosine)
    return angle


def max_surface_angle(positions, left_triangles, right_triangles):
    """
    Computes the maximum boundary angle and returns it.
    Used by preprocess.py

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
        Scalar tensor with the maximum angle between two triangles.
    """
    angles = compute_surface_angles_torch(positions, left_triangles, right_triangles)
    return torch.max(angles)

