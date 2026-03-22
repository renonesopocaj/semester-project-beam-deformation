from collections import defaultdict, deque
import torch
import numpy as np
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import extract_surface_faces

EPS = 1e9

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


def compute_boundary_dihedral_angles_torch(positions, left_triangles, right_triangles):
    """

    Args:
        positions: torch.Tensor
            Node coordinates with shape `(num_nodes, 3)`.
        left_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
        right_triangles: torch.Tensor
            Oriented triangle indices with shape `(3, num_pairs)`.
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
    left_unit = left_normals / torch.clamp(torch.linalg.norm(left_normals, dim=1, keepdim=True), min=EPS)
    right_unit = right_normals / torch.clamp(torch.linalg.norm(right_normals, dim=1, keepdim=True), min=EPS)
    cosine = torch.sum(left_unit * right_unit, dim=1)
    cosine = torch.clamp(cosine, min=-1.0 + EPS, max=1.0 - EPS)
    return torch.arccos(cosine)


def compute_max_boundary_dihedral_angle_torch(positions, left_triangles, right_triangles):
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
    angles = compute_boundary_dihedral_angles_torch(positions, left_triangles, right_triangles)
    return torch.max(angles)

