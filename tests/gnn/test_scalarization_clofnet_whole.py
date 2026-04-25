"""
Diagnostic prints for whole-mesh ClofNet scalarization invariance under SE(3).

Two meshes are used: one beam-like lattice, and one obtained by rotating the
whole mesh and then translating it. After separate centroid centralization, all
non-degenerate directed-edge scalarization coefficients should match.
"""

import torch


def build_beam_lattice(nx, ny, nz, length, width, height):
    """
    Regular grid of vertices filling a 3D beam parallelepiped.

    Args:
        nx: int
            Number of samples along the beam length.
        ny: int
            Number of samples along the beam width.
        nz: int
            Number of samples along the beam height.
        length: float
            Beam extent along x.
        width: float
            Beam extent along y.
        height: float
            Beam extent along z.

    return:
        torch.Tensor
            Positions of shape [N, 3].
    """
    xs = torch.linspace(0.0, length, nx, dtype=torch.float64)
    ys = torch.linspace(0.0, width, ny, dtype=torch.float64)
    zs = torch.linspace(0.0, height, nz, dtype=torch.float64)
    pts = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                pts.append([xs[ix], ys[iy], zs[iz]])
    return torch.tensor(pts, dtype=torch.float64)


def lattice_index(ix, iy, iz, nx, ny):
    """
    Linear index for a node at grid coordinates (ix, iy, iz).

    Args:
        ix: int
            x grid index.
        iy: int
            y grid index.
        iz: int
            z grid index.
        nx: int
            Grid size along x.
        ny: int
            Grid size along y.

    return:
        int
            Flat index consistent with build_beam_lattice iteration order.
    """
    return ix + nx * (iy + ny * iz)


def add_undirected_edge(edge_set, a, b):
    """
    Add one undirected edge to a set.

    Args:
        edge_set: set
            Set of ordered node-index tuples with smaller index first.
        a: int
            First node index.
        b: int
            Second node index.

    return:
        None
    """
    if a == b:
        return
    edge_set.add((min(a, b), max(a, b)))


def build_triangulation_like_edges(nx, ny, nz):
    """
    Build a simple tetra/face-edge-style undirected connectivity for the lattice.

    Args:
        nx: int
            Grid size along x.
        ny: int
            Grid size along y.
        nz: int
            Grid size along z.

    return:
        list
            Undirected edges as (source, target) node-index tuples.
    """
    edge_set = set()
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                src = lattice_index(ix, iy, iz, nx, ny)
                if ix + 1 < nx:
                    add_undirected_edge(edge_set, src, lattice_index(ix + 1, iy, iz, nx, ny))
                if iy + 1 < ny:
                    add_undirected_edge(edge_set, src, lattice_index(ix, iy + 1, iz, nx, ny))
                if iz + 1 < nz:
                    add_undirected_edge(edge_set, src, lattice_index(ix, iy, iz + 1, nx, ny))
                if ix + 1 < nx and iy + 1 < ny:
                    add_undirected_edge(edge_set, src, lattice_index(ix + 1, iy + 1, iz, nx, ny))
                if ix + 1 < nx and iz + 1 < nz:
                    add_undirected_edge(edge_set, src, lattice_index(ix + 1, iy, iz + 1, nx, ny))
                if iy + 1 < ny and iz + 1 < nz:
                    add_undirected_edge(edge_set, src, lattice_index(ix, iy + 1, iz + 1, nx, ny))
    return sorted(edge_set)


def directed_edges_from_undirected(undirected_edges):
    """
    Duplicate undirected edges into both directed orders.

    Args:
        undirected_edges: list
            Edge list with one tuple per undirected edge.

    return:
        list
            Directed edge tuples.
    """
    directed_edges = []
    for row, col in undirected_edges:
        directed_edges.append((row, col))
        directed_edges.append((col, row))
    return directed_edges


def centralize_positions(pos):
    """
    Subtract the global centroid from all node positions.

    Args:
        pos: torch.Tensor
            Node positions of shape [N, 3].

    return:
        torch.Tensor
            Centered node positions of shape [N, 3].
    """
    return pos - pos.mean(dim=0, keepdim=True)


def rotation_matrix():
    """
    Construct a deterministic proper 3D rotation matrix.

    Args:
        None

    return:
        torch.Tensor
            Rotation matrix of shape [3, 3] with determinant 1.
    """
    angle_z = torch.tensor(0.73, dtype=torch.float64)
    angle_y = torch.tensor(-0.41, dtype=torch.float64)
    cz = torch.cos(angle_z)
    sz = torch.sin(angle_z)
    cy = torch.cos(angle_y)
    sy = torch.sin(angle_y)
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=torch.float64)
    return rz @ ry


def transform_mesh(pos, rot, translation):
    """
    Rotate the whole mesh and then translate it.

    Args:
        pos: torch.Tensor
            Original positions [N, 3].
        rot: torch.Tensor
            Rotation matrix [3, 3].
        translation: torch.Tensor
            Translation vector [3].

    return:
        torch.Tensor
            Transformed positions [N, 3].
    """
    return pos @ rot.T + translation


def frame_and_coeffs(x_centered, row, col, eps):
    """
    Paper local frame and six scalarization coefficients for one directed edge.

    Args:
        x_centered: torch.Tensor
            Centered positions [N, 3].
        row: int
            Source node index.
        col: int
            Target node index.
        eps: float
            Small value used in norm denominators.

    return:
        tuple
            (coeffs, cross_norm), with coeffs shape [6].
    """
    x_row = x_centered[row]
    x_col = x_centered[col]
    a = x_row - x_col
    a = a / (torch.linalg.norm(a) + eps)
    cross = torch.linalg.cross(x_row, x_col)
    cross_norm = torch.linalg.norm(cross)
    b = cross / (cross_norm + eps)
    cvec = torch.linalg.cross(a, b)
    cvec = cvec / (torch.linalg.norm(cvec) + eps)
    coeff_row = torch.stack([torch.dot(x_row, a), torch.dot(x_row, b), torch.dot(x_row, cvec)])
    coeff_col = torch.stack([torch.dot(x_col, a), torch.dot(x_col, b), torch.dot(x_col, cvec)])
    return torch.cat([coeff_row, coeff_col], dim=0), cross_norm


def scalarize_all_edges(x_centered, directed_edges, eps, degenerate_tol):
    """
    Scalarize all non-degenerate directed edges.

    Args:
        x_centered: torch.Tensor
            Centered positions [N, 3].
        directed_edges: list
            Directed edge list.
        eps: float
            Small norm denominator value.
        degenerate_tol: float
            Minimum cross-product norm required for a valid frame.

    return:
        tuple
            (kept_edges, coeffs, skipped_edges) where coeffs has shape [E_kept, 6].
    """
    kept_edges = []
    skipped_edges = []
    coeff_list = []
    for row, col in directed_edges:
        coeffs, cross_norm = frame_and_coeffs(x_centered, row, col, eps)
        if cross_norm <= degenerate_tol:
            skipped_edges.append((row, col, float(cross_norm)))
        else:
            kept_edges.append((row, col))
            coeff_list.append(coeffs)
    return kept_edges, torch.stack(coeff_list, dim=0), skipped_edges


def print_summary(coeffs_a, coeffs_b, kept_edges, skipped_edges, rtol, atol):
    """
    Print allclose result and coefficient differences.

    Args:
        coeffs_a: torch.Tensor
            Original-mesh scalarization coefficients [E, 6].
        coeffs_b: torch.Tensor
            Transformed-mesh scalarization coefficients [E, 6].
        kept_edges: list
            Directed edges used in the comparison.
        skipped_edges: list
            Directed edges skipped because the frame is degenerate.
        rtol: float
            Relative tolerance for torch.allclose.
        atol: float
            Absolute tolerance for torch.allclose.

    return:
        None
    """
    diff = coeffs_a - coeffs_b
    abs_diff = torch.abs(diff)
    same = torch.allclose(coeffs_a, coeffs_b, rtol=rtol, atol=atol)
    max_value, max_flat_idx = torch.max(abs_diff.reshape(-1), dim=0)
    edge_idx = int(max_flat_idx // diff.shape[1])
    coeff_idx = int(max_flat_idx % diff.shape[1])
    print("=" * 72)
    print("Whole-mesh scalarization coefficients invariant after rotation + translation?")
    print("final result (all coefficients same within tolerance):", bool(same))
    print("kept directed edges:", len(kept_edges))
    print("skipped degenerate directed edges:", len(skipped_edges))
    print("max absolute difference:", float(max_value))
    print("max difference edge:", kept_edges[edge_idx])
    print("max difference coefficient index [0:3 row, 3:6 col]:", coeff_idx)
    print("difference at max:", float(diff[edge_idx, coeff_idx]))
    print("mean absolute difference:", float(abs_diff.mean()))
    print("first 10 per-edge max absolute differences:")
    print(torch.max(abs_diff[:10], dim=1).values.detach().cpu().numpy())
    print("first 5 coefficient rows original:")
    print(coeffs_a[:5].detach().cpu().numpy())
    print("first 5 coefficient rows transformed:")
    print(coeffs_b[:5].detach().cpu().numpy())
    print("first 5 coefficient-row differences original - transformed:")
    print(diff[:5].detach().cpu().numpy())


def main():
    """
    Build the two meshes, scalarize all valid directed edges, and print diagnostics.

    Args:
        None

    return:
        None
    """
    eps = 1e-8
    degenerate_tol = 1e-10
    rtol = 1e-5
    atol = 1e-8
    nx = 4
    ny = 4
    nz = 4
    pos = build_beam_lattice(nx, ny, nz, 1.0, 0.4, 0.3)
    undirected_edges = build_triangulation_like_edges(nx, ny, nz)
    directed_edges = directed_edges_from_undirected(undirected_edges)
    rot = rotation_matrix()
    translation = torch.tensor([3.4, -1.7, 0.9], dtype=torch.float64)
    transformed_pos = transform_mesh(pos, rot, translation)
    x_centered = centralize_positions(pos)
    transformed_centered = centralize_positions(transformed_pos)
    kept_edges, coeffs_a, skipped_edges = scalarize_all_edges(x_centered, directed_edges, eps, degenerate_tol)
    transformed_kept_edges, coeffs_b, transformed_skipped_edges = scalarize_all_edges(transformed_centered, directed_edges, eps, degenerate_tol)
    if kept_edges != transformed_kept_edges:
        print("WARNING: kept edge lists differ after transformation")
    if len(skipped_edges) != len(transformed_skipped_edges):
        print("WARNING: skipped edge counts differ after transformation")
    print("rotation determinant:", float(torch.linalg.det(rot)))
    print("translation vector:", translation.detach().cpu().numpy())
    print_summary(coeffs_a, coeffs_b, kept_edges, skipped_edges, rtol, atol)


def test_scalarization_whole_mesh_diagnostic():
    """
    Pytest entry: run main so `pytest -s` shows the diagnostic prints.

    Args:
        None

    return:
        None
    """
    main()


if __name__ == "__main__":
    main()
