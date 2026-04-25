"""
Diagnostic prints for ClofNet-style scalarization (paper Eq. 2–3) on a beam-like lattice.

Uses paper-exact normalized frame (a, b, c) and six inner-product coefficients per directed edge.
No assertions: prints coefficients, differences, and torch.allclose outcomes.
"""

import torch


def build_beam_lattice(nx, ny, nz, length, width, height):
    """
    Regular grid of vertices filling [0, length] x [0, width] x [0, height].

    Args:
        nx: int
            Number of samples along x (>= 2 for a non-degenerate beam step).
        ny: int
            Number of samples along y.
        nz: int
            Number of samples along z.
        length: float
            Beam extent along x.
        width: float
            Beam extent along y.
        height: float
            Beam extent along z.

    return:
        torch.Tensor
            Positions of shape [N, 3] with N = nx * ny * nz, row-major ix, iy, iz.
    """
    if nx < 2:
        raise ValueError("nx must be at least 2 for lattice edges along x")
    if ny < 2 or nz < 2:
        raise ValueError("ny and nz must be at least 2 for a 3D lattice")
    dx = length / float(nx - 1)
    dy = width / float(ny - 1)
    dz = height / float(nz - 1)
    pts = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                pts.append([ix * dx, iy * dy, iz * dz])
    return torch.tensor(pts, dtype=torch.float64)


def lattice_index(ix, iy, iz, nx, ny):
    """
    Linear index for a node at grid coordinates (ix, iy, iz).

    Args:
        ix: int
            x grid index in [0, nx-1].
        iy: int
            y grid index in [0, ny-1].
        iz: int
            z grid index in [0, nz-1].
        nx: int
            Grid size along x.
        ny: int
            Grid size along y.

    return:
        int
            Flat index consistent with build_beam_lattice iteration order.
    """
    return ix + nx * (iy + ny * iz)


def centralize_positions(pos):
    """
    Subtract the global centroid from all node positions.

    Args:
        pos: torch.Tensor
            Node positions of shape [N, 3].

    return:
        torch.Tensor
            Centered positions of shape [N, 3].
    """
    centroid = pos.mean(dim=0, keepdim=True)
    return pos - centroid


def frame_and_coeffs(x_centered, row, col, eps):
    """
    Paper local frame and scalarization for one directed edge (row -> col).

    Args:
        x_centered: torch.Tensor
            Centered positions [N, 3].
        row: int
            Source node index i (edge i -> j with j = col).
        col: int
            Target node index j.
        eps: float
            Small constant added to norms for numerical stability.

    return:
        tuple
            (a, b, c, coeff_row, coeff_col) each torch.Tensor shape [3].
    """
    x_row = x_centered[row]
    x_col = x_centered[col]
    diff = x_row - x_col
    norm_diff = torch.linalg.norm(diff)
    a = diff / (norm_diff + eps)
    cross = torch.linalg.cross(x_row, x_col)
    norm_cross = torch.linalg.norm(cross)
    b = cross / (norm_cross + eps)
    cvec = torch.linalg.cross(a, b)
    norm_c = torch.linalg.norm(cvec)
    cvec = cvec / (norm_c + eps)
    coeff_row = torch.stack([torch.dot(x_row, a), torch.dot(x_row, b), torch.dot(x_row, cvec)])
    coeff_col = torch.stack([torch.dot(x_col, a), torch.dot(x_col, b), torch.dot(x_col, cvec)])
    return a, b, cvec, coeff_row, coeff_col


def six_coeffs_from_frame(x_centered, row, col, eps):
    """
    Concatenate coeff_row and coeff_col into one length-6 tensor.

    Args:
        x_centered: torch.Tensor
            Centered positions [N, 3].
        row: int
            Edge source index.
        col: int
            Edge target index.
        eps: float
            Stability epsilon for norms.

    return:
        torch.Tensor
            Shape [6]: [coeff_row; coeff_col].
    """
    _a, _b, _c, coeff_row, coeff_col = frame_and_coeffs(x_centered, row, col, eps)
    return torch.cat([coeff_row, coeff_col], dim=0)


def print_edge_block(title, x_centered, row, col, eps):
    """
    Print basis and six coefficients for one directed edge.

    Args:
        title: str
            Label for this edge.
        x_centered: torch.Tensor
            Centered positions [N, 3].
        row: int
            Source node index.
        col: int
            Target node index.
        eps: float
            Stability epsilon for norms.

    return:
        None
    """
    a, b, cvec, coeff_row, coeff_col = frame_and_coeffs(x_centered, row, col, eps)
    print(title)
    print("  a:", a.detach().cpu().numpy())
    print("  b:", b.detach().cpu().numpy())
    print("  c:", cvec.detach().cpu().numpy())
    print("  coeff_row (source in frame of row->col):", coeff_row.detach().cpu().numpy())
    print("  coeff_col (target in frame of row->col):", coeff_col.detach().cpu().numpy())


def print_comparison(label, u, v, rtol, atol):
    """
    Print whether two tensors are allclose and their difference.

    Args:
        label: str
            Description of the comparison.
        u: torch.Tensor
            First tensor (1D).
        v: torch.Tensor
            Second tensor (1D), same shape as u.
        rtol: float
            Relative tolerance for torch.allclose.
        atol: float
            Absolute tolerance for torch.allclose.

    return:
        None
    """
    diff = u - v
    ok = torch.allclose(u, v, rtol=rtol, atol=atol)
    print(label)
    print("  are they equal (torch.allclose)?", bool(ok))
    print("  difference (u - v):", diff.detach().cpu().numpy())


def run_case(title, x_raw, i, j, h, k, eps, rtol, atol):
    """
    Run one diagnostic case: print four edges and three comparisons.

    Args:
        title: str
            Case title string.
        x_raw: torch.Tensor
            Uncentered lattice positions [N, 3].
        i: int
            First endpoint index for reference edge (i, j).
        j: int
            Second endpoint index for reference edge (i, j).
        h: int
            First endpoint index for second edge (h, k).
        k: int
            Second endpoint index for second edge (h, k).
        eps: float
            Norm stability epsilon.
        rtol: float
            allclose rtol.
        atol: float
            allclose atol.

    return:
        None
    """
    x = centralize_positions(x_raw)
    print("=" * 72)
    print(title)
    print("indices i, j, h, k:", i, j, h, k)
    seg_ij = torch.linalg.norm(x[i] - x[j])
    seg_hk = torch.linalg.norm(x[h] - x[k])
    print("||x_i - x_j|| (centered):", float(seg_ij))
    print("||x_h - x_k|| (centered):", float(seg_hk))
    print_edge_block("edge (i -> j)", x, i, j, eps)
    print_edge_block("edge (h -> k)", x, h, k, eps)
    print_edge_block("edge (j -> i)", x, j, i, eps)
    print_edge_block("edge (k -> h)", x, k, h, eps)
    _ai, _bi, _ci, coeff_i_ij, _ = frame_and_coeffs(x, i, j, eps)
    _ah, _bh, _ch, coeff_h_hk, _ = frame_and_coeffs(x, h, k, eps)
    _, _, _, _, coeff_k_hk = frame_and_coeffs(x, h, k, eps)
    _, _, _, coeff_j_ij, _ = frame_and_coeffs(x, i, j, eps)
    six_ji = six_coeffs_from_frame(x, j, i, eps)
    six_kh = six_coeffs_from_frame(x, k, h, eps)
    print("--- comparisons (coefficients are 3-vectors per endpoint in that edge's frame) ---")
    print_comparison("i in frame (i,j) vs h in frame (h,k)", coeff_i_ij, coeff_h_hk, rtol, atol)
    print_comparison("j in frame (i,j) vs k in frame (h,k)", coeff_j_ij, coeff_k_hk, rtol, atol)
    print_comparison("full six coeffs edge (j,i) vs edge (k,h)", six_ji, six_kh, rtol, atol)


def main():
    """
    Build lattice, run translation case and same-length case, print diagnostics.

    Args:
        None

    return:
        None
    """
    eps = 1e-8
    rtol = 1e-5
    atol = 1e-8
    nx = 5
    ny = 5
    nz = 3
    length = 1.0
    width = 1.0
    height = 0.2
    x_raw = build_beam_lattice(nx, ny, nz, length, width, height)
    dx = length / float(nx - 1)
    dy = width / float(ny - 1)
    iy0 = 1
    iz0 = 1
    i = lattice_index(0, iy0, iz0, nx, ny)
    j = lattice_index(1, iy0, iz0, nx, ny)
    h = lattice_index(1, iy0, iz0, nx, ny)
    k = lattice_index(2, iy0, iz0, nx, ny)
    t_vec = x_raw[j] - x_raw[i]
    err_h = torch.linalg.norm(x_raw[h] - (x_raw[i] + t_vec))
    err_k = torch.linalg.norm(x_raw[k] - (x_raw[j] + t_vec))
    run_case("Case 1: (h,k) is a translation of (i,j) by one beam step along x (same mesh, local pair shift).", x_raw, i, j, h, k, eps, rtol, atol)
    print("sanity: ||x_h - (x_i + t)|| raw:", float(err_h), "||x_k - (x_j + t)|| raw:", float(err_k))
    i2 = i
    j2 = j
    h2 = lattice_index(2, 0, 0, nx, ny)
    k2 = lattice_index(2, 1, 0, nx, ny)
    len_ij = torch.linalg.norm(x_raw[i2] - x_raw[j2])
    len_hk = torch.linalg.norm(x_raw[h2] - x_raw[k2])
    if not torch.isclose(len_ij, len_hk, rtol=rtol, atol=atol):
        print("WARNING Case 2 setup: edge lengths differ:", float(len_ij), float(len_hk))
    run_case("Case 2: ||x_h - x_k|| == ||x_i - x_j|| only (different direction and location; y-edge vs x-edge).", x_raw, i2, j2, h2, k2, eps, rtol, atol)


def test_scalarization_diagnostic():
    """
    Pytest entry: run main so `pytest -s` shows prints.

    Args:
        None

    return:
        None
    """
    main()


if __name__ == "__main__":
    main()
