"""
test_scalarization_clofnet.py

Diagnostic prints for ClofNet-style scalarization (paper Eq. 2–3) on a beam-like lattice.
Uses paper-exact normalized frame (a, b, c) and six inner-product coefficients per directed edge.
No assertions: prints coefficients, differences, and torch.allclose outcomes.
"""

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_float_vec(t):
    """
    Convert a length-3 torch tensor or similar to a 1D float64 numpy vector.

    Args:
        t: torch.Tensor
            Vector of shape [3] or broadcastable.

    return:
        numpy.ndarray
            Shape [3], dtype float64.
    """
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy().astype(np.float64).reshape(3)
    return np.asarray(t, dtype=np.float64).reshape(3)


def _fmt_sci(t):
    """
    Format a scalar or length-3 numeric vector in scientific notation with two fractional digits.

    Args:
        t: torch.Tensor or numpy.ndarray or float
            Scalar or 1D length-3 values.

    return:
        str
            Printable string.
    """
    arr = np.asarray(t, dtype=np.float64).reshape(-1)
    return np.array2string(
        arr,
        separator=" ",
        formatter={"float_kind": lambda x: f"{float(x):.2e}"},
    )


def _fmt_sci_scalar(x):
    """
    Format one float in scientific notation with two fractional digits.

    Args:
        x: float or scalar tensor
            Numeric value.

    return:
        str
            Printable string.
    """
    return f"{float(x):.2e}"


def _wrapped_title(title):
    """
    Wrap a long plot title so it stays inside the figure boundary.

    Args:
        title: str
            Raw title.

    return:
        str
            Wrapped title.
    """
    return "\n".join(wrap(title, width=78))


def _unit_vector(vec_np):
    """
    Normalize a vector for plotting.

    Args:
        vec_np: numpy.ndarray
            Shape [3].

    return:
        numpy.ndarray
            Shape [3], normalized if non-zero.
    """
    norm = float(np.linalg.norm(vec_np))
    if norm < 1e-15:
        return vec_np
    return vec_np / norm


def _show_or_close(fig, interactive):
    """
    Show figures for interactive script runs, otherwise close them for pytest.

    Args:
        fig: figure object
            Figure object.
        interactive: bool
            Whether to show the figure interactively.

    return:
        None
    """
    if interactive:
        plt.ion()
        fig.show()
        plt.pause(0.001)
    else:
        plt.close(fig)


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
    """
    a, b, cvec, coeff_row, coeff_col = frame_and_coeffs(x_centered, row, col, eps)
    print(title)
    print("  a:", _fmt_sci(a))
    print("  b:", _fmt_sci(b))
    print("  c:", _fmt_sci(cvec))
    print("  coeff_row (source in frame of row->col):", _fmt_sci(coeff_row))
    print("  coeff_col (target in frame of row->col):", _fmt_sci(coeff_col))


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
    """
    diff = u - v
    ok = torch.allclose(u, v, rtol=rtol, atol=atol)
    print(label)
    print("  are they equal (torch.allclose)?", bool(ok))
    print("  difference (u - v):", _fmt_sci(diff))


def _bbox_scale(points_np):
    """
    Scalar length from axis-aligned bounding box of Nx3 points (for arrow scale).

    Args:
        points_np: numpy.ndarray
            Shape [N, 3].

    return:
        float
            Max edge extent of the bbox, or 1.0 if degenerate.
    """
    lo = points_np.min(axis=0)
    hi = points_np.max(axis=0)
    ext = float(np.linalg.norm(hi - lo))
    if ext < 1e-15:
        return 1.0
    return ext


def _set_equal_3d_limits(ax, points_np):
    """
    Set equal-ish 3D limits including points and arrow endpoints.

    Args:
        ax: axes object
            3D axes.
        points_np: numpy.ndarray
            Shape [N, 3].

    return:
        None
    """
    lo = points_np.min(axis=0)
    hi = points_np.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * max(float(np.max(hi - lo)), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_nodes_edges_tvec(x_raw, i, j, h, k, t_vec, title, save_path, interactive):
    """
    3D plot: raw nodes i,j,h,k, edges (i,j) vs (h,k), translation arrows for t_vec.

    Args:
        x_raw: torch.Tensor
            Raw positions [N, 3].
        i: int
            Index of node i.
        j: int
            Index of node j.
        h: int
            Index of node h.
        k: int
            Index of node k.
        t_vec: torch.Tensor
            Translation vector shape [3], maps i to h when rigid.
        title: str
            Figure title.
        save_path: pathlib.Path
            PNG output path.
        interactive: bool
            Whether to show the figure interactively.

    return:
        None
    """
    xi = _to_numpy_float_vec(x_raw[i])
    xj = _to_numpy_float_vec(x_raw[j])
    xh = _to_numpy_float_vec(x_raw[h])
    xk = _to_numpy_float_vec(x_raw[k])
    tv = _to_numpy_float_vec(t_vec)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(_wrapped_title(title), fontsize=8, pad=8)
    ax.scatter([xi[0]], [xi[1]], [xi[2]], c="tab:blue", s=60, label="i")
    ax.scatter([xj[0]], [xj[1]], [xj[2]], c="tab:cyan", s=60, label="j")
    ax.scatter([xh[0]], [xh[1]], [xh[2]], c="tab:orange", s=60, label="h")
    ax.scatter([xk[0]], [xk[1]], [xk[2]], c="tab:red", s=60, label="k")
    ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], c="tab:blue", linewidth=2, label="edge (i,j)")
    ax.plot([xh[0], xk[0]], [xh[1], xk[1]], [xh[2], xk[2]], c="tab:orange", linewidth=2, label="edge (h,k)")
    ax.quiver(xi[0], xi[1], xi[2], tv[0], tv[1], tv[2], color="green", arrow_length_ratio=0.15, label="t_vec from i")
    ax.quiver(xj[0], xj[1], xj[2], tv[0], tv[1], tv[2], color="olive", arrow_length_ratio=0.15, label="t_vec from j")
    _set_equal_3d_limits(ax, np.stack([xi, xj, xh, xk, xi + tv, xj + tv], axis=0))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _show_or_close(fig, interactive)


def _plot_nodes_frames(x_centered, i, j, h, k, eps, title, save_path, interactive):
    """
    3D plot: centered nodes and local frames (a,b,c) at i for edge (i,j) and at h for edge (h,k).

    Args:
        x_centered: torch.Tensor
            Centered positions [N, 3].
        i: int
            Source index for first frame.
        j: int
            Target index for first frame.
        h: int
            Source index for second frame.
        k: int
            Target index for second frame.
        eps: float
            Stability epsilon passed to frame_and_coeffs.
        title: str
            Figure title.
        save_path: pathlib.Path
            PNG output path.
        interactive: bool
            Whether to show the figure interactively.

    return:
        None
    """
    a_ij, b_ij, c_ij, _, _ = frame_and_coeffs(x_centered, i, j, eps)
    a_hk, b_hk, c_hk, _, _ = frame_and_coeffs(x_centered, h, k, eps)
    Oi = _to_numpy_float_vec(x_centered[i])
    Oh = _to_numpy_float_vec(x_centered[h])
    Oj = _to_numpy_float_vec(x_centered[j])
    Ok = _to_numpy_float_vec(x_centered[k])
    ai = _unit_vector(_to_numpy_float_vec(a_ij))
    bi = _unit_vector(_to_numpy_float_vec(b_ij))
    ci = _unit_vector(_to_numpy_float_vec(c_ij))
    ah = _unit_vector(_to_numpy_float_vec(a_hk))
    bh = _unit_vector(_to_numpy_float_vec(b_hk))
    ch = _unit_vector(_to_numpy_float_vec(c_hk))
    pts = np.stack([Oi, Oj, Oh, Ok], axis=0)
    scale = 0.12 * _bbox_scale(pts)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(_wrapped_title(title), fontsize=8, pad=8)
    ax.scatter([Oi[0]], [Oi[1]], [Oi[2]], c="tab:blue", s=60, label="i")
    ax.scatter([Oj[0]], [Oj[1]], [Oj[2]], c="tab:cyan", s=60, label="j")
    ax.scatter([Oh[0]], [Oh[1]], [Oh[2]], c="tab:orange", s=60, label="h")
    ax.scatter([Ok[0]], [Ok[1]], [Ok[2]], c="tab:red", s=60, label="k")
    ax.plot([Oi[0], Oj[0]], [Oi[1], Oj[1]], [Oi[2], Oj[2]], c="tab:blue", linewidth=1, linestyle="--")
    ax.plot([Oh[0], Ok[0]], [Oh[1], Ok[1]], [Oh[2], Ok[2]], c="tab:orange", linewidth=1, linestyle="--")
    for n, vec in enumerate((ai, bi, ci)):
        label = "frame (i,j): a,b,c" if n == 0 else None
        vec_scaled = vec * scale
        ax.quiver(Oi[0], Oi[1], Oi[2], vec_scaled[0], vec_scaled[1], vec_scaled[2], color="tab:blue",
                  arrow_length_ratio=0.12, label=label)
    for n, vec in enumerate((ah, bh, ch)):
        label = "frame (h,k): a,b,c" if n == 0 else None
        vec_scaled = vec * scale
        ax.quiver(Oh[0], Oh[1], Oh[2], vec_scaled[0], vec_scaled[1], vec_scaled[2], color="tab:orange",
                  arrow_length_ratio=0.12, label=label)
    endpoints = [Oi, Oj, Oh, Ok]
    endpoints.extend([Oi + vec * scale for vec in (ai, bi, ci)])
    endpoints.extend([Oh + vec * scale for vec in (ah, bh, ch)])
    _set_equal_3d_limits(ax, np.stack(endpoints, axis=0))
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_xlabel("x (centered)")
    ax.set_ylabel("y (centered)")
    ax.set_zlabel("z (centered)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _show_or_close(fig, interactive)


def run_case(title, x_raw, i, j, h, k, t_vec, eps, rtol, atol, output_dir, case_slug, interactive):
    """
    Run one diagnostic case: print four edges, three comparisons, and two 3D figures.

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
        t_vec: torch.Tensor
            Translation x_h - x_i in raw coordinates (diagnostic).
        eps: float
            Norm stability epsilon.
        rtol: float
            allclose rtol.
        atol: float
            allclose atol.
        output_dir: pathlib.Path
            Directory for PNG outputs.
        case_slug: str
            Filename prefix, e.g. case1 or case2.
        interactive: bool
            Whether to show figures interactively.
    """
    output_dir = Path(output_dir)
    x = centralize_positions(x_raw)
    err_map_jk = torch.linalg.norm(x_raw[k] - (x_raw[j] + t_vec))
    print("\n" + "=" * 72)
    print(title)
    print("indices i, j, h, k:", i, j, h, k)
    print("x_raw[i]:", _fmt_sci(x_raw[i]))
    print("x_raw[j]:", _fmt_sci(x_raw[j]))
    print("x_raw[h]:", _fmt_sci(x_raw[h]))
    print("x_raw[k]:", _fmt_sci(x_raw[k]))
    print("t_vec (x_raw[h] - x_raw[i]), maps i -> h:", _fmt_sci(t_vec))
    print("||x_raw[k] - (x_raw[j] + t_vec)|| raw:", _fmt_sci_scalar(err_map_jk))
    seg_ij = torch.linalg.norm(x[i] - x[j])
    seg_hk = torch.linalg.norm(x[h] - x[k])
    print("||x_i - x_j|| (centered):", _fmt_sci_scalar(seg_ij))
    print("||x_h - x_k|| (centered):", _fmt_sci_scalar(seg_hk))
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
    out_a = output_dir / f"{case_slug}_nodes_edges_tvec.png"
    out_b = output_dir / f"{case_slug}_nodes_frames.png"
    _plot_nodes_edges_tvec(x_raw, i, j, h, k, t_vec, title + " - raw nodes, edges, t_vec", out_a, interactive)
    _plot_nodes_frames(x, i, j, h, k, eps, title + " - centered nodes + frames (i,j) and (h,k)", out_b, interactive)
    print("saved:", out_a)
    print("saved:", out_b)


def main(output_dir, interactive):
    """
    Build lattice, run Case 1 (pure translation) and Case 2 (oblique edge directions), print diagnostics, write plots.

    Args:
        output_dir: pathlib.Path
            Directory for PNG outputs (created if missing).
        interactive: bool
            Whether to show figures interactively.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    i = lattice_index(0, 0, 0, nx, ny)
    j = lattice_index(1, 1, 1, nx, ny)
    h = lattice_index(0, 1, 0, nx, ny)
    k = lattice_index(1, 2, 1, nx, ny)
    v1_ij = x_raw[j] - x_raw[i]
    v1_hk = x_raw[k] - x_raw[h]
    print("Case 1 raw edge vectors: (j-i) =", _fmt_sci(v1_ij), "(k-h) =", _fmt_sci(v1_hk))
    t_vec_case1 = x_raw[h] - x_raw[i]
    err_h = torch.linalg.norm(x_raw[h] - (x_raw[i] + t_vec_case1))
    err_k = torch.linalg.norm(x_raw[k] - (x_raw[j] + t_vec_case1))
    run_case(
        "Case 1: (h,k) is a pure translation of (i,j); edges span x,y,z (all deltas nonzero).",
        x_raw,
        i,
        j,
        h,
        k,
        t_vec_case1,
        eps,
        rtol,
        atol,
        output_dir,
        "case1",
        interactive,
    )
    print("sanity Case1: ||x_h - (x_i + t)|| raw:", _fmt_sci_scalar(err_h), "||x_k - (x_j + t)|| raw:",
          _fmt_sci_scalar(err_k))
    i2 = lattice_index(0, 0, 1, nx, ny)
    j2 = lattice_index(1, 1, 2, nx, ny)
    h2 = lattice_index(2, 0, 0, nx, ny)
    k2 = lattice_index(3, 2, 1, nx, ny)
    v_ij = x_raw[j2] - x_raw[i2]
    v_hk = x_raw[k2] - x_raw[h2]
    dot_ij_hk = torch.dot(v_ij, v_hk)
    cross_ij_hk = torch.linalg.norm(torch.linalg.cross(v_ij, v_hk))
    print("Case 2 raw edge vectors: (j-i) =", _fmt_sci(v_ij), "(k-h) =", _fmt_sci(v_hk))
    print("Case 2 dot((j-i), (k-h)) (0 if orthogonal):", _fmt_sci_scalar(dot_ij_hk))
    print("Case 2 ||cross((j-i), (k-h))|| (0 if parallel):", _fmt_sci_scalar(cross_ij_hk))
    t_vec_case2 = x_raw[h2] - x_raw[i2]
    run_case(
        "Case 2: both edges span x,y,z (all deltas nonzero); not parallel and not orthogonal.",
        x_raw,
        i2,
        j2,
        h2,
        k2,
        t_vec_case2,
        eps,
        rtol,
        atol,
        output_dir,
        "case2",
        interactive,
    )
    if interactive:
        print("Interactive plots are open. Close the figure windows to finish the script.")
        plt.show(block=True)


def test_scalarization_diagnostic():
    """
    Pytest entry: run main so `pytest -s` shows prints.
    """
    out = Path.cwd() / "scalarization_clofnet_plots"
    main(out, False)


if __name__ == "__main__":
    main(Path.cwd() / "scalarization_clofnet_plots", True)
