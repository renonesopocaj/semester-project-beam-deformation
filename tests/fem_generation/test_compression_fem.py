"""
test_compression_fem.py

Runnable script (not a unittest): six FEA parameter sweeps with f_x=0, baseline from
conf/fem_generation/fem_rotated_nu.yaml, graphs kept in memory (save patched), and a
multi-panel PNG written under tests/fem_generation/compression_fem_output/.
Run from repo root, FEniCS env recommended:
                     python tests/fem_generation/test_compression_fem.py
"""

from pathlib import Path
from unittest.mock import patch
from src.fem_generation.build_parameter_tuples import _resolve_n_cells
import matplotlib
from src.fem_generation.beam_fea_solver import fea_simulation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from src.fem_generation.build_parameter_tuples import _train_axis_values

# This file is a runnable script, not a pytest/unittest target.
__test__ = False


def _repo_root():
    """
    Repository root directory (parent of package and conf).

    return: Path
        Absolute path to repo root.
    """
    return Path(__file__).resolve().parents[2]


def _mid_value(axis_list):
    """
    Deterministic baseline scalar from a train-axis value list.

    Args:
        axis_list: list
            Ordered values from _train_axis_values for one axis.
    return: float
        Element at index len//2.
    """
    n = len(axis_list)
    return float(axis_list[n // 2])


def _load_fem_rotated_nu_config():
    """
    Load fem_rotated_nu.yaml from conf/fem_generation.

    return: dict
        Parsed YAML config.
    """
    cfg_path = _repo_root() / "conf" / "fem_generation" / "fem_rotated_nu.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _baseline_combo_from_cfg(cfg):
    """
    Build baseline (fx, fy, fz), L, W, D and defaults dict from config.
    f_x is always zero for this study (no axial traction on the free end).

    Args:
        cfg: dict
            Parsed fem_rotated_nu.yaml.
    return: tuple
        (fx, fy, fz, L, W, D, defaults_dict)
    """

    defaults = cfg["defaults"]
    _fx_list, fy_list, fz_list, lengths, widths, depths = _train_axis_values(cfg)
    fx = 0.0
    fy = _mid_value(fy_list)
    fz = _mid_value(fz_list)
    L = _mid_value(lengths)
    W = _mid_value(widths)
    D = _mid_value(depths)
    return fx, fy, fz, L, W, D, defaults


def _resolved_mesh_counts(defaults, L, W, D):
    """
    Resolve NL, NW, ND from defaults specs and current geometry.

    Args:
        defaults: dict
            Config defaults including NL, NW, ND.
        L: float
            Beam length.
        W: float
            Beam width.
        D: float
            Beam depth.
    return: tuple
        (NL, NW, ND) as ints.
    """


    NL = _resolve_n_cells(defaults["NL"], L, "NL")
    NW = _resolve_n_cells(defaults["NW"], W, "NW")
    ND = _resolve_n_cells(defaults["ND"], D, "ND")
    return NL, NW, ND


def _edge_stretch_stats(lst_graph_tstep):
    """
    Edge stretch ratio ||x_i(t)-x_j(t)|| / ||x_i(0)-x_j(0)|| from graph positions and edge_index.

    Args:
        lst_graph_tstep: list
            PyG Data list; t=0 is rest geometry; edge_index shared across steps.
    return: tuple
        (mean_t_mean_e_stretch, max_t_mean_e_stretch, max_t_max_e_stretch)
        mean_t_mean_e: mean over time of per-step mean over directed edges;
        max_t_mean_e: max over time of per-step mean over edges;
        max_t_max_e: max over time of per-step max over edges.
    """
    eps_len = 1e-12
    ei0 = lst_graph_tstep[0].edge_index
    i = ei0[0].detach().cpu().numpy().astype(np.int64)
    j = ei0[1].detach().cpu().numpy().astype(np.int64)
    P0 = lst_graph_tstep[0].x.detach().cpu().numpy()
    diff0 = P0[i] - P0[j]
    len0 = np.linalg.norm(diff0, axis=1)
    denom = np.maximum(len0, eps_len)
    m_t = []
    M_t = []
    for k in range(len(lst_graph_tstep)):
        ei = lst_graph_tstep[k].edge_index
        if not torch.equal(ei, ei0):
            raise ValueError("edge_index must be identical across time steps for stretch stats")
        Pk = lst_graph_tstep[k].x.detach().cpu().numpy()
        diffk = Pk[i] - Pk[j]
        lenk = np.linalg.norm(diffk, axis=1)
        ratio = lenk / denom
        m_t.append(float(np.mean(ratio)))
        M_t.append(float(np.max(ratio)))
    arr_m = np.asarray(m_t, dtype=float)
    arr_M = np.asarray(M_t, dtype=float)
    return float(np.mean(arr_m)), float(np.max(arr_m)), float(np.max(arr_M))


def _edge_yz_component_max_max_stretch(lst_graph_tstep):
    """
    Per directed edge use component-wise stretch |Δq|/|Δq0| for q in {y,z} (indices 1 and 2).

    At each time: M_y(t)=max_e ratio_y, M_z(t)=max_e ratio_z. Return max_t M_y and max_t M_z.

    Args:
        lst_graph_tstep: list
            PyG Data list; t=0 rest; same edge_index as _edge_stretch_stats.
    return: tuple
        (max_t_max_e_stretch_y, max_t_max_e_stretch_z)
    """
    import torch

    eps = 1e-12
    ei0 = lst_graph_tstep[0].edge_index
    i = ei0[0].detach().cpu().numpy().astype(np.int64)
    j = ei0[1].detach().cpu().numpy().astype(np.int64)
    P0 = lst_graph_tstep[0].x.detach().cpu().numpy()
    diff0 = P0[i] - P0[j]
    den_y = np.maximum(np.abs(diff0[:, 1]), eps)
    den_z = np.maximum(np.abs(diff0[:, 2]), eps)
    My_t = []
    Mz_t = []
    for k in range(len(lst_graph_tstep)):
        ei = lst_graph_tstep[k].edge_index
        if not torch.equal(ei, ei0):
            raise ValueError("edge_index must be identical across time steps for yz stretch stats")
        Pk = lst_graph_tstep[k].x.detach().cpu().numpy()
        diffk = Pk[i] - Pk[j]
        rat_y = np.abs(diffk[:, 1]) / den_y
        rat_z = np.abs(diffk[:, 2]) / den_z
        My_t.append(float(np.max(rat_y)))
        Mz_t.append(float(np.max(rat_z)))
    return float(np.max(np.asarray(My_t, dtype=float))), float(np.max(np.asarray(Mz_t, dtype=float)))


def _fixed_params_caption(sweep_key, fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis):
    """
    Multi-line caption of simulation inputs fixed for this sweep (swept axis omitted).

    Args:
        sweep_key: str
            One of D, L, W, f_y, f_z, nu.
        fx0: float
            Fixed f_x (always 0 in this script).
        fy0: float
            Baseline f_y when not swept.
        fz0: float
            Baseline f_z when not swept.
        L0: float
            Baseline L when not swept.
        W0: float
            Baseline W when not swept.
        D0: float
            Baseline D when not swept.
        nu0: float
            Baseline nu when not swept.
        defaults: dict
            YAML defaults.
        geom_axis: ndarray
            First value used for representative NL,NW,ND when L, W, or D is swept.
    return: str
        Text for an axes text box.
    """
    g0 = float(geom_axis[0])
    if sweep_key == "D":
        Lm, Wm, Dm = L0, W0, g0
    elif sweep_key == "L":
        Lm, Wm, Dm = g0, W0, D0
    elif sweep_key == "W":
        Lm, Wm, Dm = L0, g0, D0
    else:
        Lm, Wm, Dm = L0, W0, D0
    NL, NW, ND = _resolved_mesh_counts(defaults, Lm, Wm, Dm)
    lines = []
    lines.append("fixed params (not swept):")
    lines.append("f_x=%.6g" % fx0)
    if sweep_key != "f_y":
        lines.append("f_y=%.6g" % fy0)
    if sweep_key != "f_z":
        lines.append("f_z=%.6g" % fz0)
    if sweep_key != "L":
        lines.append("L=%.6g" % L0)
    if sweep_key != "W":
        lines.append("W=%.6g" % W0)
    if sweep_key != "D":
        lines.append("D=%.6g" % D0)
    if sweep_key != "nu":
        lines.append("nu=%.6g" % nu0)
    lines.append(
        "E=%.6g rho=%.6g eta_m=%.6g eta_k=%.6g"
        % (defaults["E"], defaults["rho"], defaults["eta_m"], defaults["eta_k"])
    )
    lines.append(
        "alpha_m=%.6g alpha_f=%.6g"
        % (defaults["alpha_m"], defaults["alpha_f"])
    )
    lines.append(
        "T=%.6g Nsteps=%d Tc_factor=%.6g solver=%s"
        % (
            defaults["total_time"],
            int(defaults["num_steps"]),
            defaults["cutoff_time_factor"],
            defaults["linear_solver"],
        )
    )
    lines.append("mesh NL,NW,ND=%d,%d,%d (ref L,W,D=%.4g,%.4g,%.4g)" % (NL, NW, ND, Lm, Wm, Dm))
    return "\n".join(lines)


def _noop_save_graphs_and_plot(*args):
    """
    Stub for beam_fea_solver._save_graphs_and_plot: no disk output.

    return: tuple
        Dummy (save_path_graph, simulation_name).
    """
    return "/tmp", "no_save"


def _run_one_simulation(L, W, D, fx, fy, fz, nu, defaults, output_root, fea_simulation_fn):
    """
    Run a single fea_simulation with saving patched out.

    Args:
        L: float
            Beam length.
        W: float
            Beam width.
        D: float
            Beam depth.
        fx: float
            Peak x traction.
        fy: float
            Peak y traction.
        fz: float
            Peak z traction.
        nu: float
            Poisson ratio for elastic_params.
        defaults: dict
            YAML defaults (E, rho, damping, time, mesh specs, etc.).
        output_root: str
            Passed through to fea_simulation (unused when save is mocked).
        fea_simulation_fn: callable
            fea_simulation bound or imported function.
    return: list
        lst_graph_tstep from the solver.
    """
    NL, NW, ND = _resolved_mesh_counts(defaults, L, W, D)
    elastic_params = (defaults["E"], nu)
    damping_params = (defaults["eta_m"], defaults["eta_k"])
    newmark_params = (defaults["alpha_m"], defaults["alpha_f"])
    with patch("src.fem_generation.beam_fea_solver._save_graphs_and_plot", _noop_save_graphs_and_plot):
        lst, _, _ = fea_simulation_fn(
            L=L,
            W=W,
            D=D,
            NL=NL,
            NW=NW,
            ND=ND,
            elastic_params=elastic_params,
            density=defaults["rho"],
            damping_params=damping_params,
            newmark_params=newmark_params,
            initial_force_x=fx,
            initial_force_y=fy,
            initial_force_z=fz,
            cutoff_time_factor=defaults["cutoff_time_factor"],
            total_time=defaults["total_time"],
            num_steps=defaults["num_steps"],
            mode="compression_test",
            output_root=output_root,
            verbose=False,
            first_trajectory_debug=False,
            linear_solver=defaults["linear_solver"],
        )
    return lst


def run_compression_sweeps_and_plot(output_dir):
    """
    Run 60 FEA simulations (six sweeps of ten points), build edge stretch curves, save two PNGs.

    Args:
        output_dir: Path
            Directory to create and write compression_sweeps.png and compression_sweeps_yz_max_max.png into.
    return: tuple
        (path_3d_stretch_png, path_yz_component_png)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_fem_rotated_nu_config()
    fx0, fy0, fz0, L0, W0, D0, defaults = _baseline_combo_from_cfg(cfg)
    nu0 = float(defaults["nu"])

    npts = 10
    geom_axis = np.linspace(0.01, 2.0, npts)
    force_axis = np.linspace(0.0, 2.0, npts)
    nu_axis = np.linspace(0.0001, 0.45, npts)

    sweeps = []
    run_idx = 0
    total_runs = 6 * npts

    # 1) fixed f_y, f_z, W, L, nu — vary D
    mean_mean_e = np.zeros(npts)
    max_mean_e = np.zeros(npts)
    max_max_e = np.zeros(npts)
    yz_max_y = np.zeros(npts)
    yz_max_z = np.zeros(npts)
    for i, Dv in enumerate(geom_axis):
        run_idx += 1
        print("run %d/%d  sweep=D  D=%s" % (run_idx, total_runs, Dv))
        lst = _run_one_simulation(L0, W0, Dv, fx0, fy0, fz0, nu0, defaults, "/tmp", fea_simulation)
        mean_mean_e[i], max_mean_e[i], max_max_e[i] = _edge_stretch_stats(lst)
        yz_max_y[i], yz_max_z[i] = _edge_yz_component_max_max_stretch(lst)
    cap = _fixed_params_caption("D", fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis)
    sweeps.append(
        ("D", geom_axis, mean_mean_e.copy(), max_mean_e.copy(), max_max_e.copy(), cap, yz_max_y.copy(), yz_max_z.copy())
    )

    # 2) fixed f_y, f_z, W, D, nu — vary L
    mean_mean_e = np.zeros(npts)
    max_mean_e = np.zeros(npts)
    max_max_e = np.zeros(npts)
    yz_max_y = np.zeros(npts)
    yz_max_z = np.zeros(npts)
    for i, Lv in enumerate(geom_axis):
        run_idx += 1
        print("run %d/%d  sweep=L  L=%s" % (run_idx, total_runs, Lv))
        lst = _run_one_simulation(Lv, W0, D0, fx0, fy0, fz0, nu0, defaults, "/tmp", fea_simulation)
        mean_mean_e[i], max_mean_e[i], max_max_e[i] = _edge_stretch_stats(lst)
        yz_max_y[i], yz_max_z[i] = _edge_yz_component_max_max_stretch(lst)
    cap = _fixed_params_caption("L", fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis)
    sweeps.append(
        ("L", geom_axis, mean_mean_e.copy(), max_mean_e.copy(), max_max_e.copy(), cap, yz_max_y.copy(), yz_max_z.copy())
    )

    # 3) fixed f_y, f_z, L, D, nu — vary W
    mean_mean_e = np.zeros(npts)
    max_mean_e = np.zeros(npts)
    max_max_e = np.zeros(npts)
    yz_max_y = np.zeros(npts)
    yz_max_z = np.zeros(npts)
    for i, Wv in enumerate(geom_axis):
        run_idx += 1
        print("run %d/%d  sweep=W  W=%s" % (run_idx, total_runs, Wv))
        lst = _run_one_simulation(L0, Wv, D0, fx0, fy0, fz0, nu0, defaults, "/tmp", fea_simulation)
        mean_mean_e[i], max_mean_e[i], max_max_e[i] = _edge_stretch_stats(lst)
        yz_max_y[i], yz_max_z[i] = _edge_yz_component_max_max_stretch(lst)
    cap = _fixed_params_caption("W", fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis)
    sweeps.append(
        ("W", geom_axis, mean_mean_e.copy(), max_mean_e.copy(), max_max_e.copy(), cap, yz_max_y.copy(), yz_max_z.copy())
    )

    # 4) fixed f_z, W, D, L, nu — vary f_y
    mean_mean_e = np.zeros(npts)
    max_mean_e = np.zeros(npts)
    max_max_e = np.zeros(npts)
    yz_max_y = np.zeros(npts)
    yz_max_z = np.zeros(npts)
    for i, fyv in enumerate(force_axis):
        run_idx += 1
        print("run %d/%d  sweep=f_y  f_y=%s" % (run_idx, total_runs, fyv))
        lst = _run_one_simulation(L0, W0, D0, fx0, fyv, fz0, nu0, defaults, "/tmp", fea_simulation)
        mean_mean_e[i], max_mean_e[i], max_max_e[i] = _edge_stretch_stats(lst)
        yz_max_y[i], yz_max_z[i] = _edge_yz_component_max_max_stretch(lst)
    cap = _fixed_params_caption("f_y", fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis)
    sweeps.append(
        ("f_y", force_axis, mean_mean_e.copy(), max_mean_e.copy(), max_max_e.copy(), cap, yz_max_y.copy(), yz_max_z.copy())
    )

    # 5) fixed f_y, W, D, L, nu — vary f_z
    mean_mean_e = np.zeros(npts)
    max_mean_e = np.zeros(npts)
    max_max_e = np.zeros(npts)
    yz_max_y = np.zeros(npts)
    yz_max_z = np.zeros(npts)
    for i, fzv in enumerate(force_axis):
        run_idx += 1
        print("run %d/%d  sweep=f_z  f_z=%s" % (run_idx, total_runs, fzv))
        lst = _run_one_simulation(L0, W0, D0, fx0, fy0, fzv, nu0, defaults, "/tmp", fea_simulation)
        mean_mean_e[i], max_mean_e[i], max_max_e[i] = _edge_stretch_stats(lst)
        yz_max_y[i], yz_max_z[i] = _edge_yz_component_max_max_stretch(lst)
    cap = _fixed_params_caption("f_z", fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis)
    sweeps.append(
        ("f_z", force_axis, mean_mean_e.copy(), max_mean_e.copy(), max_max_e.copy(), cap, yz_max_y.copy(), yz_max_z.copy())
    )

    # 6) fixed f_y, W, D, L, f_z — vary nu
    mean_mean_e = np.zeros(npts)
    max_mean_e = np.zeros(npts)
    max_max_e = np.zeros(npts)
    yz_max_y = np.zeros(npts)
    yz_max_z = np.zeros(npts)
    for i, nuv in enumerate(nu_axis):
        run_idx += 1
        print("run %d/%d  sweep=nu  nu=%s" % (run_idx, total_runs, nuv))
        lst = _run_one_simulation(L0, W0, D0, fx0, fy0, fz0, nuv, defaults, "/tmp", fea_simulation)
        mean_mean_e[i], max_mean_e[i], max_max_e[i] = _edge_stretch_stats(lst)
        yz_max_y[i], yz_max_z[i] = _edge_yz_component_max_max_stretch(lst)
    cap = _fixed_params_caption("nu", fx0, fy0, fz0, L0, W0, D0, nu0, defaults, geom_axis)
    sweeps.append(
        ("nu", nu_axis, mean_mean_e.copy(), max_mean_e.copy(), max_max_e.copy(), cap, yz_max_y.copy(), yz_max_z.copy())
    )

    for name, xs, y1, y2, y3, cap, yz_y, yz_z in sweeps:
        if not np.all(np.isfinite(y1)) or not np.all(np.isfinite(y2)) or not np.all(np.isfinite(y3)):
            print("warning: non-finite stretch metrics for sweep %s" % name)
        if not np.all(np.isfinite(yz_y)) or not np.all(np.isfinite(yz_z)):
            print("warning: non-finite yz component stretch for sweep %s" % name)

    fig, axes = plt.subplots(6, 1, figsize=(11, 22), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.35, w_pad=0.05)
    for ax, (name, xs, y1, y2, y3, cap, yz_y, yz_z) in zip(axes, sweeps):
        ax.plot(xs, y1, label="mean_t mean_e stretch", zorder=2)
        ax.plot(xs, y2, label="max_t mean_e stretch", zorder=2)
        ax.plot(xs, y3, label="max_t max_e stretch", zorder=2)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("edge stretch ratio", fontsize=11)
        ax.set_title("sweep: %s" % name, fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.35, zorder=1)
        ax.text(
            0.02,
            0.98,
            cap,
            transform=ax.transAxes,
            fontsize=7,
            family="monospace",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.5", alpha=0.9),
            zorder=5,
        )

    out_path = output_dir / "compression_sweeps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(6, 1, figsize=(11, 22), constrained_layout=True)
    fig2.set_constrained_layout_pads(h_pad=0.35, w_pad=0.05)
    for ax, (name, xs, y1, y2, y3, cap, yz_y, yz_z) in zip(axes2, sweeps):
        ax.plot(xs, yz_y, label="max_t max_e |Δy|/|Δy0|", zorder=2)
        ax.plot(xs, yz_z, label="max_t max_e |Δz|/|Δz0|", zorder=2)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("component edge stretch", fontsize=11)
        ax.set_title("sweep: %s (Y/Z component max)" % name, fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.35, zorder=1)
        ax.text(
            0.02,
            0.98,
            cap,
            transform=ax.transAxes,
            fontsize=7,
            family="monospace",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.5", alpha=0.9),
            zorder=5,
        )

    out_path_yz = output_dir / "compression_sweeps_yz_max_max.png"
    fig2.savefig(out_path_yz, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    return out_path, out_path_yz


def main():
    """
    Default entry: write plots next to this file under compression_fem_output/.

    return: None
    """
    default_out = "compression_fem_output"
    path_3d, path_yz = run_compression_sweeps_and_plot(default_out)
    print("Wrote %s" % (path_3d.resolve(),))
    print("Wrote %s" % (path_yz.resolve(),))


if __name__ == "__main__":
    main()
