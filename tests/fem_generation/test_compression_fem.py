"""
test_compression_fem.py

Integration-style checks for axial (x) response of the clamped beam with f_x=0: six parameter sweeps
read baseline settings from conf/fem_generation/fem_rotated_nu.yaml, run FEA in memory
(saving disabled via mock), and plot mean / max spatial-mean x-displacement vs each swept parameter.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from src.fem_generation.build_parameter_tuples import _train_axis_values, _resolve_n_cells
import dolfin as df
from src.fem_generation.beam_fea_solver import fea_simulation

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


def _x_displacement_stats(lst_graph_tstep):
    """
    Mean over time of spatial-mean x displacement, and max over time of spatial-mean x displacement.

    Args:
        lst_graph_tstep: list
            PyG Data list from fea_simulation; index 0 is undeformed reference positions in x.
    return: tuple
        (mean_t_spatial_mean_ux, max_t_spatial_mean_ux)
    """
    x_ref = lst_graph_tstep[0].x[:, 0].detach().cpu().numpy()
    m_t = []
    for k in range(len(lst_graph_tstep)):
        xt = lst_graph_tstep[k].x[:, 0].detach().cpu().numpy()
        m_t.append(float(np.mean(xt - x_ref)))
    arr = np.asarray(m_t, dtype=float)
    return float(np.mean(arr)), float(np.max(arr))


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
        lst, _, _ = fea_simulation_fn(L=L, W=W, D=D, NL=NL, NW=NW, ND=ND,
                                      elastic_params=elastic_params, density=defaults["rho"],
                                      damping_params=damping_params,
                                      newmark_params=newmark_params,
                                      initial_force_x=fx, initial_force_y=fy, initial_force_z=fz,
                                      cutoff_time_factor=defaults["cutoff_time_factor"],
                                      total_time=defaults["total_time"], num_steps=defaults["num_steps"],
                                      mode="compression_test", output_root=output_root, verbose=False,
                                      first_trajectory_debug=False, linear_solver=defaults["linear_solver"])
    return lst


class TestCompressionFem(unittest.TestCase):
    """
    Heavy integration test: 60 transient beam simulations (6 sweeps x 10 points) and a diagnostic figure.
    """

    def test_compression_parameter_sweeps_and_plot(self):
        """
        Run six sweep families (10 points each) with f_x=0 throughout; x-displacement statistics; write a PNG.

        return: None
            Asserts metrics are finite and output image exists.
        """
        cfg = _load_fem_rotated_nu_config()
        fx0, fy0, fz0, L0, W0, D0, defaults = _baseline_combo_from_cfg(cfg)
        self.assertEqual(fx0, 0.0)
        nu0 = float(defaults["nu"])

        npts = 10
        geom_axis = np.linspace(0.01, 2.0, npts)
        force_axis = np.linspace(0.0, 2.0, npts)
        nu_axis = np.linspace(0.0001, 0.45, npts)

        sweeps = []

        # 1) fixed f_y, f_z, W, L, nu — vary D
        mean_a = np.zeros(npts)
        max_a = np.zeros(npts)
        for i, Dv in enumerate(geom_axis):
            lst = _run_one_simulation(L0, W0, Dv, fx0, fy0, fz0, nu0, defaults, "/tmp", fea_simulation)
            mean_a[i], max_a[i] = _x_displacement_stats(lst)
        sweeps.append(("D", geom_axis, mean_a.copy(), max_a.copy()))

        # 2) fixed f_y, f_z, W, D, nu — vary L
        mean_a = np.zeros(npts)
        max_a = np.zeros(npts)
        for i, Lv in enumerate(geom_axis):
            lst = _run_one_simulation(Lv, W0, D0, fx0, fy0, fz0, nu0, defaults, "/tmp", fea_simulation)
            mean_a[i], max_a[i] = _x_displacement_stats(lst)
        sweeps.append(("L", geom_axis, mean_a.copy(), max_a.copy()))

        # 3) fixed f_y, f_z, L, D, nu — vary W
        mean_a = np.zeros(npts)
        max_a = np.zeros(npts)
        for i, Wv in enumerate(geom_axis):
            lst = _run_one_simulation(L0, Wv, D0, fx0, fy0, fz0, nu0, defaults, "/tmp", fea_simulation)
            mean_a[i], max_a[i] = _x_displacement_stats(lst)
        sweeps.append(("W", geom_axis, mean_a.copy(), max_a.copy()))

        # 4) fixed f_z, W, D, L, nu — vary f_y
        mean_a = np.zeros(npts)
        max_a = np.zeros(npts)
        for i, fyv in enumerate(force_axis):
            lst = _run_one_simulation(L0, W0, D0, fx0, fyv, fz0, nu0, defaults, "/tmp", fea_simulation)
            mean_a[i], max_a[i] = _x_displacement_stats(lst)
        sweeps.append(("f_y", force_axis, mean_a.copy(), max_a.copy()))

        # 5) fixed f_y, W, D, L, nu — vary f_z
        mean_a = np.zeros(npts)
        max_a = np.zeros(npts)
        for i, fzv in enumerate(force_axis):
            lst = _run_one_simulation(L0, W0, D0, fx0, fy0, fzv, nu0, defaults, "/tmp", fea_simulation)
            mean_a[i], max_a[i] = _x_displacement_stats(lst)
        sweeps.append(("f_z", force_axis, mean_a.copy(), max_a.copy()))

        # 6) fixed f_y, W, D, L, f_z — vary nu
        mean_a = np.zeros(npts)
        max_a = np.zeros(npts)
        for i, nuv in enumerate(nu_axis):
            lst = _run_one_simulation(L0, W0, D0, fx0, fy0, fz0, nuv, defaults, "/tmp", fea_simulation)
            mean_a[i], max_a[i] = _x_displacement_stats(lst)
        sweeps.append(("nu", nu_axis, mean_a.copy(), max_a.copy()))

        for name, xs, mean_y, max_y in sweeps:
            self.assertTrue(np.all(np.isfinite(mean_y)), msg="non-finite mean metric for sweep %s" % name)
            self.assertTrue(np.all(np.isfinite(max_y)), msg="non-finite max metric for sweep %s" % name)

        fig, axes = plt.subplots(6, 1, figsize=(7, 16), constrained_layout=True)
        for ax, (name, xs, mean_y, max_y) in zip(axes, sweeps):
            ax.plot(xs, mean_y, label="mean_t mean_i u_x")
            ax.plot(xs, max_y, label="max_t mean_i u_x")
            ax.set_xlabel(name)
            ax.set_ylabel("displacement (x)")
            ax.set_title("sweep: %s" % name)
            ax.legend(loc="best")

        out_dir = tempfile.mkdtemp(prefix="compression_fem_")
        out_path = Path(out_dir) / "compression_sweeps.png"
        fig.savefig(out_path)
        plt.close(fig)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 0)
        try:
            out_path.unlink()
            os.rmdir(out_dir)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
