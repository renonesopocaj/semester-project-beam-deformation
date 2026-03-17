"""
main_run_simulation.py

Generates train / val / test parameter sets from a YAML config file,
then runs beam FEA simulations by calling fea_simulation()

The idea is that
1) validation has values in-distribution (i.e. in the range train), but different from train
2) test has out-of-distribution values (outside the range of train), but I want to generate it in a way such that there
are test samples that have only one feature (between force_x, force_y, force_z, lengths, width) out of distribution,
and then some that have 2, some 3, and so on. Every test sample must have at least one feature out of distribution.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.fem_generation.beam_fea_solver import fea_simulation

def _graph_filepath(params, dir_path):
    """
    Reconstruct the .pt path that fea_simulation will write.
    """
    p = params
    name = (f"graphsL{p['L']}_W{p['W']}_D{p['D']}"
            f"_NL{p['NL']}_NW{p['NW']}_ND{p['ND']}"
            f"_E{p['E']}_nu{p['nu']}_rho{p['rho']}"
            f"_em{p['eta_m']}_ek{p['eta_k']}"
            f"_Pix{p['fx']}_Piy{p['fy']}_Piz{p['fz']}"
            f"_T{p['total_time']}"
            f"_Tc{p['cutoff_time_factor'] * p['total_time']}"
            f"_Nsteps{p['num_steps']}")
    return dir_path / f"{name}.pt"

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def linspace_list(start, stop, n):
    """Return a Python list from np.linspace."""
    return list(np.linspace(start, stop, n))


def vals_excluding(start, stop, n, train_values):
    """
    Generate *n* values uniformly spread in [start, stop] that do NOT appear
    in *train_values*.  If start == stop (constant feature), just repeat that
    value *n* times.
    """
    if abs(start - stop) < 1e-12:
        return [start] * n
    train_set = set(np.round(train_values, 10))
    factor = 2
    while True:
        candidates = np.linspace(start, stop, factor * n)
        vals = [x for x in candidates if round(x, 10) not in train_set]
        if len(vals) >= n:
            return vals[:n]
        factor *= 2


def _in_range(val, lo, hi, tol=1e-9):
    return (lo - tol) <= val <= (hi + tol)


def filter_all_in_distribution(combos, train_ranges):
    """
    Remove any sample where *every* feature is inside the training range.
    Each combo is ((fx, fy, fz), L, W).
    train_ranges maps feature name -> (min, max).
    """
    kept = []
    for (fx, fy, fz), L, W in combos:
        n_ood = sum([
            not _in_range(fx, *train_ranges["fx"]),
            not _in_range(fy, *train_ranges["fy"]),
            not _in_range(fz, *train_ranges["fz"]),
            not _in_range(L,  *train_ranges["L"]),
            not _in_range(W,  *train_ranges["W"]),
        ])
        if n_ood >= 1:
            kept.append(((fx, fy, fz), L, W))
    return kept


# ---------------------------------------------------------------------------
#  Split builders
# ---------------------------------------------------------------------------

def _train_axis_values(cfg):
    """Return per-axis training lists (fx, fy, fz, lengths, widths)."""
    t = cfg["train"]
    fx = linspace_list(t["force_x"]["start"], t["force_x"]["stop"], t["force_x"]["n"])
    fy = linspace_list(t["force_y"]["start"], t["force_y"]["stop"], t["force_y"]["n"])
    fz = linspace_list(t["force_z"]["start"], t["force_z"]["stop"], t["force_z"]["n"])
    lengths = linspace_list(t["lengths"]["start"], t["lengths"]["stop"], t["lengths"]["n"])
    widths  = linspace_list(t["widths"]["start"],  t["widths"]["stop"],  t["widths"]["n"])
    return fx, fy, fz, lengths, widths


def _train_ranges(cfg):
    """Min/max of each training feature (used for OOD filtering)."""
    t = cfg["train"]
    return {
        "fx": (t["force_x"]["start"], t["force_x"]["stop"]),
        "fy": (t["force_y"]["start"], t["force_y"]["stop"]),
        "fz": (t["force_z"]["start"], t["force_z"]["stop"]),
        "L":  (t["lengths"]["start"],  t["lengths"]["stop"]),
        "W":  (t["widths"]["start"],   t["widths"]["stop"]),
    }


def build_train(cfg):
    fx, fy, fz, lengths, widths = _train_axis_values(cfg)
    forces = list(product(fx, fy, fz))
    return list(product(forces, lengths, widths))


def build_val(cfg):
    """
    Validation: values inside the training range but different from training
    values.  Number of per-feature samples is derived from val_fraction.
    """
    t = cfg["train"]
    fx_tr, fy_tr, fz_tr, L_tr, W_tr = _train_axis_values(cfg)

    n_train = len(build_train(cfg))
    n_per = max(2, int((cfg["val_fraction"] * n_train) ** (1.0 / 5)))

    fx = vals_excluding(t["force_x"]["start"], t["force_x"]["stop"], n_per, fx_tr)
    fy = vals_excluding(t["force_y"]["start"], t["force_y"]["stop"], n_per, fy_tr)
    fz = vals_excluding(t["force_z"]["start"], t["force_z"]["stop"], n_per, fz_tr)
    lengths = vals_excluding(t["lengths"]["start"], t["lengths"]["stop"], n_per, L_tr)
    widths  = vals_excluding(t["widths"]["start"],  t["widths"]["stop"],  n_per, W_tr)

    forces = list(product(fx, fy, fz))
    return list(product(forces, lengths, widths))


def build_test(cfg):
    """
    Test: each feature list is half OOD values + half in-distribution values
    (excluding exact training points).  After the Cartesian product, any sample
    whose features are ALL in-distribution is removed, guaranteeing every test
    sample has at least one OOD feature.
    """
    t   = cfg["train"]
    ood = cfg["test_ood_ranges"]
    fx_tr, fy_tr, fz_tr, L_tr, W_tr = _train_axis_values(cfg)

    n_train = len(build_train(cfg))
    n_per  = max(2, int((cfg["test_fraction"] * n_train) ** (1.0 / 5)))
    n_half = max(1, n_per // 2)

    def _test_values(id_start, id_stop, ood_start, ood_stop, n_h, train_vals):
        ood_part = linspace_list(ood_start, ood_stop, n_h)
        id_part  = vals_excluding(id_start, id_stop, n_h, train_vals)
        return ood_part + id_part

    fx = _test_values(t["force_x"]["start"], t["force_x"]["stop"],
                      ood["force_x"]["start"], ood["force_x"]["stop"], n_half, fx_tr)
    fy = _test_values(t["force_y"]["start"], t["force_y"]["stop"],
                      ood["force_y"]["start"], ood["force_y"]["stop"], n_half, fy_tr)
    fz = _test_values(t["force_z"]["start"], t["force_z"]["stop"],
                      ood["force_z"]["start"], ood["force_z"]["stop"], n_half, fz_tr)
    lengths = _test_values(t["lengths"]["start"], t["lengths"]["stop"],
                           ood["lengths"]["start"], ood["lengths"]["stop"], n_half, L_tr)
    widths  = _test_values(t["widths"]["start"],  t["widths"]["stop"],
                           ood["widths"]["start"],  ood["widths"]["stop"],  n_half, W_tr)

    forces = list(product(fx, fy, fz))
    combos = list(product(forces, lengths, widths))

    ranges = _train_ranges(cfg)
    return filter_all_in_distribution(combos, ranges)


# ---------------------------------------------------------------------------
#  Filename (must match what beam_fea_solver writes)
# ---------------------------------------------------------------------------

def _graph_filepath(params, dir_path):
    """Reconstruct the .pt path that fea_simulation will write."""
    p = params
    name = (f"graphsL{p['L']}_W{p['W']}_D{p['D']}"
            f"_NL{p['NL']}_NW{p['NW']}_ND{p['ND']}"
            f"_E{p['E']}_nu{p['nu']}_rho{p['rho']}"
            f"_em{p['eta_m']}_ek{p['eta_k']}"
            f"_Pix{p['fx']}_Piy{p['fy']}_Piz{p['fz']}"
            f"_T{p['total_time']}"
            f"_Tc{p['cutoff_time_factor'] * p['total_time']}"
            f"_Nsteps{p['num_steps']}")
    return dir_path / f"{name}.pt"


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

def run_simulations(config_path, verbose, skip_existing):
    cfg = load_config(config_path)
    defaults    = cfg["defaults"]
    output_root = cfg["output_root"]

    splits = {
        "train": build_train(cfg),
        "val":   build_val(cfg),
        "test":  build_test(cfg),
    }

    for mode in ("train", "val", "test"):
        combos   = splits[mode]
        dir_path = Path(output_root) / mode / "graphs"
        dir_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  {mode.upper()}: {len(combos)} simulations")
        print(f"{'=' * 60}")

        for idx, ((fx, fy, fz), L, W) in tqdm(enumerate(combos), total=len(combos)):
            # Build a param dict for the skip-check filename
            p = {**defaults, "L": L, "W": W, "fx": fx, "fy": fy, "fz": fz}
            if _graph_filepath(p, dir_path).exists() and skip_existing:
                continue

            if verbose:
                print(f"\nSim {idx + 1}/{len(combos)}:  "
                      f"fx={fx:.4f}  fy={fy:.4f}  fz={fz:.4f}  L={L:.4f}  W={W:.4f}")

            fea_simulation(
                L=L, W=W, D=defaults["D"],
                NL=defaults["NL"], NW=defaults["NW"], ND=defaults["ND"],
                elastic_params=(defaults["E"], defaults["nu"]),
                density=defaults["rho"],
                damping_params=(defaults["eta_m"], defaults["eta_k"]),
                newmark_params=(defaults["alpha_m"], defaults["alpha_f"]),
                initial_force_x=fx,
                initial_force_y=fy,
                initial_force_z=fz,
                cutoff_time_factor=defaults["cutoff_time_factor"],
                total_time=defaults["total_time"],
                num_steps=defaults["num_steps"],
                mode=mode,
                output_root=output_root,
                verbose=verbose
            )


if __name__ == "__main__":
    config = "conf/fem_generation/config_rotated.yaml"
    run_simulations(config, verbose=False, skip_existing=False)