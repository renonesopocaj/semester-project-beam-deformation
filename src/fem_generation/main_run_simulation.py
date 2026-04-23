"""
main_run_simulation.py

Calls functions in build_parameter_tuples.py to generate train / val / test parameter sets from a YAML config file,
then runs beam FEA simulations by calling fea_simulation()
To see how parameter sets are generated see build_parmaeter_tuples.py
"""

import json
from src.fem_generation.build_parameter_tuples import (build_all_splits, _list_ood_feature_names,
                                                       _graph_filepath, _split_target_counts, _train_ranges,
                                                       _resolve_n_cells, _build_trajectory_params_json)
from pathlib import Path
import yaml
from tqdm import tqdm
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.filterwarnings("ignore", category=QuadratureRepresentationDeprecationWarning)
# df.set_log_level(df.LogLevel.ERROR)  # optional; does not control this warning

# Round-based equality used when deciding whether two float values should be treated as the same grid value.
DECIMALS = 10
# Small tolerance for interval checks.
TOL = 1e-9
ARE_EQUAL_THRESHOLD = 1e-12
# Threshold used to decide whether a range is effectively constant.

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """
    Load a YAML config file.

    Args:
        path: str
            config filepath
    return: dict
        parsed yaml config
    """
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
#  Selected trajectory helpers
# ---------------------------------------------------------------------------

def _selected_indices_for_mode(selected_trajs, mode):
    """
    Return the set of trajectory indices that should get longer rollouts
    for the given split mode.

    Args:
        selected_trajs: dict
            selected_trajs section from config
        mode: str
            split mode: "train", "val", or "test"
    return: set of int
        trajectory indices to run with longer rollout
    """
    if mode == "val":
        return set(selected_trajs.get("selected_validation_trajectories", []))
    elif mode == "test":
        return set(selected_trajs.get("selected_test_trajectories", []))
    return set()


def _save_extra_info(output_root, selected_trajs, ood_by_traj):
    """
    Save selected_trajectories.json and test_ood_features.json into the
    extra_info/ directory alongside the split folders.

    Args:
        output_root: str or pathlib.Path
            root output directory (sibling of train/val/test)
        selected_trajs: dict
            selected_trajs section from config
        ood_by_traj: dict or None
            OOD feature mapping per trajectory index (None if test not built)
    """
    extra_info_dir = Path(output_root) / "extra_info"
    extra_info_dir.mkdir(parents=True, exist_ok=True)
    with open(extra_info_dir / "selected_trajectories.json", "w") as f:
        json.dump(selected_trajs, f, indent=2)
    print(f"Saved selected_trajectories.json => {extra_info_dir / 'selected_trajectories.json'}")
    if ood_by_traj is not None:
        with open(extra_info_dir / "test_ood_features.json", "w") as f:
            json.dump(ood_by_traj, f, indent=2)
        print(f"Saved test_ood_features.json => {extra_info_dir / 'test_ood_features.json'}")


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------

def _run_only_json(cfg, defaults, output_root, requested_modes, selected_trajs):
    if "test" not in requested_modes:
        raise ValueError("only_json requires 'test' in modes.")
    _, _, n_test = _split_target_counts(cfg)
    if n_test == 0:
        raise ValueError("only_json requires a non-zero test split (check test_fraction / train size).")
    all_splits = build_all_splits(cfg)
    test_combos = all_splits["test"]
    test_dir = Path(output_root) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_ranges = _train_ranges(cfg)
    # Save trajectory_params.json (replaces old test_ood_features.json).
    _build_trajectory_params_json(test_combos, defaults, test_dir, train_ranges)
    ood_by_traj = {str(idx): _list_ood_feature_names(combo, train_ranges) for idx, combo in enumerate(test_combos)}
    ood_path = test_dir / "test_ood_features.json"
    with open(ood_path, "w") as f:
        json.dump(ood_by_traj, f, indent=2)
    print(json.dumps(ood_by_traj, indent=2))
    # Save extra_info (selected trajectories + test_ood_features)
    _save_extra_info(output_root, selected_trajs, ood_by_traj)

def _mode_isvalid(requested_modes, valid_modes):
    for mode in requested_modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Valid modes are: train, val, test.")

def run_simulations(config_path, verbose, skip_existing, modes):
    """
    1. load yaml
    2. build only the requested split parameter lists
    3. create output folders
    4. launch fea_simulation once per parameter combination

    Split construction is deterministic (no randomness): same config yields the same
    combo order and trajectory indices as enumerate(combos) in the simulation loop.

    Args:
        config_path: str
            path of config file
        verbose: bool
            printing control
        skip_existing: bool
            skip existing graphs if True
        modes: tuple or list
            requested splits, e.g. ("val", "test")
    """
    cfg = load_config(config_path)
    defaults = cfg["defaults"]
    output_root = cfg["output_root"]
    selected_trajs = cfg.get("selected_trajs", {})
    # Check mode is valid
    requested_modes = tuple(str(mode).lower() for mode in modes)
    valid_modes = {"train", "val", "test"}
    _mode_isvalid(requested_modes, valid_modes)
    # If only json mode, create only the jsons
    if cfg["only_json"]:
        _run_only_json(cfg, defaults, output_root, requested_modes, selected_trajs)
        return
    # Deferred import so MPI/Dolfin is never initialized for only_json runs.
    from src.fem_generation.beam_fea_solver import fea_simulation
    # Allocate combination builder
    n_train, n_val, n_test = _split_target_counts(cfg)
    expected_counts = {"train": n_train, "val": n_val, "test": n_test}
    all_splits = build_all_splits(cfg)
    splits = {mode: all_splits[mode] for mode in requested_modes}
    for mode in requested_modes:
        expected_len = expected_counts[mode] + (1 if mode == "test" else 0)
        if len(splits[mode]) != expected_len:
            raise RuntimeError(f"Split '{mode}' has length {len(splits[mode])}, "
                               f"but the expected target is {expected_len}.")
    # Save trajectory_params.json for every requested split.
    train_ranges = _train_ranges(cfg)
    for mode in requested_modes:
        split_dir = Path(output_root) / mode
        _build_trajectory_params_json(splits[mode], defaults, split_dir, train_ranges)
    # Create OOD json
    ood_by_traj = None
    if "test" in requested_modes and len(splits["test"]) > 0:
        test_dir = Path(output_root) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        ood_by_traj = {str(idx): _list_ood_feature_names(combo, train_ranges) for idx, combo in enumerate(splits["test"])}
        ood_path = test_dir / "test_ood_features.json"
        with open(ood_path, "w") as f:
            json.dump(ood_by_traj, f, indent=2)
    # Save extra_info (selected trajectories + test_ood_features)
    _save_extra_info(output_root, selected_trajs, ood_by_traj)
    # Read selected trajectory overrides
    selected_total_time = selected_trajs.get("selected_total_time", defaults["total_time"])
    selected_num_steps = selected_trajs.get("num_steps", defaults["num_steps"])

    # FOR EACH REQUESTED MODE
    for mode in requested_modes:
        combos = splits[mode]
        dir_path = Path(output_root) / mode / "graphs"
        dir_path.mkdir(parents=True, exist_ok=True)
        mode_selected_indices = _selected_indices_for_mode(selected_trajs, mode)

        sim_target = expected_counts[mode] + (1 if mode == "test" else 0)
        print("\n" + "=" * 60 + f"  {mode.upper()}: {len(combos)} simulations (target {sim_target})" + "=" * 60)

        # MAIN LOOP FOR EACH MODE
        for idx, ((fx, fy, fz), L, W, D) in tqdm(enumerate(combos), total=len(combos)):
            is_selected = idx in mode_selected_indices
            sim_total_time = selected_total_time if is_selected else defaults["total_time"]
            sim_num_steps = selected_num_steps if is_selected else defaults["num_steps"]
            p = dict(defaults)
            p["L"], p["W"], p["D"] = L, W, D
            p["NL"] = _resolve_n_cells(defaults["NL"], L, "NL")
            p["NW"] = _resolve_n_cells(defaults["NW"], W, "NW")
            p["ND"] = _resolve_n_cells(defaults["ND"], D, "ND")
            p["fx"], p["fy"], p["fz"] = fx, fy, fz
            p["total_time"], p["num_steps"] = sim_total_time, sim_num_steps
            if _graph_filepath(p, dir_path).exists() and skip_existing:
                continue

            if verbose:
                tag = " [SELECTED]" if is_selected else ""
                print(f"\nSim {idx + 1}/{len(combos)}:  fx={fx:.4f}  fy={fy:.4f}  fz={fz:.4f}"
                      f"  L={L:.4f}  W={W:.4f}  D={D:.4f}  T={sim_total_time}  Nsteps={sim_num_steps}{tag}")

            fea_simulation(L=L, W=W, D=D,
                           NL=p["NL"], NW=p["NW"], ND=p["ND"],
                           elastic_params=(defaults["E"], defaults["nu"]),
                           density=defaults["rho"],
                           damping_params=(defaults["eta_m"], defaults["eta_k"]),
                           newmark_params=(defaults["alpha_m"], defaults["alpha_f"]),
                           initial_force_x=fx, initial_force_y=fy, initial_force_z=fz,
                           cutoff_time_factor=defaults["cutoff_time_factor"],
                           total_time=sim_total_time, num_steps=sim_num_steps,
                           mode=mode, output_root=output_root,
                           verbose=verbose,
                           first_trajectory_debug=(idx == 0),
                           linear_solver=defaults["linear_solver"])


if __name__ == "__main__":
    config = "conf/fem_generation/extreme.yaml"
    MODES = ("train", "val", "test")
    run_simulations(config, verbose=False, skip_existing=False, modes=MODES)
