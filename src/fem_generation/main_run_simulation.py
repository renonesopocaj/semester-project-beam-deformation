"""
main_run_simulation.py
Main access point to run simulations.
"""

from itertools import product, chain
from math import sqrt
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.fem_generation.command_creator import build_command, run_command

RESULTS_DIR = Path("Large_1700_rotated/train/graphs/")


def filename_from_params(params):
    """
    Creates simulation name from the parameters.
    """
    L = params["L"]
    W = params["W"]
    D = params["D"]
    NL = params["NL"]
    NW = params["NW"]
    ND = params["ND"]
    E = params["E"]
    nu = params["nu"]
    density = params["rho"]
    eta_m_val = params["eta_m"]
    eta_k_val = params["eta_k"]
    initial_force_x = params.get("initial_force_x", 0.0)
    initial_force_y = params.get("initial_force_y", 0.0)
    initial_force_z = params.get("initial_force_z", 0.0)
    total_time = params["total_time"]
    cutoff_time_factor = params["cutoff_time_factor"]
    num_steps = params["num_steps"]
    simulation_name = (
        f"graphsL{L}_W{W}_D{D}_NL{NL}_NW{NW}_ND{ND}_"
        f"E{E}_nu{nu}_rho{density}_"
        f"em{eta_m_val}_ek{eta_k_val}"
        f"_Pix{initial_force_x}_Piy{initial_force_y}_Piz{initial_force_z}_"
        f"T{total_time}_Tc{cutoff_time_factor * total_time}_Nsteps{num_steps}"
    )
    return RESULTS_DIR / f"{simulation_name}.pt"


def generate_val_excluding_train(start, stop, n_val, train_values):
    train_set = set(np.round(train_values, 10))
    factor = 2
    while True:
        candidates = np.linspace(start, stop, factor * n_val)
        vals = []
        for x in candidates:
            xr = round(x, 10)
            if xr not in train_set:
                vals.append(x)
            if len(vals) == n_val:
                return vals
        factor *= 2


def give_config(config_str, mode, train_samples):
    if mode not in ["train", "test", "val"]:
        raise NotImplementedError(f"Invalid config_str: {config_str}")
    if config_str not in ["original", "rotated"]:
        raise NotImplementedError(f"Invalid config_str: {config_str}")

    TEST_PERCENTAGE = 0.1
    VAL_PERCENTAGE = 0.1
    # ================== OG ==================
    START_FORCES_OG = 0.25
    STOP_FORCES_OG = 1.5
    START_LENGTHS_OG = 0.5
    STOP_LENGTHS_OG = 1.0
    START_WIDTHS_OG = 0.06
    STOP_WIDTHS_OG = 0.14
    # Test
    START_FORCES_TEST_OG = 1.6
    STOP_FORCES_TEST_OG = 2
    START_LENGTHS_TEST_OG = 1.1
    STOP_LENGTHS_TEST_OG = 1.5
    START_WIDTHS_TEST_OG = 0.15
    STOP_WIDTHS_TEST_OG = 0.2
    # Number of samples
    N_SAMPLES_FORCES_OG = 10
    N_SAMPLES_LENGTHS_OG = 10
    N_SAMPLES_WIDTHS_OG = 10
    # ================== ROTATION ==================
    START_FORCE_X_ROTATION = 0.0
    STOP_FORCE_X_ROTATION = 1.5
    START_FORCE_Y_ROTATION = 0.0
    STOP_FORCE_Y_ROTATION = 1.5
    START_FORCE_Z_ROTATION = 0.0
    STOP_FORCE_Z_ROTATION = 1.5
    # Test
    START_FORCE_X_TEST_ROTATION = 1.6
    STOP_FORCE_X_TEST_ROTATION = 2
    START_FORCE_Y_TEST_ROTATION = 1.6
    STOP_FORCE_Y_TEST_ROTATION = 2
    START_FORCE_Z_TEST_ROTATION = 1.6
    STOP_FORCE_Z_TEST_ROTATION = 2
    # Number of samples
    N_SAMPLES_FORCE_ROTATION = 7
    N_SAMPLES_LENGTHS_ROTATION = 6
    N_SAMPLES_WIDTHS_ROTATION = 6
    # Rotation
    forces_x_rotation = list(np.linspace(START_FORCE_X_ROTATION, STOP_FORCE_X_ROTATION, N_SAMPLES_FORCE_ROTATION))
    forces_y_rotation = list(np.linspace(START_FORCE_Y_ROTATION, STOP_FORCE_Y_ROTATION, N_SAMPLES_FORCE_ROTATION))
    forces_z_rotation = list(np.linspace(START_FORCE_Z_ROTATION, STOP_FORCE_Z_ROTATION, N_SAMPLES_FORCE_ROTATION))
    forces_train_rotation = list(product(forces_x_rotation, forces_y_rotation, forces_z_rotation))
    lengths_train_rotation = list(np.linspace(START_LENGTHS_OG, STOP_LENGTHS_OG, N_SAMPLES_LENGTHS_ROTATION))
    widths_train_rotation = list(np.linspace(START_WIDTHS_OG, STOP_WIDTHS_OG, N_SAMPLES_WIDTHS_ROTATION))
    train_samples_rotation = (N_SAMPLES_FORCE_ROTATION ** 3) * N_SAMPLES_LENGTHS_ROTATION * N_SAMPLES_WIDTHS_ROTATION

    if mode != "train":
        test_samples = TEST_PERCENTAGE * train_samples_rotation
        per_feature_test_samples = int(test_samples ** 1 / 3)
        val_samples = VAL_PERCENTAGE * train_samples_rotation
        per_feature_val_samples = int(val_samples ** 1 / 3)

    if config_str == "original":
        train_samples_rotation = N_SAMPLES_FORCES_OG * N_SAMPLES_LENGTHS_OG * N_SAMPLES_WIDTHS_OG
        # Train
        forces_train_rotation = list(np.linspace(START_FORCES_OG, STOP_FORCES_OG, N_SAMPLES_FORCES_OG))
        lengths_train_rotation = list(np.linspace(START_LENGTHS_OG, STOP_LENGTHS_OG, N_SAMPLES_LENGTHS_OG))
        widths_train_rotation = list(np.linspace(START_WIDTHS_OG, STOP_WIDTHS_OG, N_SAMPLES_WIDTHS_OG))

        if mode == "train":
            return forces_train_rotation, lengths_train_rotation, widths_train_rotation, train_samples_rotation

        elif mode == "val":
            forces_val = list(generate_val_excluding_train(START_FORCES_OG, STOP_FORCES_OG, per_feature_val_samples,
                                                           forces_train_rotation))
            lengths_val = list(generate_val_excluding_train(START_LENGTHS_OG, STOP_LENGTHS_OG, per_feature_val_samples,
                                                            lengths_train_rotation))
            widths_val = list(generate_val_excluding_train(START_WIDTHS_OG, STOP_WIDTHS_OG, per_feature_val_samples,
                                                           widths_train_rotation))
            return forces_val, lengths_val, widths_val

        else:
            # FIXME: could do this slightly better and change one parameter at a time, i.e.
            # first only lengths out of distribution, then only forces, then only widths
            forces_test = list(product(forces_train_rotation[:per_feature_val_samples],
                                       forces_train_rotation[:per_feature_val_samples],
                                       forces_train_rotation[:per_feature_val_samples]))
            lengths_test = list(np.linspace(START_LENGTHS_TEST_OG, STOP_LENGTHS_TEST_OG, per_feature_test_samples))
            widths_test = list(np.linspace(START_WIDTHS_TEST_OG, STOP_WIDTHS_TEST_OG, per_feature_test_samples))
            return forces_test, lengths_test, widths_test

    else:  # rotated
        if mode == "train":
            return forces_train_rotation, lengths_train_rotation, widths_train_rotation, train_samples_rotation

        elif mode == "val":
            forces_x_val = list(generate_val_excluding_train(START_FORCE_X_ROTATION, STOP_FORCE_X_ROTATION,
                                                             per_feature_val_samples, forces_train_rotation))
            forces_y_val = list(generate_val_excluding_train(START_FORCE_Y_ROTATION, STOP_FORCE_Y_ROTATION,
                                                             per_feature_val_samples, forces_train_rotation))
            forces_z_val = list(generate_val_excluding_train(START_FORCE_Z_ROTATION, STOP_FORCE_Z_ROTATION,
                                                             per_feature_val_samples, forces_train_rotation))
            forces_val = product(forces_x_val, forces_y_val, forces_z_val)
            lengths_val = list(generate_val_excluding_train(START_LENGTHS_OG, STOP_LENGTHS_OG, per_feature_val_samples,
                                                            lengths_train_rotation))
            widths_val = list(generate_val_excluding_train(START_WIDTHS_OG, STOP_WIDTHS_OG, per_feature_val_samples,
                                                           widths_train_rotation))
            return forces_val, lengths_val, widths_val

        else:  # test
            # FIXME: could do this slightly better and change one parameter at a time, i.e.
            # first only lengths out of distribution, then only forces, then only widths
            forces_x_test = list(np.linspace(START_FORCE_X_TEST_ROTATION, STOP_FORCE_X_TEST_ROTATION,
                                             per_feature_val_samples))
            forces_y_test = list(np.linspace(START_FORCE_Y_TEST_ROTATION, STOP_FORCE_Y_TEST_ROTATION,
                                             per_feature_val_samples))
            forces_z_test = list(np.linspace(START_FORCE_Z_TEST_ROTATION, STOP_FORCE_Z_TEST_ROTATION,
                                             per_feature_val_samples))
            forces_test = product(forces_x_test, forces_y_test, forces_z_test)
            lengths_test_rotation = list(np.linspace(START_LENGTHS_TEST_OG, STOP_LENGTHS_TEST_OG, per_feature_test_samples))
            widths_test_rotation = list(np.linspace(START_WIDTHS_TEST_OG, STOP_WIDTHS_TEST_OG, per_feature_test_samples))
            return forces_test, lengths_test_rotation, widths_test_rotation


def run_simulations():
    # Default parameters
    default_params = {"L": 1.0, "W": 0.1, "D": 0.1, "NL": 8, "NW": 4, "ND": 4, "E": 1000.0, "nu": 0.3, "rho": 1.0,
                      "eta_m": 0.01, "eta_k": 0.01, "alpha_m": 0.0, "alpha_f": 0.0, "total_time": 4.0,
                      "cutoff_time_factor": 0.2, "num_steps": 50}
    # For 'train' dataset
    # forces = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    # lengths = [0.5, 0.6, 0.75, 0.8, 0.9, 1.0]
    # widths = [0.1, 0.11, 0.12, 0.14, 0.08, 0.06]
    forces, lengths, widths = give_config("rotated", "train")
    param_combinations = list(product(forces, lengths, widths))
    print(f"Total simulations to run: {len(param_combinations)}")
    for idx, (initial_force, L, W) in tqdm(enumerate(param_combinations), total=len(param_combinations)):
        print(f"\n=== Simulation Set {idx + 1} ===")
        params = default_params.copy()
        params.update({"initial_force_x": initial_force[0], "initial_force_y": initial_force[1],
                       "initial_force_z": initial_force[2], "L": L, "W": W, "mode": "train"})
        filename = filename_from_params(params)
        if filename.exists():
            print(f"Skipping simulation, results already exist at {filename}")
            continue
        command = build_command(params)
        print(
            f"\nRunning train simulation with initial_force_x={initial_force[0]}, initial_force_y={initial_force[1]}, "
            f"initial_force_z={initial_force[2]} L={L}, W={W}")
        run_command(command)


if __name__ == "__main__":
    run_simulations()
