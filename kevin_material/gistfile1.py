import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

RESULTS_DIR = Path("Results_large_10_pow_3/train/graphs/")


def filename_from_params(params):
    """simulation_name = (
        f"L{L}_W{W}_D{D}_NL{NL}_NW{NW}_ND{ND}_"
        f"E{E}_nu{nu}_rho{density}_"
        f"em{eta_m_val}_ek{eta_k_val}"
        f"_Pix{initial_force_x}_Piy{initial_force_y}_Piz{initial_force_z}_"
        f"T{total_time}_Tc{cutoff_time_factor * total_time}_Nsteps{num_steps}"
    )
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


def run_simulations():
    # Default parameters
    default_params = {
        "L": 1.0,
        "W": 0.1,
        "D": 0.1,
        "NL": 8,
        "NW": 4,
        "ND": 4,
        "E": 1000.0,
        "nu": 0.3,
        "rho": 1.0,
        "eta_m": 0.01,
        "eta_k": 0.01,
        "alpha_m": 0.0,
        "alpha_f": 0.0,
        "total_time": 4.0,
        "cutoff_time_factor": 0.2,
        "num_steps": 50,
        "initial_force_x": 0.0,
        "initial_force_z": 0.0,
    }

    # For 'train' dataset
    # forces = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    # lengths = [0.5, 0.6, 0.75, 0.8, 0.9, 1.0]
    # widths = [0.1, 0.11, 0.12, 0.14, 0.08, 0.06]

    # check if filename already exists, if so, skip
    n_samples = 10

    forces = list(np.linspace(0.25, 1.5, n_samples))
    lengths = list(np.linspace(0.5, 1.0, n_samples))
    widths = list(np.linspace(0.06, 0.14, n_samples))

    param_combinations = list(product(forces, lengths, widths))
    print(f"Total simulations to run: {len(param_combinations)}")
    for idx, (initial_force_y, L, W) in tqdm(
        enumerate(param_combinations), total=len(param_combinations)
    ):
        print(f"\n=== Simulation Set {idx + 1} ===")

        params = default_params.copy()
        params.update(
            {
                "initial_force_y": initial_force_y,
                "L": L,
                "W": W,
                "mode": "train",
            }
        )
        filename = filename_from_params(params)
        if filename.exists():
            print(f"Skipping simulation, results already exist at {filename}")
            continue
        command = build_command(params)
        print(
            f"\nRunning train simulation with initial_force={initial_force_y}, L={L}, W={W}"
        )
        run_command(command)


def build_command(params):
    """Builds the command list for subprocess.run from parameters."""
    command = ["python", "beam_fea_solver.py"]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def run_command(command):
    """Runs a command and ensures the output is printed."""
    try:
        # Use subprocess.Popen to capture and print the output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        # Read and print the output line by line as it becomes available
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except Exception as e:
        print(f"An error occurred while running the command: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_simulations()
