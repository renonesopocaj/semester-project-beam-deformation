"""
command_creator.py
Creates the CLI commands used by main_run_simulation.py to run one simulation, for many simulations.
"""

import subprocess
import sys

def build_command(params):
    """
    Builds the command list for subprocess.run from parameters.
    """
    command = ["python", "beam_fea_solver.py"]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command


def run_command(command):
    """
    Runs a command and ensures the output is printed.
    """
    try:
        # Use subprocess.Popen to capture and print the output in real-time
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        # Read and print the output line by line as it becomes available
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except Exception as e:
        print(f"An error occurred while running the command: {e}", file=sys.stderr)