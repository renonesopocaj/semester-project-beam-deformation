#!/bin/bash
#SBATCH --job-name=semester-project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=196G  
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "STDOUT file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "STDERR file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
echo "Dataset root expected at: $(pwd)/dataset"
echo "Artifacts expected at: $(pwd)/saved_models"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject

mkdir -p saved_models
python -m src.gnn.preprocess --multirun features=all,no_abs_pos,no_edge_vel,no_force_edge,no_node_norms,no_past_pos,no_rotation_angle,no_stretch_edge paths=slurm_rotated_beam,slurm_single_direction_beam