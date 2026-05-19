#!/bin/bash
#SBATCH --job-name=sweep_gunet
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=39 # max 40 per node
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jacopo.senoner@epfl.ch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "STDOUT file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "STDERR file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject
export WANDB_INIT_TIMEOUT=300
mkdir -p saved_models

srun python -m src.gnn.hyperparam_opt -cn graph_unet_sweeps