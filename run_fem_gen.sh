#!/bin/bash
#SBATCH --job-name=femgen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=39 # max 40 per node
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
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

srun python -m src.fem_generation.main_run_fem --multirun geometry=allgeoms,longer force_axes=triax,single,biax size=small,medium,large