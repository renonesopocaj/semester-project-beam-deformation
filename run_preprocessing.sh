#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=39 # max 40 per node
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "STDOUT file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "STDERR file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject

python -m src.gnn.preprocess --multirun features=all,nonlin,edges_w_norms \
    dataset_path=slurm_biax_allgeoms,slurm_single_allgeoms,slurm_biax_longer,slurm_single_longer,slurm_triax_allgeoms,slurm_triax_longer \
    size.dataset_size=large,small,medium