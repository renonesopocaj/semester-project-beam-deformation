#!/bin/bash
#SBATCH --job-name=semester-project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=750G
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=19 # max 39 per node
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
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
# Two ranks (matches devices=two_gpu): explicit torchrun avoids Lightning staying at world_size 1.
srun python -m src.gnn.train \
  computational=in_memory_slurm_2gpu \
  computational.torch_compile=True \
  lr_scheduler=plateau \
  training=slurm \
  training.total_epochs=100 \
  model=mgn_large \
  devices=two_gpu \
  data=slurm \
  data.dataset_dir="/scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large"