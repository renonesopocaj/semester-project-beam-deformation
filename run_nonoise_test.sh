#!/bin/bash
#SBATCH --job-name=nonoise
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=39 # max 40 per node
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
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
echo "Dataset root expected at: $(pwd)/dataset"
echo "Artifacts expected at: $(pwd)/saved_models"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject
mkdir -p saved_models

srun python -m src.gnn.train computational=in_memory_slurm_1gpu_littlebatch \
  computational.num_workers=0 \
  computational.torch_compile=True \
  lr_scheduler=plateau_long \
  training=slurm \
  noise=noise_per_epoch_noprev \
  training.total_epochs=300 \
  model=mgn_deep_midmsg \
  devices=one_gpu \
  data=slurm \
  data.dataset_dir="/scratch/izar/senoner/dataset/processed/biax_longer/z_isotr_centr_all/small"