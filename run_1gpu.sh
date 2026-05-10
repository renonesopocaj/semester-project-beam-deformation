#!/bin/bash
#SBATCH --job-name=train
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
echo "Dataset root expected at: $(pwd)/dataset"
echo "Artifacts expected at: $(pwd)/saved_models"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject
# mgn_message_mid w/ z_isotr_centr_log_abs_noxtm1_nonormnode_noedgevel_noedgeforce_nostretch_norot
# mgn_deep w/ z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot
mkdir -p saved_models
srun python -m src.gnn.train computational=in_memory_slurm_1gpu \
  computational.torch_compile=True \
  lr_scheduler=plateau_long \
  training=slurm \
  training.total_epochs=300 \
  model=mgn_deep_midmsg \
  data.dataset_dir="/scratch/izar/senoner/dataset/processed/biax_longer/z_isotr_centr_all/medium"