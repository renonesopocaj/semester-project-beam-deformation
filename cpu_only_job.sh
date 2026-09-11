#!/bin/bash
#SBATCH --job-name=fem_gen
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jacopo.senoner@epfl.ch

cd "$SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "STDOUT file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "STDERR file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
echo "Dataset root expected at: $(pwd)/dataset"
echo "Artifacts expected at: $(pwd)/saved_models"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 90G"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

# mgn_message_mid w/ z_isotr_centr_log_abs_noxtm1_nonormnode_noedgevel_noedgeforce_nostretch_norot
# mgn_deep w/ z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot

# srun python -m src.fem_generation.main_run_fem size=medium force_axes=biax
# srun python -m src.gnn.preprocess size=medium features=nonlin dataset_path=slurm_biax_longer
# srun python -m src.gnn.preprocess --multirun dataset_path=slurm_biax_longer normalizer=z_isotropic,robust_isotropic,robust_anisotropic,z_anisotropic features=nonlin size=medium
srun python -m src.gnn.preprocess --multirun dataset_path=slurm_biax_longer features=nonlin,equivariant,noabspos_yesedge,nostretch_abspos_nonlin,only_edge_feat size=large