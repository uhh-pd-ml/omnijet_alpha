#!/bin/bash
# NOTE: This script is an example and should be adjusted to your needs.
# The fields which need to be adjusted are marked with "ADJUST THIS".

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu
#SBATCH --constraint=GPU
#SBATCH --time=48:00:00
#SBATCH --job-name TokTrain
#SBATCH --output /data/dust/user/birkjosc/beegfs/gabbro_output//logs/slurm_logs/%x_%j.log      # ADJUST THIS to your log path
#SBATCH --mail-user <your-email-address>  # ADJUST THIS to your email address

echo "Starting job $SLURM_JOB_ID with the following script:"
echo "----------------------------------------------------------------------------"
echo
cat $0

export REPO_DIR="/home/birkjosc/repositories/omnijet_alpha/"  # ADJUST THIS to your repository path
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

cd $REPO_DIR

LOGFILE="/data/dust/user/birkjosc/beegfs/gabbro_output//logs/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"  # ADJUST THIS to your log path

PYTHON_COMMAND="python gabbro/train.py experiment=example_experiment_tokenization"

# run the python command in the singularity container
# ADJUST THIS to your singularity image path, or replace with docker://jobirk/omnijet:latest
singularity exec --nv --bind /beegfs:/beegfs -B /data  \
    --env JOB_ID="$SLURM_JOB_ID" --env SLURM_LOGFILE="$LOGFILE" \
    /data/dust/user/birkjosc/singularity_images/omnijet-latest.sif \
    bash -c "source /opt/conda/bin/activate && $PYTHON_COMMAND"

## ---------------------- End of job script -----------------------------------
################################################################################
