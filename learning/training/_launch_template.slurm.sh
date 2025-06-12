#!/bin/bash -l

#SBATCH --job-name="SAME_AS_WANDB?"
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --mail-user=EMAIL
#SBATCH --mail-type=ALL
#SBATCH --err ERROR_LOG.%j.%N.txt
#SBATCH --out OUTPUT_LOG.%j.%N.txt

# Other useful SBATCH options:
# --nodes=# (number of nodes to use)
# --ntasks-per-node=# (number of tasks to run on each node)
# --cpus-per-gpu=# (number of CPUs to allocate per GPU)

# Consider using paralel (https://sulis-hpc.github.io/advanced/ensemble/gnuparallel.html)
# PARALLEL_OPTIONS="--jobs $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.txt --arg-file ARGFILE"
# SRUN_OPTIONS="--nodes 1 --ntask 1 --exclusive"
# parallel $PARALLEL_OPTIONS srun $SRUN_OPTIONS python training.py
# - can split arguments into multiple files and handle combinations of arguments automatically

# Steps to run a training script on the cluster:
# 1. Make a copy of this file (eg, cp _launch_template.sh new_launch_script.sh)
#    - come up with a better file name than new_launch_script.sh
#    - maybe something based on the model name, project name, etc.
# 2. Edit the new file and set the
#    - job-name
#    - mail-user
#    - ERROR_LOG and OUTPUT_LOG (eg, base on job name)
#    - ENVIRONMENT, MODEL_NAME, PROJECT_NAME, description, ARCHITECTURE_NAME, DATA_NAME
#    - python script line
#    - any other changes you deem necessary (eg, a different GPU or more memory)
# 3. Run the sbatch script with: sbatch new_launch_script.sh

# End the script if any command fails
set -euo pipefail

# Print the current date and hostname for debugging
date
hostname

# Load conda and activate an environment
module load miniconda3
conda activate ENVIRONMENT



python training.py MODEL_NAME PROJECT_NAME "Sample description of training run..." ARCHITECTURE_NAME DATA_NAME

# Print the date again to see how long the job took
date
