#!/bin/bash
#SBATCH --job-name=generate_molecules_gpu      # descriptive job name
#SBATCH --output=./runs/logs/generate_molecules_%j.out  # standard output file
#SBATCH --error=./runs/logs/generate_molecules_%j.err   # standard error file
#SBATCH --partition=scavenge_gpu                   # partition to submit to
#SBATCH --time=24:00:00                        # maximum runtime (24 hours)
#SBATCH --ntasks=1                             # number of tasks
#SBATCH --cpus-per-task=6                      # allocate 6 CPU cores
#SBATCH --mem-per-cpu=4G                       # allocate 4 GB per CPU core
#SBATCH --gpus=2                               # request 2 GPUs
#SBATCH --requeue                              # allow job to restart on interruption

# Load necessary modules
module load miniconda                          # load the Miniconda module

# Activate the Conda environment
source activate 583final                      # activate the appropriate environment

# Print job details
echo "======================================="
echo "Job started on $(hostname) at $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs allocated: $SLURM_JOB_GPUS"
echo "Using Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "======================================="

# Run your script
python full_run.py

# Print completion message
echo "======================================="
echo "Job completed on $(hostname) at $(date)"
echo "======================================="

