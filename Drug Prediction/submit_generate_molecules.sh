#!/bin/bash
#SBATCH --job-name=generate_molecules_gpu     
#SBATCH --output=./runs/logs/generate_molecules_%j.out  
#SBATCH --error=./runs/logs/generate_molecules_%j.err   
#SBATCH --partition=scavenge_gpu                  
#SBATCH --time=24:00:00                       
#SBATCH --ntasks=1                            
#SBATCH --cpus-per-task=6                    
#SBATCH --mem-per-cpu=4G                     
#SBATCH --gpus=2                             
#SBATCH --requeue                             

module load miniconda                       

ENV_NAME="test_run_env_583_final"
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} not found. Creating from environment.yml..."
    conda env create -f environment.yml
fi

source activate $ENV_NAME


echo "======================================="
echo "Job started on $(hostname) at $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs allocated: $SLURM_JOB_GPUS"
echo "Using Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "======================================="


python main.py --dataset pdbbind


echo "======================================="
echo "Job completed on $(hostname) at $(date)"
echo "======================================="

