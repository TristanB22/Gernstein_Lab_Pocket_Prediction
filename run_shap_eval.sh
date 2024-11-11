#!/bin/bash

#SBATCH --job-name=evaluating_pocket_model
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=6
#SBATCH --output=evaluation_output.txt

python -m pip install --upgrade pip

module purge
module load PyTorch
module load CUDA/12.1.1

python3 evaluate_model.py
