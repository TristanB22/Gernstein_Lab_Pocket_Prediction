#!/bin/bash

#SBATCH --job-name=attn_pocket_prediction
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=5
#SBATCH --output=attn_pocket_pred_out.txt

module purge
module load CUDA
module load cuDNN
module load pytorch

source a
