#!/bin/bash
#SBATCH --job-name=drug_prediction
#SBATCH --output=drug_prediction_%j.log
#SBATCH --error=drug_prediction_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1                       # Number of GPUs
#SBATCH --mem=5G                          # Memory allocation
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4              

# Load necessary modules
module load miniconda

# Activate Conda environment
source activate assignment_2

# Prioritize Conda's lib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Run your Python script
python3 full_run.py
