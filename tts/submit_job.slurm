#!/bin/bash
#SBATCH --account=bgbh-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --job-name=winogrande_tts
#SBATCH --output=tts_output_%j.txt

# --- Your environment setup and commands go below this line ---

# Load necessary modules
module load miniforge3-python
eval "$(mamba shell hook --shell bash)"
mamba activate ./tts_env

# Run your parallel generation script
# Add the -u flag to force real-time logging
python -u tts_parallel_all_kokoro.py