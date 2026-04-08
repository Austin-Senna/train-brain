#!/bin/bash
#SBATCH --account=bgbh-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --job-name=comps_wugs_dist_tts
#SBATCH --output=tts_wugs_dist_%j.txt

module load miniforge3-python
eval "$(mamba shell hook --shell bash)"
mamba activate ./tts_env

python -u tts_comps_wugs_dist.py
