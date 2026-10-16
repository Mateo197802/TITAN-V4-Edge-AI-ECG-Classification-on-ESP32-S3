#!/bin/bash
#SBATCH --job-name=titan-loss-ablation
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
exec "$HOME/V4_CEDIA_LINK/.venv_audit/bin/python" -u train_phase5.py
