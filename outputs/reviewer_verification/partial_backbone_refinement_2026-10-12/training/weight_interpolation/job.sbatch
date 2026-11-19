#!/bin/bash
#SBATCH --job-name=titan-weight-merge
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$HOME/V4_CEDIA_LINK/.venv_audit/bin/python" -u run_merge.py
