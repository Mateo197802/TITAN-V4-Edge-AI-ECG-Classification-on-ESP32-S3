#!/bin/bash
#SBATCH --job-name=titan-plateau-audit
#SBATCH --partition=gpu-dev
#SBATCH --exclude=compute-0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2
exec "$HOME/V4_CEDIA_LINK/.venv_audit/bin/python" -u audit_plateau_cedia.py
