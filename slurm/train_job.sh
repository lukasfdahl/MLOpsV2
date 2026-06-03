#!/bin/bash
#SBATCH --job-name=mlops-train
#SBATCH --output=/ceph/project/MLOPS_KLS/runs/slurm_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif
PROJECT=/ceph/project/MLOPS_KLS
VENV=~/mlops_venv

echo "=== Training Job | $(date) | $(hostname) ==="
cd $PROJECT

# Run training (Jenkins already rsynced latest code before submitting)
singularity exec --nv --bind $PROJECT:/app $CONTAINER \
    python /app/src/main.py