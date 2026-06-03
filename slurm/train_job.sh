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

# Run training with final config (full dataset)
singularity exec --nv --bind $PROJECT:/app $CONTAINER \
    bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml python src/main.py"