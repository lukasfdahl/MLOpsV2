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

# Ensure full dataset is checked out from DVC cache
echo "=== DVC Checkout | $(date) ==="
singularity exec $CONTAINER \
    $VENV/bin/dvc checkout

# Run training with final config (full dataset)
singularity exec --nv --bind $PROJECT:/app $CONTAINER \
    bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml python src/main.py"


# Version the trained model with DVC
echo "DVC model versioning | $(date)"
singularity exec $CONTAINER \
    $VENV/bin/dvc add runs/models/best_model.pth

singularity exec $CONTAINER \
    $VENV/bin/dvc push

# Commit the updated .dvc pointer file back to git
git config user.email "ailab@mlops"
git config user.name "AI-LAB"
git add runs/models/best_model.pth.dvc
git commit -m "model update: new best_model from training run"
git push origin development