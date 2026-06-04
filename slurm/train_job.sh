#!/bin/bash
#SBATCH --job-name=mlops-train
#SBATCH --output=/ceph/project/MLOPS_KLS/runs/slurm_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=15

# fail - exit!
set -e

# Load necessary modules
CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif
PROJECT=/ceph/project/MLOPS_KLS
VENV=~/mlops_venv

# Increase the maximum number of open files to avoid "Too many open files" errors with DataLoader workers
ulimit -n 65536

# Log job start
echo "=== Training Job | $(date) | $(hostname) ==="
cd $PROJECT

# Pull latest code first (Source of truth is GitHub)
echo "=== Syncing to latest code | $(date) ==="
git fetch origin development
git reset --hard origin/development

# Install project requirements into the venv
echo "=== Installing requirements | $(date) ==="
singularity exec --nv --bind $PROJECT:/app $CONTAINER \
    pip install -r /app/requirements.txt --quiet

# Ensure full dataset is checked out from DVC cache
echo "=== DVC Checkout | $(date) ==="

# 1. NUKE the entire temporary folder to clear ALL stuck SQLite locks and state
rm -rf .dvc/tmp/*
rm -f .git/index.lock

# 2. Disable DVC analytics prompt to prevent silent headless hanging
export DVC_NO_ANALYTICS=true

# 3. Pull new data 
# echo "Pulling dataset updates from remote..."
# singularity exec $CONTAINER \
#     $VENV/bin/dvc pull -v

# 4. Checkout data directly from Ceph cache
singularity exec $CONTAINER \
    $VENV/bin/dvc checkout -v

# Read multi_gpu setting from config to decide single vs DDP launch
NUM_GPUS=$(singularity exec $CONTAINER \
    python3 -c "import yaml; print(yaml.safe_load(open('/ceph/project/MLOPS_KLS/config/final_train.config.yaml'))['training']['multi_gpu'])")

echo "=== Starting Training | multi_gpu=$NUM_GPUS | $(date) ==="

#  check if NUM_GPUS is greater than 1 to decide between single GPU or DDP multi-GPU training
if [ "$NUM_GPUS" -gt 1 ]; then
    # DDP multi-GPU training via torchrun
    echo "Launching DDP training on $NUM_GPUS GPUs with torchrun"
    singularity exec --nv --bind $PROJECT:/app $CONTAINER \
        bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml \
        torchrun --nproc_per_node=$NUM_GPUS src/main.py"
else
    # Single GPU training
    echo "Launching single GPU training"
    singularity exec --nv --bind $PROJECT:/app $CONTAINER \
        bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml python src/main.py"
fi

# Version the trained model with DVC
echo "=== DVC model versioning | $(date) ==="

# Remove locks again just in case the training step caused a weird state
rm -rf .dvc/tmp/*
rm -f .git/index.lock

singularity exec $CONTAINER \
    $VENV/bin/dvc add runs/models/best_model.pth

singularity exec $CONTAINER \
    $VENV/bin/dvc push

# Commit the updated .dvc pointer file back to git
git config user.email "ailab@mlops"
git config user.name "AI-LAB"
git pull origin development --rebase
git add runs/models/best_model.pth.dvc
git commit -m "model update: new best_model from training run"
git push origin development

# Launch MLflow UI for experiment tracking
echo "=== Starting MLflow UI | $(date) ==="
echo "To view results run on your local PC:"
echo "  ssh -L 5000:$(hostname):5000 ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk"
echo "Then open: http://localhost:5000"
singularity exec /ceph/container/pytorch/pytorch_25.09.sif \
    python -m mlflow ui --backend-store-uri sqlite:////ceph/project/MLOPS_KLS/mlflow.db --host 0.0.0.0 --port 5000