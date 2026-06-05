#!/bin/bash
#SBATCH --job-name=mlops-train
#SBATCH --output=/ceph/project/MLOPS_KLS/runs/slurm_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
# NOTE: GPU/mem/CPU counts are overridden at submit time by slurm/submit_train.sh
#       which reads multi_gpu from final_train.config.yaml.
#SBATCH --begin=now

# AI-LAB L4 limits per GPU: 15 CPUs, 24 GB RAM
#   1 GPU -> 15 CPUs, 24G   |   2 GPUs -> 30 CPUs, 48G

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

# Code is already synced by Jenkins rsync before this job was submitted
# Install project requirements into the venv
echo "=== Installing requirements | $(date) ==="
singularity exec --nv --bind $PROJECT:/app $CONTAINER \
    pip install -r /app/requirements.txt --quiet

# Ensure full dataset is checked out from DVC cache
# Controlled by skip_dvc_checkout in config
# set to False only when dataset has changed
SKIP_DVC=$(singularity exec --bind $PROJECT:/app $CONTAINER \
    python3 -c "import yaml; print(yaml.safe_load(open('/app/config/final_train.config.yaml'))['settings']['skip_dvc_checkout'])")

if [ "$SKIP_DVC" = "True" ]; then
    echo "=== DVC Checkout SKIPPED (skip_dvc_checkout=True in config) | $(date) ==="
else
    echo "=== DVC Checkout | $(date) ==="

    # Remove surface locks left by killed jobs, but KEEP the SQLite database
    rm -f .dvc/tmp/lock .dvc/tmp/rwlock
    rm -f .git/index.lock

    # Disable DVC analytics prompt to prevent silent headless hanging
    export DVC_NO_ANALYTICS=true

    # Pull new data
    # echo "Pulling dataset updates from remote..."
    # singularity exec $CONTAINER \
    #     $VENV/bin/dvc pull -v

    # Checkout data directly from Ceph cache
    singularity exec $CONTAINER \
        $VENV/bin/dvc checkout -v
fi

# Read multi_gpu setting from config to decide single vs DDP launch
NUM_GPUS=$(singularity exec --bind $PROJECT:/app $CONTAINER \
    python3 -c "import yaml; print(yaml.safe_load(open('/app/config/final_train.config.yaml'))['training']['multi_gpu'])")

echo "Training with NUM_GPUS=$NUM_GPUS (SBATCH allocated: 2 GPUs)"

echo "=== Starting Training | multi_gpu=$NUM_GPUS | $(date) ==="

#  check if NUM_GPUS is greater than 1 to decide between single GPU or DDP multi-GPU training
if [ "$NUM_GPUS" -gt 1 ]; then
    # DDP multi-GPU training via torchrun
    echo "Launching DDP training on $NUM_GPUS GPUs with torchrun"
    # --nv passes CUDA_VISIBLE_DEVICES automatically — don't override it explicitly
    singularity exec --nv --bind $PROJECT:/app \
        $CONTAINER \
        bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml \
        torchrun --nproc_per_node=$NUM_GPUS \
        --master_addr=localhost --master_port=29500 \
        src/main.py"
else
    # Single GPU training
    echo "Launching single GPU training"
    singularity exec --nv --bind $PROJECT:/app $CONTAINER \
        bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml python src/main.py"
fi

# Version the trained model with DVC
echo "=== DVC model versioning | $(date) ==="

# GENTLE CLEAN again before adding the model, just in case
rm -f .dvc/tmp/lock .dvc/tmp/rwlock
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

echo "=== Training job complete | $(date) ==="
echo ""
echo "To view MLflow results, run this on your local PC:"
echo "  ssh -L 5000:$(hostname):5000 ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk"
echo "  mlflow ui --backend-store-uri sqlite:////ceph/project/MLOPS_KLS/mlflow.db --host 0.0.0.0 --port 5000"
echo "Then open: http://localhost:5000"