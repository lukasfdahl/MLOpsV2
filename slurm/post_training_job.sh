#!/bin/bash
#SBATCH --job-name=mlops-post-train
#SBATCH --output=/ceph/project/MLOPS_KLS/runs/slurm_post_%j.log
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=ailab-l4-04
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --begin=now

set -e

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif
PROJECT=/ceph/project/MLOPS_KLS

echo "=== Post-Training Optimization | $(date) | $(hostname) ==="
cd $PROJECT

echo "=== Installing requirements | $(date) ==="
singularity exec --bind $PROJECT:/app $CONTAINER \
    pip install -r /app/requirements.txt --quiet

echo "=== Running quantization, pruning sweep and fine-tune | $(date) ==="
singularity exec --nv --bind $PROJECT:/app $CONTAINER \
    bash -c "cd /app && TRAIN_CONFIG=config/final_train.config.yaml python src/post_training_runner.py"

# Version the optimized model with DVC
echo "=== DVC versioning optimized model | $(date) ==="
rm -f .dvc/tmp/lock .dvc/tmp/rwlock .git/index.lock
singularity exec $CONTAINER \
    ~/mlops_venv/bin/dvc add runs/models/optimized/optimized_weights.pth

singularity exec $CONTAINER \
    ~/mlops_venv/bin/dvc push

# Verify the optimized model is in the remote before committing its pointer to git,
# so git never references data the remote does not have.
echo "=== Verifying optimized model is in DVC remote before committing pointer | $(date) ==="
PUSH_STATUS=$(singularity exec $CONTAINER ~/mlops_venv/bin/dvc status -c 2>&1 || true)
echo "$PUSH_STATUS"
if echo "$PUSH_STATUS" | grep -q "optimized_weights.pth"; then
    echo "ERROR: optimized_weights.pth still missing from the remote after push — aborting before committing pointer."
    exit 1
fi

git config user.email "ailab@mlops"
git config user.name "AI-LAB"
git add -A
git stash
git pull origin development --rebase
git stash pop
git add runs/models/optimized/optimized_weights.pth.dvc
git commit -m "model update: optimized (pruned+quantized) weights"
git push origin development

echo "=== Post-training complete | $(date) ==="