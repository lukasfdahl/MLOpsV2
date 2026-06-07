#!/bin/bash
#SBATCH --job-name=dvc-pull
#SBATCH --output=/ceph/project/MLOPS_KLS/slurm/pull_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif
PROJECT=/ceph/project/MLOPS_KLS

echo "=== DVC Pull | $(date) ==="
cd $PROJECT

singularity exec $CONTAINER \
    ~/mlops_venv/bin/dvc pull --jobs 16 -v

echo "=== DVC Checkout | $(date) ==="
singularity exec $CONTAINER \
    ~/mlops_venv/bin/dvc checkout

echo "=== Done | $(date) ==="