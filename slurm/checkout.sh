#!/bin/bash
#SBATCH --job-name=dvc-checkout
#SBATCH --output=/ceph/project/MLOPS_KLS/slurm/checkout_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif
PROJECT=/ceph/project/MLOPS_KLS

echo "=== DVC Checkout | $(date) ==="
cd $PROJECT

singularity exec $CONTAINER \
    ~/mlops_venv/bin/dvc checkout

echo "=== Done | $(date) ==="
ls $PROJECT/data/dvc/ | wc -l
