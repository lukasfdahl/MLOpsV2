#!/bin/bash
# submit_train.sh
# Reads multi_gpu from final_train.config.yaml and submits train_job.sh
# with the correct GPU/CPU/memory allocation.
# Usage: bash slurm/submit_train.sh

PROJECT=/ceph/project/MLOPS_KLS
CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif

# Read multi_gpu directly with grep — no singularity, no warnings
NUM_GPUS=$(grep 'multi_gpu:' $PROJECT/config/final_train.config.yaml | awk '{print $2}')

echo "Read NUM_GPUS='${NUM_GPUS}' from config"
MEM=$(( NUM_GPUS * 24 ))
CPUS=$(( NUM_GPUS * 15 ))

echo "Submitting training job: $NUM_GPUS GPU(s), ${MEM}G RAM, $CPUS CPUs"

sbatch \
    --gres=gpu:${NUM_GPUS} \
    --mem=${MEM}G \
    --cpus-per-task=${CPUS} \
    $PROJECT/slurm/train_job.sh
