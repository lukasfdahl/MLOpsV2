#!/bin/bash
# submit_train.sh
# Reads multi_gpu from final_train.config.yaml and submits train_job.sh
# with the correct GPU/CPU/memory allocation.
# Usage: bash slurm/submit_train.sh

PROJECT=/ceph/project/MLOPS_KLS
CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif

NUM_GPUS=$(singularity exec --bind $PROJECT:/app $CONTAINER \
    python3 -c "import yaml; print(yaml.safe_load(open('/app/config/final_train.config.yaml'))['training']['multi_gpu'])")

MEM=$(( NUM_GPUS * 24 ))
CPUS=$(( NUM_GPUS * 15 ))

echo "Submitting training job: $NUM_GPUS GPU(s), ${MEM}G RAM, $CPUS CPUs"

sbatch \
    --gres=gpu:${NUM_GPUS} \
    --mem=${MEM}G \
    --cpus-per-task=${CPUS} \
    $PROJECT/slurm/train_job.sh
