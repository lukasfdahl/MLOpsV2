#!/bin/bash
# submit_train.sh
# Reads multi_gpu from final_train.config.yaml and submits train_job.sh
# with the correct GPU/CPU/memory allocation.
# Usage: bash slurm/submit_train.sh

PROJECT=/ceph/project/MLOPS_KLS
CONTAINER=/ceph/container/pytorch/pytorch_25.09.sif

# Suppress all warnings (e.g. pynvml FutureWarning) so only the value is captured
NUM_GPUS=$(singularity exec --bind $PROJECT:/app $CONTAINER \
    python3 -W ignore -c "import yaml; print(yaml.safe_load(open('/app/config/final_train.config.yaml'))['training']['multi_gpu'])" 2>/dev/null)

echo "Read NUM_GPUS='${NUM_GPUS}' from config"
MEM=$(( NUM_GPUS * 24 ))
CPUS=$(( NUM_GPUS * 15 ))

echo "Submitting training job: $NUM_GPUS GPU(s), ${MEM}G RAM, $CPUS CPUs"

sbatch \
    --gres=gpu:${NUM_GPUS} \
    --mem=${MEM}G \
    --cpus-per-task=${CPUS} \
    $PROJECT/slurm/train_job.sh
