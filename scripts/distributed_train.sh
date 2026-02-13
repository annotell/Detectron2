#!/bin/bash
set -euo pipefail

# Distributed training entrypoint for Detectron2 (Cascade Mask R-CNN)
# Supports both local (single machine) and GCP (multi-node K8s) modes.
#
# Required env vars (both modes):
#   DATA_ROOT       - Path to dataset directory (contains images/ and annos/)
#   CLASS_MAP       - Class mapping string, e.g. "Car:Vehicle,Person:Pedestrian"
#
# GCP mode env vars (set by K8s indexed Job + autobaan-training orchestrator):
#   TRAINING_MODE=gcp
#   WORLD_SIZE      - Total number of nodes
#   MASTER_ADDR     - DNS name of pod-0 (headless service)
#   MASTER_PORT     - NCCL master port (default: 29500)
#   JOB_COMPLETION_INDEX - Pod index (0..N-1), becomes node rank
#   GCS_DATASET_PATH    - gs:// path to download dataset from
#   GCS_CHECKPOINT_PATH - gs:// path to upload checkpoints to (master only)

MODE=${TRAINING_MODE:-local}
NUM_GPUS=${NUM_GPUS_PER_NODE:-1}

echo "=== GPU Diagnostics ==="
nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "======================="

if [ "$MODE" = "gcp" ]; then
    RANK=${JOB_COMPLETION_INDEX:-0}
    OUTPUT=${OUTPUT_DIR:-/checkpoints/${RUN_TAG:-run}}

    echo "[GCP] rank=$RANK | world_size=$WORLD_SIZE | master=$MASTER_ADDR:${MASTER_PORT:-29500}"

    # Download dataset from GCS to local emptyDir volume
    echo "Downloading dataset from $GCS_DATASET_PATH..."
    python /opt/gcs_helper.py download "$GCS_DATASET_PATH" /data

    # Run distributed training (no exec — need to upload checkpoints after)
    python /opt/Detectron2/training/train.py \
        --num-gpus "$NUM_GPUS" \
        --num-machines "$WORLD_SIZE" \
        --machine-rank "$RANK" \
        --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT:-29500}" \
        --data-root "$DATA_ROOT" \
        --class-map "$CLASS_MAP" \
        --output-dir "$OUTPUT"

    # Upload checkpoints to GCS (master pod only)
    if [ "$RANK" = "0" ]; then
        echo "Uploading checkpoints to $GCS_CHECKPOINT_PATH..."
        python /opt/gcs_helper.py upload "$OUTPUT" "$GCS_CHECKPOINT_PATH"
    fi
else
    OUTPUT=${OUTPUT_DIR:-${DATA_ROOT}/output_2d}

    echo "[Local] nproc=$NUM_GPUS"

    exec python /opt/Detectron2/training/train.py \
        --num-gpus "$NUM_GPUS" \
        --data-root "$DATA_ROOT" \
        --class-map "$CLASS_MAP" \
        --output-dir "$OUTPUT"
fi
