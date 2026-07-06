#!/bin/bash
# Evaluate AudioGround on CM_test and/or UnAV-100.
#
# Usage:
#   bash run_eval_audioground.sh [CKPT_PATH] [DATASET]
#
#   CKPT_PATH : path to training checkpoint (default: output/audioground/checkpoint_best.pth)
#   DATASET   : cm_test | unav100 | all  (default: all)
#
# Examples:
#   bash run_eval_audioground.sh
#   bash run_eval_audioground.sh output/audioground/checkpoint_best.pth cm_test
#   bash run_eval_audioground.sh output/audioground/checkpoint_0.pth all

set -e

CONDA_ENV=qwen2audio-grpo
GPUS=${AUDIOGROUND_GPUS:-"4,5,6,7"}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

CKPT=${1:-"output/audioground/checkpoint_best.pth"}
DATASET=${2:-"all"}

cd "$(dirname "$0")"

run_eval() {
    local dataset=$1
    echo ""
    echo "========================================"
    echo " Evaluating: $dataset"
    echo " Checkpoint: $CKPT"
    echo " GPUs: $GPUS  (nproc=$NPROC)"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$GPUS \
    conda run -n $CONDA_ENV --no-capture-output \
      torchrun \
        --nproc_per_node=$NPROC \
        --master_port=29520 \
        eval_audioground.py \
          --dataset $dataset \
          --ckpt "$CKPT" \
          --batch 8 \
          --out output/eval
}

if [ "$DATASET" = "all" ]; then
    run_eval cm_test
    run_eval unav100
else
    run_eval "$DATASET"
fi
