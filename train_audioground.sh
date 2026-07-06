#!/bin/bash
# Train AudioGround on 4 GPUs (4,5,6,7) using AudioGround-IT data.
# Paper settings: 6K steps, effective batch 24, LoRA r=32 α=64.

set -e

CONDA_ENV=qwen2audio-grpo
CFG=configs/audioground.yaml
GPUS=${AUDIOGROUND_GPUS:-"0,1,2,3,4,5,6,7"}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=$GPUS \
conda run -n $CONDA_ENV --no-capture-output \
  python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port=29510 \
    train.py \
    --cfg-path $CFG \
    --options run.world_size=$NPROC run.batch_size_train=2 run.accum_grad_iters=2
