#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="/workspace/MemVLN"
QWEN3_PATH="/workspace/ckpts/Qwen3-VL-4B-Instruct"
STREAMVGGT_PATH="/workspace/ckpts/StreamVGGT-1B.pth"
DATA_BASE="/workspace/data/leo2_data/vln"
OUTPUT_BASE="/workspace/outputs"

RUN_NOTE="qwen3_4b_full_fifo_nb100_dummy_off_seed42_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/${RUN_NOTE}.log"
DIST_PORT=3013

test -f "$QWEN3_PATH/model.safetensors.index.json"
test -f "$STREAMVGGT_PATH"
test -f "$DATA_BASE/vln/r2r/train/train_gt.json.gz"
test -f "$DATA_BASE/vln/rxr/train_v2/annotations.json"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE/wandb"
cd "$CODE_DIR"

nohup env \
    CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    NCCL_SOCKET_IFNAME=bond0 \
    NCCL_SOCKET_FAMILY=AF_INET \
    NCCL_DEBUG=INFO \
    WANDB_MODE=offline \
    WANDB_DIR="$OUTPUT_BASE/wandb" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    python launch.py \
        --mode accelerate \
        --config configs/default.yaml \
        --num_nodes 1 \
        --gpu_per_node 8 \
        --cpu_per_task 16 \
        --strategy ddp \
        --mixed_precision bf16 \
        --port "$DIST_PORT" \
        llm.cfg_path="$QWEN3_PATH" \
        vision3d.ckpt_path="$STREAMVGGT_PATH" \
        data.annotation_base="$DATA_BASE" \
        task.training.epochs=1 \
        task.training.dummy_lr_decay=false \
        llm.mem_module.memory_consolidate_type=fifo \
        llm.mem_module.episode_max_frame_num=100 \
        base_dir="$OUTPUT_BASE" \
        note="$RUN_NOTE" \
        rng_seed=42 \
        '~task.vlnce' \
    >"$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f '$LOG_FILE'"
