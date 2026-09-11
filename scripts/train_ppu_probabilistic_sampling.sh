#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="/workspace/MemVLN"
QWEN3_PATH="/workspace/ckpts/Qwen3-VL-4B-Instruct"
STREAMVGGT_PATH="/workspace/ckpts/StreamVGGT-1B.pth"
DATA_BASE="/workspace/data/leo2_data/vln"
OUTPUT_BASE="/workspace/outputs"

# Match the old 8-GPU, batchsize=1, gradient_accumulation_steps=1 run:
# "Scheduler Init: Active=172330, Skipped=0, Total=172330".
# This is the length after the old sampler trimmed ranks to equal item counts.
MAX_UPDATES=172330
# Old 8-rank scheduler: 400 ticks / 8 = 50 actual optimizer updates.
WARMUP_UPDATES=50

RUN_NOTE="qwen3_4b_full_frme_nb100_dataset_p50_dummy_off_u${MAX_UPDATES}_seed42_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/${RUN_NOTE}.log"
DIST_PORT=3012

test -f "$QWEN3_PATH/model.safetensors.index.json"
test -f "$STREAMVGGT_PATH"
test -f "$DATA_BASE/vln/r2r/train/train_gt.json.gz"
test -f "$DATA_BASE/vln/rxr/train_v2/annotations.json"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE/wandb"
cd "$CODE_DIR"

nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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
        task=sft_r2r_rxr_w_traj_sum \
        llm=qwen3-4b-mem \
        vision3d=streamvggt \
        llm.cfg_path="$QWEN3_PATH" \
        vision3d.ckpt_path="$STREAMVGGT_PATH" \
        data.annotation_base="$DATA_BASE" \
        pretrained_ckpt_path="" \
        task.training.epochs=1 \
        task.training.lr=2e-5 \
        task.training.dummy_lr_decay=false \
        training.max_updates="$MAX_UPDATES" \
        training.gradient_accumulation_steps=1 \
        training.schedule.args.warmup_steps="$WARMUP_UPDATES" \
        training.save_interval=5000 \
        dataloader.train.batchsize=1 \
        dataloader.train.num_workers=1 \
        task.dataset_mix.VLNDatasetMix.use_custom_sampler=true \
        task.dataset_mix.VLNDatasetMix.ratio_unit=episode \
        'task.dataset_mix.VLNDatasetMix.ratio=[1,1,1,1]' \
        task.dataset_mix.VLNDatasetMix.sampling.mode=probabilistic_episode \
        task.dataset_mix.VLNDatasetMix.sampling.dataset_probabilities.r2r=0.5 \
        task.dataset_mix.VLNDatasetMix.sampling.dataset_probabilities.rxr=0.5 \
        data.vln.memory_prompt=true \
        llm.mem_module.memory_consolidate_type=frme \
        llm.mem_module.episode_max_frame_num=100 \
        llm.mem_module.frame_max_tokens_num=100 \
        llm.mem_module.memory_token_num=300 \
        base_dir="$OUTPUT_BASE" \
        note="$RUN_NOTE" \
        rng_seed=42 \
        '~task.vlnce' \
    >"$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "PID: $PID"
echo "Max optimizer updates: $MAX_UPDATES"
echo "Warmup optimizer updates: $WARMUP_UPDATES"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f '$LOG_FILE'"
