#!/usr/bin/env bash
set -euo pipefail

# R2R-only: geometry_memory; without trajectory summarization.
# Old episode sampler, dummy LR off. Start a separate run for each variant.
CODE_DIR="${CODE_DIR:-/workspace/MemVLN}"
QWEN3_PATH="${QWEN3_PATH:-/workspace/ckpts/Qwen3-VL-4B-Instruct}"
STREAMVGGT_PATH="${STREAMVGGT_PATH:-/workspace/ckpts/StreamVGGT-1B.pth}"
DATA_BASE="${DATA_BASE:-/workspace/data/leo2_data/vln}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/outputs}"
GPU_IDS="${GPU_IDS:-8,9,10,11,12,13,14,15}"
DIST_PORT="${DIST_PORT:-3013}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
CONSOLIDATION="${CONSOLIDATION:-fifo}"
case "$CONSOLIDATION" in
    fifo|frme) ;;
    *) echo "CONSOLIDATION must be fifo or frme" >&2; exit 2 ;;
esac

[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer" >&2; exit 2; }

RUN_NOTE="qwen3_4b_r2r_geometry_memory_${CONSOLIDATION}_nb100_dummy_off_e${EPOCHS}_seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/${RUN_NOTE}.log"

test -f "$CODE_DIR/launch.py"
test -f "$QWEN3_PATH/model.safetensors.index.json"
test -f "$STREAMVGGT_PATH"
test -f "$DATA_BASE/vln/r2r/train/train_gt.json.gz"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE/wandb"
cd "$CODE_DIR"

nohup env \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
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
        task=sft_r2r \
        llm=qwen3-4b-mem \
        vision3d=streamvggt \
        mode=train \
        llm.cfg_path="$QWEN3_PATH" \
        vision3d.ckpt_path="$STREAMVGGT_PATH" \
        data.annotation_base="$DATA_BASE" \
        'pretrained_ckpt_path=""' \
        task.training.epochs="$EPOCHS" \
        task.training.lr=2e-5 \
        task.training.dummy_lr_decay=false \
        training.max_updates=null \
        training.gradient_accumulation_steps=1 \
        training.schedule.args.warmup_steps=400 \
        training.save_interval=5000 \
        training.checkpoint_total_limit=5 \
        dataloader.train.batchsize=1 \
        dataloader.train.num_workers=1 \
        task.dataset_mix.VLNDatasetMix.use_custom_sampler=true \
        task.dataset_mix.VLNDatasetMix.sampling.mode=epoch \
        task.dataset_mix.VLNDatasetMix.ratio_unit=sample \
        'task.dataset_mix.VLNDatasetMix.ratio=[1]' \
        data.vln.memory_prompt=true \
        data.vln.obs_hist_sample=nearest \
        data.vln.max_num_frames=16 \
        data.vln.aug_stop_steps=false \
        llm.mem_module.param_dtype=bf16 \
        llm.mem_module.memory_consolidate_type="$CONSOLIDATION" \
        llm.mem_module.episode_max_frame_num=100 \
        llm.mem_module.frame_max_tokens_num=100 \
        llm.mem_module.memory_token_num=300 \
        llm.mem_module.memory_merge_vggt_feat=false \
        llm.mem_module.visual_obs_mask_ratio_nav=0.0 \
        llm.mem_module.visual_obs_mask_ratio_aux=1.0 \
        base_dir="$OUTPUT_BASE" \
        note="$RUN_NOTE" \
        rng_seed="$SEED" \
        '~task.vlnce' \
    >"$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "PID: $PID"
echo "Run: $RUN_NOTE"
echo "GPUs: $GPU_IDS; port: $DIST_PORT"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f '$LOG_FILE'"

