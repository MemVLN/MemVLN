#!/usr/bin/env bash
set -euo pipefail

# R2R-only Full: geometry + memory + trajectory summarization + auxiliary masking.
# Compare bank capacity 50 with the FIFO-100 Full control; keep readout 300.
# Matched to train_ppu_r2r_full.sh: old episode sampler, dummy LR off.
# Queue A/B may override GPU_IDS and DIST_PORT; this script still launches eight ranks.
CODE_DIR="${CODE_DIR:-/workspace/MemVLN}"
QWEN3_PATH="${QWEN3_PATH:-/workspace/ckpts/Qwen3-VL-4B-Instruct}"
STREAMVGGT_PATH="${STREAMVGGT_PATH:-/workspace/ckpts/StreamVGGT-1B.pth}"
DATA_BASE="${DATA_BASE:-/workspace/data/leo2_data/vln}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/outputs}"
GPU_IDS="${GPU_IDS:-8,9,10,11,12,13,14,15}"
DIST_PORT="${DIST_PORT:-3013}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
CONSOLIDATION="fifo"  # Fixed for this named ablation.

[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be a positive integer" >&2; exit 2; }

RUN_NOTE="qwen3_4b_r2r_full_${CONSOLIDATION}_nb50_nm300_dummy_off_e${EPOCHS}_seed${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/${RUN_NOTE}.log"

for REQUIRED_FILE in \
    "$CODE_DIR/launch.py" \
    "$QWEN3_PATH/model.safetensors.index.json" \
    "$STREAMVGGT_PATH" \
    "$DATA_BASE/vln/r2r/train/train_gt.json.gz"; do
    [[ -f "$REQUIRED_FILE" && -r "$REQUIRED_FILE" ]] || {
        echo "Missing or unreadable file: $REQUIRED_FILE" >&2
        exit 1
    }
done

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
        task=sft_r2r_w_traj_sum \
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
        'task.dataset_mix.VLNDatasetMix.ratio=[1,1]' \
        data.vln.memory_prompt=true \
        data.vln.obs_hist_sample=nearest \
        data.vln.max_num_frames=16 \
        data.vln.aug_stop_steps=false \
        llm.mem_module.param_dtype=bf16 \
        llm.mem_module.memory_consolidate_type="$CONSOLIDATION" \
        llm.mem_module.episode_max_frame_num=50 \
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
echo "Setting: $CONSOLIDATION; Nb=50; Nm=300; merge_vggt=false; aux_mask=1"
echo "GPUs: $GPU_IDS; port: $DIST_PORT"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f '$LOG_FILE'"

