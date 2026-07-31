#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Default values
GPU_COUNT=4
CONFIG_NAME="default"
MODEL_PATH=""
EPOCH_TAG=0

# Help function
usage() {
    echo "Usage: $0 [options] [hydra_overrides...]"
    echo "Options:"
    echo "  -g, --gpus       Number of GPUs to use (and splits to create). Default: 4"
    echo "  -c, --config     Hydra config name (without .yaml). Default: default"
    echo "  -m, --model      Path to the checkpoint file. (Required)"
    echo "  -e, --epoch      Epoch tag for result folder name. Default: 0"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -g 4 -m checkpoints/best.pth -e 0 task.vlnce.R2R.args.save_video=10"
    exit 1
}

# Parse named arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g|--gpus)
            GPU_COUNT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -e|--epoch)
            EPOCH_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            # Collect unknown options (assumed to be Hydra overrides)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check requirements
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path (-m) is required."
    usage
fi
if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPU count must be a positive integer, got '$GPU_COUNT'."
    exit 2
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Split evaluation requires an exact checkpoint file: $MODEL_PATH"
    exit 2
fi

echo "========================================================"
echo "Starting Evaluation:"
echo "  GPUs/Splits: $GPU_COUNT"
echo "  Config Name: $CONFIG_NAME"
echo "  Checkpoint:  $MODEL_PATH"
echo "  Epoch Tag:   $EPOCH_TAG"
echo "  Overrides:   ${POSITIONAL_ARGS[*]}"
echo "========================================================"

# Launch Parallel Jobs
PIDS=()
for IDX in $(seq 0 $((GPU_COUNT-1))); do
    GPU_ID=$(( IDX % GPU_COUNT ))
    
    echo "Launching Split $IDX on GPU $GPU_ID ..."
    
    # We pass the named args as Python flags (--split-id ...)
    # We pass the POSITIONAL_ARGS (overrides) at the end for Hydra
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_DIR/run_eval_split.py" \
        --config-name "$CONFIG_NAME" \
        --model-path "$MODEL_PATH" \
        --split-id $IDX \
        --split-num $GPU_COUNT \
        --epoch "$EPOCH_TAG" \
        "${POSITIONAL_ARGS[@]}" &
    PIDS+=("$!")
    
    # Slight delay to prevent CPU spikes during initialization
    sleep 2
done

FAILED=0
for IDX in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$IDX]}"; then
        echo "Error: Split $IDX failed."
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "One or more evaluation splits failed; results are incomplete."
    exit 1
fi

echo "All splits finished. Results saved in eval_results/<dataset>/epoch_$EPOCH_TAG"
