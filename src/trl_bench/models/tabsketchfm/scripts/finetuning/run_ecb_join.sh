#!/bin/bash
# ==============================================================================
# ECB-Join Finetuning Workflow
# ==============================================================================
# This script runs the complete finetuning workflow for ECB-Join task.
# ECB-Join is a multi-class classification task with 56 classes representing
# different join column groups in European Central Bank statistical tables.
#
# Usage (from project root):
#   # With pretrained TabSketchFM checkpoint (default)
#   bash scripts/finetuning/run_ecb_join.sh
#
#   # With raw BERT (no tabular pretraining - baseline)
#   bash scripts/finetuning/run_ecb_join.sh --model_name_or_path bert-base-uncased
#
#   # With custom checkpoint
#   bash scripts/finetuning/run_ecb_join.sh --model_name_or_path path/to/checkpoint.ckpt
#
#   # Skip preprocessing (if already done)
#   bash scripts/finetuning/run_ecb_join.sh --skip_preprocessing
# ==============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is two levels up from scripts/finetuning/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default parameters
# MODEL_NAME_OR_PATH can be:
#   - a .ckpt file (pretrained TabSketchFM checkpoint)
#   - a model name (e.g., bert-base-uncased for raw BERT baseline)
MODEL_NAME_OR_PATH="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"
DATA_DIR="ecb_join_processed"
OUTPUT_DIR="./ecb_join_finetuned"
DEVICES=4
MAX_EPOCHS=50
BATCH_SIZE=64
RANDOM_SEED=0
SKIP_PREPROCESSING=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --skip_preprocessing)
            SKIP_PREPROCESSING=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "ECB-Join finetuning workflow (56-class classification)."
            echo ""
            echo "Options:"
            echo "  --model_name_or_path PATH  .ckpt checkpoint OR model name (e.g., bert-base-uncased)"
            echo "                             (default: pretrained TabSketchFM checkpoint)"
            echo "  --data_dir PATH            Directory with preprocessed data"
            echo "  --output_dir PATH          Output directory for finetuned model"
            echo "  --devices INT              Number of GPUs (default: 4)"
            echo "  --max_epochs INT           Maximum epochs (default: 50)"
            echo "  --batch_size INT           Batch size (default: 64)"
            echo "  --random_seed INT          Random seed (default: 0)"
            echo "  --skip_preprocessing       Skip data preprocessing step"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine if using checkpoint or raw model
IS_CHECKPOINT=false
CHECKPOINT_ARG=""
BASE_MODEL="bert-base-uncased"

if [[ "$MODEL_NAME_OR_PATH" == *.ckpt ]] && [[ -f "$MODEL_NAME_OR_PATH" ]]; then
    IS_CHECKPOINT=true
    CHECKPOINT_ARG="--checkpoint $MODEL_NAME_OR_PATH"
    echo "Mode: Pretrained TabSketchFM checkpoint"
else
    BASE_MODEL="$MODEL_NAME_OR_PATH"
    echo "Mode: Raw model ($MODEL_NAME_OR_PATH) - no tabular pretraining"
fi

echo "================================================"
echo "TabSketchFM: ECB-Join Finetuning Workflow"
echo "================================================"
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Devices: $DEVICES"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "Task: Multi-class classification (56 classes)"
echo "================================================"
echo ""

# Step 1: Preprocessing (optional)
if [[ -z "$SKIP_PREPROCESSING" ]]; then
    echo "Step 1: Preprocessing ecb-join tables..."
    echo "-------------------------------------------"
    mkdir -p "$DATA_DIR"

    python tabsketchfm/batch_fastdata.py \
        --input_dir ecb_join/tables \
        --output_dir "$DATA_DIR"

    if [ $? -ne 0 ]; then
        echo "Preprocessing failed!"
        exit 1
    fi

    # Check if preprocessing produced files
    NUM_FILES=$(find "$DATA_DIR" -name "*.json.bz2" | wc -l)
    echo "Preprocessing complete: $NUM_FILES files generated"
    echo ""
else
    echo "Skipping Step 1: Using existing preprocessed data in $DATA_DIR"
    echo ""
fi

# Step 2: Finetuning
echo "Step 2: Finetuning on ecb-join task..."
echo "-------------------------------------------"

# Build the command
CMD="python finetune.py"
CMD="$CMD --model_name_or_path $BASE_MODEL"
CMD="$CMD $CHECKPOINT_ARG"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --dataset ecb_join/labels.json"
CMD="$CMD --task_type classification"
CMD="$CMD --num_labels 56"
CMD="$CMD --accelerator gpu"
CMD="$CMD --devices $DEVICES"
CMD="$CMD --max_epochs $MAX_EPOCHS"
CMD="$CMD --train_batch_size $BATCH_SIZE"
CMD="$CMD --val_batch_size $BATCH_SIZE"
CMD="$CMD --default_root_dir $OUTPUT_DIR"
CMD="$CMD --random_seed $RANDOM_SEED"

echo "Running: $CMD"
eval $CMD

if [ $? -ne 0 ]; then
    echo "Finetuning failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "Workflow complete!"
echo "================================================"
echo ""
echo "Finetuned model saved in: $OUTPUT_DIR/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Extract embeddings: python extract_embeddings.py --checkpoint $OUTPUT_DIR/checkpoints/best.ckpt"
echo "  2. Perform search: python embedding_search.py --embeddings embeddings.pkl --k 10"
echo ""
