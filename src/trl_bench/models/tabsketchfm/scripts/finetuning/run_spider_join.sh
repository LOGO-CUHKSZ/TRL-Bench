#!/bin/bash
# ==============================================================================
# Spider-Join Finetuning Workflow
# ==============================================================================
# This script runs the complete finetuning workflow for Spider-Join task.
#
# Usage (from TRL project root):
#   # With pretrained TabSketchFM checkpoint (default)
#   bash models/tabsketchfm/scripts/finetuning/run_spider_join.sh
#
#   # With raw BERT (no tabular pretraining - baseline)
#   bash models/tabsketchfm/scripts/finetuning/run_spider_join.sh --model_name_or_path bert-base-uncased
#
#   # With custom checkpoint
#   bash models/tabsketchfm/scripts/finetuning/run_spider_join.sh --model_name_or_path path/to/model
#
#   # Skip preprocessing (if already done)
#   bash models/tabsketchfm/scripts/finetuning/run_spider_join.sh --skip_preprocessing
# ==============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is 4 levels up: finetuning -> scripts -> tabsketchfm -> models -> trl
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# TabSketchFM module directory
TABSKETCHFM_DIR="$PROJECT_ROOT/models/tabsketchfm"

cd "$PROJECT_ROOT"

# Set PYTHONPATH to include TabSketchFM modules
export PYTHONPATH="$TABSKETCHFM_DIR:$TABSKETCHFM_DIR/tabsketchfm:$PYTHONPATH"

# ==============================================================================
# Default parameters
# ==============================================================================
# MODEL_NAME_OR_PATH can be:
#   - a HuggingFace model directory (e.g., checkpoints/tabsketchfm/bert_model)
#   - a .ckpt file (Lightning checkpoint)
#   - a model name (e.g., bert-base-uncased for raw BERT baseline)
MODEL_NAME_OR_PATH="checkpoints/tabsketchfm/bert_model"
DATA_DIR="datasets/tabsketchfm/spider_join_processed_dataset"
LABELS_FILE="datasets/tabsketchfm/spider_join/spider-join/labels.json"
INPUT_TABLES_DIR="datasets/tabsketchfm/spider_join/spider-join/tables"
OUTPUT_DIR="logs/spider_join_finetuned"
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
            echo "Spider-Join finetuning workflow."
            echo ""
            echo "Options:"
            echo "  --model_name_or_path PATH  HuggingFace model dir, .ckpt checkpoint, or model name"
            echo "                             (default: checkpoints/tabsketchfm/bert_model)"
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
    echo "Mode: Lightning checkpoint"
elif [[ -d "$MODEL_NAME_OR_PATH" ]]; then
    BASE_MODEL="$MODEL_NAME_OR_PATH"
    echo "Mode: HuggingFace model directory"
else
    BASE_MODEL="$MODEL_NAME_OR_PATH"
    echo "Mode: Model name ($MODEL_NAME_OR_PATH)"
fi

echo "================================================"
echo "TabSketchFM: Spider-Join Finetuning Workflow"
echo "================================================"
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Data directory: $DATA_DIR"
echo "Labels file: $LABELS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Devices: $DEVICES"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "================================================"
echo ""

# Step 1: Preprocessing (optional)
if [[ -z "$SKIP_PREPROCESSING" ]]; then
    echo "Step 1: Preprocessing spider-join tables..."
    echo "-------------------------------------------"
    mkdir -p "$DATA_DIR"

    python "$TABSKETCHFM_DIR/tabsketchfm/batch_fastdata.py" \
        --input_dir "$INPUT_TABLES_DIR" \
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
echo "Step 2: Finetuning on spider-join task..."
echo "-------------------------------------------"

# Build the command
CMD="python $TABSKETCHFM_DIR/finetune.py"
CMD="$CMD --model_name_or_path $BASE_MODEL"
CMD="$CMD $CHECKPOINT_ARG"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --dataset $LABELS_FILE"
CMD="$CMD --task_type classification"
CMD="$CMD --num_labels 2"
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
echo "Finetuned model saved in: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Extract embeddings: python $TABSKETCHFM_DIR/extract_embeddings.py --checkpoint $OUTPUT_DIR/checkpoints/best.ckpt"
echo "  2. Perform search: python $TABSKETCHFM_DIR/embedding_search.py --embeddings embeddings.pkl --k 10"
echo ""
