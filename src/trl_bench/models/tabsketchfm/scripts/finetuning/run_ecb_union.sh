#!/bin/bash
# ==============================================================================
# ECB-Union Finetuning Workflow
# ==============================================================================
# This script runs the complete finetuning workflow for ECB-Union task.
# ECB-Union is a REGRESSION task where the label represents how many dimensions
# differ between pairs of ECB data slices (ranges from 1 to 12).
#
# The European Central Bank (ECB) organizes economic data into distinct datasets
# with multiple dimensions. Slices that share more dimensions are more comparable.
# This benchmark ranks each pair of slices by how many dimensions differ.
#
# Usage (from project root):
#   # With pretrained TabSketchFM checkpoint (default)
#   bash models/tabsketchfm/scripts/finetuning/run_ecb_union.sh
#
#   # With raw BERT (no tabular pretraining - baseline)
#   bash models/tabsketchfm/scripts/finetuning/run_ecb_union.sh --model_name_or_path bert-base-uncased
#
#   # With custom checkpoint
#   bash models/tabsketchfm/scripts/finetuning/run_ecb_union.sh --model_name_or_path path/to/checkpoint.ckpt
#
#   # Skip preprocessing (if already done)
#   bash models/tabsketchfm/scripts/finetuning/run_ecb_union.sh --skip_preprocessing
# ==============================================================================

set -e

# ==============================================================================
# RESOLVE ABSOLUTE PATHS
# ==============================================================================
# DDP (Distributed Data Parallel) spawns worker processes that need absolute
# paths to work correctly. We resolve all paths relative to the project root.

# Get the project root (where this script is run from, should be TRL root)
PROJECT_ROOT="$(pwd)"

# TabSketchFM directory (absolute)
TABSKETCHFM_DIR="${PROJECT_ROOT}/models/tabsketchfm"

# Load environment
if [ -f "${PROJECT_ROOT}/load_env" ]; then
    source "${PROJECT_ROOT}/load_env"
fi

# Set PYTHONPATH with ABSOLUTE paths for DDP compatibility
export PYTHONPATH="${TABSKETCHFM_DIR}:${TABSKETCHFM_DIR}/tabsketchfm:${PYTHONPATH:-}"

# ==============================================================================
# DEFAULT CONFIGURATION (all paths absolute)
# ==============================================================================

# Model checkpoint (can be .ckpt file or model name like bert-base-uncased)
MODEL_NAME_OR_PATH="${PROJECT_ROOT}/checkpoints/tabsketchfm/epoch=10-step=27786.ckpt"

# Dataset paths (absolute)
DATASET_DIR="${PROJECT_ROOT}/datasets/ecb_union"
INPUT_TABLES="${DATASET_DIR}/tables"
LABELS_FILE="${DATASET_DIR}/labels.json"

# Preprocessed data directory (absolute)
DATA_DIR="${TABSKETCHFM_DIR}/processed_dataset/ecb_union"

# Output directory for finetuned model (absolute)
OUTPUT_DIR="${PROJECT_ROOT}/assets/finetuned_models/tabsketchfm/ecb_union"

# Training parameters
DEVICES=4
MAX_EPOCHS=50
BATCH_SIZE=64
RANDOM_SEED=0
SKIP_PREPROCESSING=""

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            # Convert to absolute if relative path given
            if [[ "$2" == /* ]]; then
                MODEL_NAME_OR_PATH="$2"
            else
                MODEL_NAME_OR_PATH="${PROJECT_ROOT}/$2"
            fi
            shift 2
            ;;
        --data_dir)
            if [[ "$2" == /* ]]; then
                DATA_DIR="$2"
            else
                DATA_DIR="${PROJECT_ROOT}/$2"
            fi
            shift 2
            ;;
        --output_dir)
            if [[ "$2" == /* ]]; then
                OUTPUT_DIR="$2"
            else
                OUTPUT_DIR="${PROJECT_ROOT}/$2"
            fi
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
            echo "ECB-Union finetuning workflow (regression task)."
            echo ""
            echo "ECB-Union predicts how many dimensions differ between pairs of ECB"
            echo "data slices. Labels range from 1 to 12 (continuous regression)."
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

# ==============================================================================
# DETERMINE MODEL MODE
# ==============================================================================

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

# ==============================================================================
# DISPLAY CONFIGURATION
# ==============================================================================

echo "======================================================================"
echo "TABSKETCHFM ECB-UNION FINETUNING"
echo "======================================================================"
echo "Project root:       $PROJECT_ROOT"
echo "Model:              $MODEL_NAME_OR_PATH"
echo "Input tables:       $INPUT_TABLES"
echo "Labels:             $LABELS_FILE"
echo "Preprocessed data:  $DATA_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "Devices:            $DEVICES"
echo "Max epochs:         $MAX_EPOCHS"
echo "Batch size:         $BATCH_SIZE"
echo "Random seed:        $RANDOM_SEED"
echo "Task:               Regression (dimension difference, labels 1-12)"
echo "======================================================================"
echo ""

# ==============================================================================
# VALIDATE INPUT FILES
# ==============================================================================

if [[ ! -d "$INPUT_TABLES" ]]; then
    echo "Error: Input tables directory not found: $INPUT_TABLES"
    exit 1
fi

if [[ ! -f "$LABELS_FILE" ]]; then
    echo "Error: Labels file not found: $LABELS_FILE"
    exit 1
fi

# ==============================================================================
# STEP 1: PREPROCESSING (optional)
# ==============================================================================

if [[ -z "$SKIP_PREPROCESSING" ]]; then
    # Check if preprocessed data already exists
    if [ -d "$DATA_DIR" ] && [ "$(find "$DATA_DIR" -name "*.json.bz2" 2>/dev/null | head -1)" ]; then
        NUM_EXISTING=$(find "$DATA_DIR" -name "*.json.bz2" | wc -l)
        echo "======================================================================"
        echo "STEP 1: SKIPPING PREPROCESSING (preprocessed data exists)"
        echo "======================================================================"
        echo "Found $NUM_EXISTING preprocessed files in $DATA_DIR"
        echo ""
    else
        echo "======================================================================"
        echo "STEP 1: PREPROCESSING TABLES"
        echo "======================================================================"
        echo "Input:  $INPUT_TABLES"
        echo "Output: $DATA_DIR"
        echo "======================================================================"
        echo ""

        mkdir -p "$DATA_DIR"

        python "${TABSKETCHFM_DIR}/data_processing/batch_fastdata.py" \
            --input_dir "$INPUT_TABLES" \
            --output_dir "$DATA_DIR"

        if [ $? -ne 0 ]; then
            echo "Preprocessing failed!"
            exit 1
        fi

        NUM_FILES=$(find "$DATA_DIR" -name "*.json.bz2" | wc -l)
        echo ""
        echo "Preprocessing complete: $NUM_FILES files generated"
        echo ""
    fi
else
    echo "======================================================================"
    echo "STEP 1: SKIPPING PREPROCESSING (--skip_preprocessing flag set)"
    echo "======================================================================"
    echo "Using existing preprocessed data in $DATA_DIR"
    echo ""
fi

# ==============================================================================
# STEP 2: FINETUNING
# ==============================================================================

echo "======================================================================"
echo "STEP 2: FINETUNING"
echo "======================================================================"
echo "Model:      $MODEL_NAME_OR_PATH"
echo "Data:       $DATA_DIR"
echo "Labels:     $LABELS_FILE"
echo "Task type:  regression"
echo "Output:     $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the command with ABSOLUTE paths for DDP compatibility
CMD="python ${TABSKETCHFM_DIR}/finetune.py"
CMD="$CMD --model_name_or_path $BASE_MODEL"
CMD="$CMD $CHECKPOINT_ARG"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --dataset $LABELS_FILE"
CMD="$CMD --task_type regression"
CMD="$CMD --num_labels 1"
CMD="$CMD --accelerator gpu"
CMD="$CMD --devices $DEVICES"
CMD="$CMD --max_epochs $MAX_EPOCHS"
CMD="$CMD --train_batch_size $BATCH_SIZE"
CMD="$CMD --val_batch_size $BATCH_SIZE"
CMD="$CMD --default_root_dir $OUTPUT_DIR"
CMD="$CMD --random_seed $RANDOM_SEED"

echo "Running: $CMD"
echo ""
eval $CMD

if [ $? -ne 0 ]; then
    echo "Finetuning failed!"
    exit 1
fi

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
echo "======================================================================"
echo "ECB-UNION FINETUNING COMPLETE!"
echo "======================================================================"
echo "Finetuned model:    $OUTPUT_DIR"
echo "Preprocessed data:  $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Extract embeddings using the finetuned checkpoint"
echo "  2. Run downstream evaluation tasks"
echo ""
echo "To re-run with different hyperparameters:"
echo "  bash models/tabsketchfm/scripts/finetuning/run_ecb_union.sh \\"
echo "    --skip_preprocessing --batch_size 32 --max_epochs 100"
echo "======================================================================"
