#!/bin/bash
# ==============================================================================
# CKAN Subset Decoupled Pipeline
# ==============================================================================
# This script demonstrates the fully decoupled approach for CKAN subset:
# 1. Generate embeddings from pretrained model (once)
# 2. Train classifier on paired embeddings (fast, can iterate)
#
# Usage:
#   bash scripts/tasks/run_ckan_subset.sh [--skip_extraction] [--model PATH]
#
# Options:
#   --skip_extraction    Skip embedding extraction if already done
#   --model PATH         Use specific pretrained model (default: pretrained checkpoint)
#   --embedding_type     Type of embedding (cls, table, column_mean) [default: cls]
#   --combination        Combination method (concat, add, multiply, diff) [default: concat]
#   --results_dir        Output directory for results
# ==============================================================================

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default model (pretrained checkpoint)
MODEL_PATH="logs/tabsketchfm-pretrain-filtered/tabsketchfm-pretrain/bn6oq8sa/checkpoints/epoch=11-step=9084.ckpt"

# Data paths
DATA_DIR="ckan_subset_processed"
LABELS_FILE="ckan_subset/labels.json"

# Output paths
EMBEDDINGS_DIR="embeddings"
EMBEDDINGS_FILE="${EMBEDDINGS_DIR}/ckan_subset_embeddings.pkl"
RESULTS_DIR="results/ckan_subset_decoupled"

# Training parameters
TASK_NAME="ckan_subset"
TASK_TYPE="classification"
NUM_LABELS=2
HIDDEN_DIM=256
MAX_EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=2e-5
RANDOM_SEED=0

# Embedding parameters
EMBEDDING_TYPE="cls"
COMBINATION_METHOD="concat"
EXTRACTION_BATCH_SIZE=256

# Flags
SKIP_EXTRACTION=false

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --embedding_type)
            EMBEDDING_TYPE="$2"
            shift 2
            ;;
        --combination_method)
            COMBINATION_METHOD="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip_extraction           Skip embedding extraction phase"
            echo "  --model PATH                Path to pretrained model checkpoint"
            echo "  --embedding_type TYPE       Embedding type: cls, table, column_mean (default: cls)"
            echo "  --combination_method METHOD Combination: concat, add, multiply, diff (default: concat)"
            echo "  --results_dir DIR           Output directory for results"
            echo "  --hidden_dim DIM            Hidden layer size (default: 256)"
            echo "  --batch_size SIZE           Training batch size (default: 32)"
            echo "  --learning_rate LR          Learning rate (default: 2e-5)"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Verify required files exist
if [ ! -f "$LABELS_FILE" ]; then
    echo "ERROR: Labels file not found: $LABELS_FILE"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please run preprocessing first:"
    echo "  bash scripts/finetuning/run_ckan_subset.sh"
    exit 1
fi

if [ "$SKIP_EXTRACTION" = false ] && [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# ==============================================================================
# PHASE 1: EXTRACT EMBEDDINGS
# ==============================================================================

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "======================================================================"
    echo "SKIPPING EMBEDDING EXTRACTION (--skip_extraction flag set)"
    echo "======================================================================"
    echo "Using existing embeddings: $EMBEDDINGS_FILE"
    echo ""

    if [ ! -f "$EMBEDDINGS_FILE" ]; then
        echo "ERROR: Embeddings file not found: $EMBEDDINGS_FILE"
        echo "Please run without --skip_extraction first to generate embeddings"
        exit 1
    fi
else
    echo "======================================================================"
    echo "PHASE 1: EXTRACTING EMBEDDINGS"
    echo "======================================================================"
    echo "Model:           $MODEL_PATH"
    echo "Data:            $DATA_DIR"
    echo "Output:          $EMBEDDINGS_FILE"
    echo "Batch size:      $EXTRACTION_BATCH_SIZE"
    echo "======================================================================"
    echo ""

    # Create embeddings directory
    mkdir -p "$EMBEDDINGS_DIR"

    # Extract embeddings using unified extraction script
    python scripts/embedding_extraction/extract_embeddings_unified.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type pretrained \
        --data_dir "$DATA_DIR" \
        --output_file "$EMBEDDINGS_FILE" \
        --batch_size "$EXTRACTION_BATCH_SIZE"

    echo ""
    echo "✅ Phase 1 complete: Embeddings saved to $EMBEDDINGS_FILE"
    echo ""
fi

# ==============================================================================
# PHASE 2: TRAIN CLASSIFIER
# ==============================================================================

echo "======================================================================"
echo "PHASE 2: TRAINING CLASSIFIER"
echo "======================================================================"
echo "Embeddings:         $EMBEDDINGS_FILE"
echo "Labels:             $LABELS_FILE"
echo "Task:               $TASK_NAME (binary classification)"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Architecture:       input_dim → $HIDDEN_DIM → $NUM_LABELS"
echo "Batch size:         $BATCH_SIZE"
echo "Learning rate:      $LEARNING_RATE"
echo "Max epochs:         $MAX_EPOCHS"
echo "Results:            $RESULTS_DIR"
echo "======================================================================"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Train classifier on paired embeddings
python scripts/tasks/run_task.py \
    --embeddings "$EMBEDDINGS_FILE" \
    --labels "$LABELS_FILE" \
    --task_name "$TASK_NAME" \
    --task_type "$TASK_TYPE" \
    --output_dir "$RESULTS_DIR" \
    --embedding_type "$EMBEDDING_TYPE" \
    --combination_method "$COMBINATION_METHOD" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_labels "$NUM_LABELS" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --random_seed "$RANDOM_SEED" \
    --accelerator gpu \
    --devices 1

echo ""
echo "======================================================================"
echo "CKAN SUBSET DECOUPLED PIPELINE COMPLETE!"
echo "======================================================================"
echo "Embeddings:  $EMBEDDINGS_FILE"
echo "Results:     $RESULTS_DIR"
echo "Summary:     $RESULTS_DIR/results.json"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/results.json"
echo ""
echo "To run with different hyperparameters (embeddings already extracted):"
echo "  bash scripts/tasks/run_ckan_subset.sh --skip_extraction --hidden_dim 512"
echo ""
echo "To compare embedding types:"
echo "  bash scripts/tasks/run_ckan_subset.sh --skip_extraction --embedding_type table"
echo "  bash scripts/tasks/run_ckan_subset.sh --skip_extraction --embedding_type column_mean"
echo ""
echo "To compare combination methods:"
echo "  bash scripts/tasks/run_ckan_subset.sh --skip_extraction --combination_method add"
echo "  bash scripts/tasks/run_ckan_subset.sh --skip_extraction --combination_method multiply"
echo "======================================================================"
