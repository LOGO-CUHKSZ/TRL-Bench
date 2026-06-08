#!/bin/bash
# ==============================================================================
# Wiki-Union Decoupled Pipeline
# ==============================================================================
# This script demonstrates the fully decoupled approach for wiki-union:
# 1. Generate embeddings from pretrained model (once)
# 2. Train classifier on paired embeddings (fast, can iterate)
#
# Usage:
#   bash scripts/tasks/run_wiki_union.sh [--skip_extraction] [--model PATH]
#
# Options:
#   --skip_extraction    Skip embedding extraction if already done
#   --model PATH         Use specific pretrained model (default: pretrained checkpoint)
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
MODEL_PATH="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"

# Data paths
DATA_DIR="wiki_union_processed"
LABELS_FILE="wiki_union/labels.json"

# Output paths
EMBEDDINGS_DIR="embeddings"
EMBEDDINGS_FILE="${EMBEDDINGS_DIR}/wiki_union_embeddings.pkl"
RESULTS_DIR="results/wiki_union_decoupled"

# Training parameters
TASK_NAME="wiki_union"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip_extraction] [--model PATH] [--embedding_type TYPE] [--combination_method METHOD] [--results_dir DIR]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# PHASE 1: EXTRACT EMBEDDINGS
# ==============================================================================

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "======================================================================"
    echo "SKIPPING EMBEDDING EXTRACTION (--skip_extraction flag set)"
    echo "======================================================================"
    echo "Using existing embeddings: $EMBEDDINGS_FILE"
    echo ""
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
echo "Task:               $TASK_NAME"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Architecture:       input_dim → $HIDDEN_DIM → $NUM_LABELS"
echo "Results:            $RESULTS_DIR"
echo "======================================================================"
echo ""

# Train classifier on paired embeddings
python scripts/tasks/run_task.py \
    --embeddings "$EMBEDDINGS_FILE" \
    --labels "$LABELS_FILE" \
    --task_name "$TASK_NAME" \
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
echo "WIKI-UNION DECOUPLED PIPELINE COMPLETE!"
echo "======================================================================"
echo "Embeddings:  $EMBEDDINGS_FILE"
echo "Results:     $RESULTS_DIR"
echo "Summary:     $RESULTS_DIR/results.json"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/results.json"
echo ""
echo "To run with different hyperparameters (embeddings already extracted):"
echo "  bash scripts/tasks/run_wiki_union.sh --skip_extraction"
echo "======================================================================"
