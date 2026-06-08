#!/bin/bash
# ==============================================================================
# ECB-Join Decoupled Pipeline
# ==============================================================================
# This script demonstrates the fully decoupled approach for ecb-join:
#   1. Preprocess tables (if needed)
#   2. Generate embeddings from pretrained model (once, reusable)
#   3. Train classifier on paired embeddings (fast, can iterate)
#
# Usage:
#   bash scripts/tasks/run_ecb_join.sh [--skip_generation] [--model PATH]
#
# Options:
#   --skip_generation    Skip embedding extraction if already done
#   --model PATH         Use specific pretrained model (default: pretrained checkpoint)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment
if [ -f "load_env" ]; then
    source load_env
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model and data
MODEL_PATH="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"
RAW_TABLES_DIR="ecb_join/tables"
PROCESSED_DIR="ecb_join_processed"

# Labels
LABELS_FILE="ecb_join/labels.json"

# Embeddings
EMBEDDINGS_FILE="embeddings/ecb_join_embeddings.pkl"

# Output
RESULTS_DIR="results/ecb_join_decoupled"

# Training parameters
TASK_NAME="ecb_join"
NUM_LABELS=56
HIDDEN_DIM=512
MAX_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=2e-5
RANDOM_SEED=0

# Embedding parameters
EMBEDDING_TYPE="cls"
COMBINATION_METHOD="concat"
EXTRACTION_BATCH_SIZE=256

# Flags
SKIP_GENERATION=false

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_generation)
            SKIP_GENERATION=true
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
            echo "Usage: $0 [--skip_generation] [--model PATH] [--embedding_type TYPE] [--combination_method METHOD] [--results_dir DIR]"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "ECB-JOIN DECOUPLED PIPELINE"
echo "======================================================================"
echo "Model:              $MODEL_PATH"
echo "Data:               $RAW_TABLES_DIR"
echo "Labels:             $LABELS_FILE"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Results:            $RESULTS_DIR"
echo "======================================================================"
echo ""

# ==============================================================================
# STEP 1: PREPROCESS TABLES (if needed)
# ==============================================================================

if [ ! -d "$PROCESSED_DIR" ]; then
    echo "📋 STEP 1: Preprocessing tables..."
    echo "----------------------------------------------------------------------"

    mkdir -p "$PROCESSED_DIR"

    python tabsketchfm/batch_fastdata.py \
        --input_dir "$RAW_TABLES_DIR" \
        --output_dir "$PROCESSED_DIR"

    NUM_FILES=$(find "$PROCESSED_DIR" -name "*.json.bz2" | wc -l)
    echo "✅ Preprocessed $NUM_FILES tables"
    echo ""
else
    echo "⏭️  STEP 1: Skipping preprocessing (already exists)"
    echo ""
fi

# ==============================================================================
# STEP 2: EXTRACT EMBEDDINGS
# ==============================================================================

if [ "$SKIP_GENERATION" = true ]; then
    echo "======================================================================"
    echo "SKIPPING EMBEDDING EXTRACTION (--skip_generation flag set)"
    echo "======================================================================"
    echo "Using existing embeddings: $EMBEDDINGS_FILE"
    echo ""
else
    echo "======================================================================"
    echo "PHASE 1: EXTRACTING EMBEDDINGS"
    echo "======================================================================"
    echo "Model:           $MODEL_PATH"
    echo "Data:            $PROCESSED_DIR"
    echo "Output:          $EMBEDDINGS_FILE"
    echo "Batch size:      $EXTRACTION_BATCH_SIZE"
    echo "======================================================================"
    echo ""

    mkdir -p embeddings

    # Extract embeddings using unified extraction script
    bash scripts/tasks/generate_embeddings.sh \
        --model "$MODEL_PATH" \
        --data_dir "$PROCESSED_DIR" \
        --output "$EMBEDDINGS_FILE" \
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

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "======================================================================"
echo "ECB-JOIN DECOUPLED PIPELINE COMPLETE!"
echo "======================================================================"
echo "Embeddings:  $EMBEDDINGS_FILE"
echo "Results:     $RESULTS_DIR"
echo "Summary:     $RESULTS_DIR/results.json"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/results.json"
echo ""
echo "To run with different hyperparameters (embeddings already extracted):"
echo "  bash scripts/tasks/run_ecb_join.sh --skip_generation"
echo ""
echo "To try different embedding combinations:"
echo "  bash scripts/tasks/run_ecb_join.sh --skip_generation --combination_method diff"
echo "======================================================================"
