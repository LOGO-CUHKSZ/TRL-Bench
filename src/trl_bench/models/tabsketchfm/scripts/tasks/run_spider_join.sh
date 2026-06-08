#!/bin/bash
# ==============================================================================
# Spider-Join Decoupled Pipeline
# ==============================================================================
# This script demonstrates the fully decoupled approach for spider-join:
#   1. Preprocess tables (if needed)
#   2. Generate embeddings from pretrained model (once, reusable)
#   3. Train classifier on paired embeddings (fast, can iterate)
#
# Usage (from TRL project root):
#   bash models/tabsketchfm/scripts/tasks/run_spider_join.sh
#
#   # With specific checkpoint
#   bash models/tabsketchfm/scripts/tasks/run_spider_join.sh \
#       --model checkpoints/tabsketchfm/epoch=10-step=27786.ckpt
#
#   # Skip embedding generation if already done
#   bash models/tabsketchfm/scripts/tasks/run_spider_join.sh --skip_generation
# ==============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is 4 levels up: tasks -> scripts -> tabsketchfm -> models -> trl
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# TabSketchFM module directory
TABSKETCHFM_DIR="$PROJECT_ROOT/models/tabsketchfm"

cd "$PROJECT_ROOT"

# Load environment
if [ -f "load_env" ]; then
    source load_env
fi

# Set PYTHONPATH to include TabSketchFM modules
export PYTHONPATH="$TABSKETCHFM_DIR:$TABSKETCHFM_DIR/tabsketchfm:$PYTHONPATH"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model and data paths (relative to project root)
MODEL_PATH="checkpoints/tabsketchfm/epoch=10-step=27786.ckpt"
RAW_TABLES_DIR="datasets/tabsketchfm/spider_join/spider-join/tables"
PROCESSED_DIR="datasets/tabsketchfm/spider_join_processed_dataset"

# Labels
LABELS_FILE="datasets/tabsketchfm/spider_join/spider-join/labels.json"

# Embeddings
EMBEDDINGS_FILE="embeddings/spider_join_embeddings.pkl"

# Output
RESULTS_DIR="results/spider_join_decoupled"

# Training parameters
TASK_NAME="spider_join"
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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Spider-Join decoupled pipeline."
            echo ""
            echo "Options:"
            echo "  --model PATH               Model checkpoint (default: checkpoints/tabsketchfm/epoch=10-step=27786.ckpt)"
            echo "  --skip_generation          Skip embedding extraction (use existing embeddings)"
            echo "  --embedding_type TYPE      Embedding type: cls, table, column (default: cls)"
            echo "  --combination_method MTD   How to combine table pairs: concat, diff, hadamard (default: concat)"
            echo "  --results_dir DIR          Output directory for results"
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

echo "======================================================================"
echo "SPIDER-JOIN DECOUPLED PIPELINE"
echo "======================================================================"
echo "Project root:       $PROJECT_ROOT"
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
    echo "STEP 1: Preprocessing tables..."
    echo "----------------------------------------------------------------------"

    mkdir -p "$PROCESSED_DIR"

    python "$TABSKETCHFM_DIR/tabsketchfm/batch_fastdata.py" \
        --input_dir "$RAW_TABLES_DIR" \
        --output_dir "$PROCESSED_DIR"

    NUM_FILES=$(find "$PROCESSED_DIR" -name "*.json.bz2" | wc -l)
    echo "Preprocessed $NUM_FILES tables"
    echo ""
else
    echo "STEP 1: Skipping preprocessing (already exists: $PROCESSED_DIR)"
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
    bash "$TABSKETCHFM_DIR/scripts/tasks/generate_embeddings.sh" \
        --model "$MODEL_PATH" \
        --data_dir "$PROCESSED_DIR" \
        --output "$EMBEDDINGS_FILE" \
        --batch_size "$EXTRACTION_BATCH_SIZE"

    echo ""
    echo "Phase 1 complete: Embeddings saved to $EMBEDDINGS_FILE"
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
echo "Architecture:       input_dim -> $HIDDEN_DIM -> $NUM_LABELS"
echo "Results:            $RESULTS_DIR"
echo "======================================================================"
echo ""

# Train classifier on paired embeddings
python "$TABSKETCHFM_DIR/scripts/tasks/run_task.py" \
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
echo "SPIDER-JOIN DECOUPLED PIPELINE COMPLETE!"
echo "======================================================================"
echo "Embeddings:  $EMBEDDINGS_FILE"
echo "Results:     $RESULTS_DIR"
echo "Summary:     $RESULTS_DIR/results.json"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/results.json"
echo ""
echo "To run with different hyperparameters (embeddings already extracted):"
echo "  bash models/tabsketchfm/scripts/tasks/run_spider_join.sh --skip_generation"
echo ""
echo "To try different embedding combinations:"
echo "  bash models/tabsketchfm/scripts/tasks/run_spider_join.sh --skip_generation --combination_method diff"
echo "======================================================================"
