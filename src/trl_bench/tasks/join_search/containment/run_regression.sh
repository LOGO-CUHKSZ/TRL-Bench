#!/bin/bash
# ==============================================================================
# Join Search Containment (Regression)
# ==============================================================================
# This script trains a regressor for containment estimation using pre-extracted
# embeddings. It assumes embeddings have already been generated and simply
# trains the regression head.
#
# Containment is formulated as a regression task to estimate containment value
# (R² metric) between table pairs.
#
# Usage:
#   bash downstream_tasks/join_search/containment/run_regression.sh \
#       --embeddings <path> \
#       --labels <path> \
#       --output_dir <path>
#
# Options:
#   --embeddings PATH           Path to embeddings pickle file (required)
#   --labels PATH               Path to labels JSON file (required)
#   --output_dir PATH           Output directory for results (required)
#   --task_name NAME            Task name for logging (default: wiki_containment)
#   --task_type TYPE            Task type: regression, classification (default: regression)
#   --embedding_type TYPE       Embedding type: cls, table, column_mean, column (default: column)
#   --combination_method METHOD How to combine pairs: concat, add, multiply, diff (default: concat)
#   --hidden_dim DIM            Hidden dimension for MLP (default: 256)
#   --num_labels NUM            Number of output labels (default: 1 for regression)
#   --batch_size SIZE           Batch size (default: 32)
#   --max_epochs NUM            Max epochs (default: 50)
#   --learning_rate LR          Learning rate (default: 2e-5)
#   --dropout_prob PROB         Dropout probability (default: 0.1)
#   --random_seed SEED          Random seed (default: 0)
#   --accelerator TYPE          Accelerator type: gpu, cpu (default: gpu)
#   --devices NUM               Number of devices (default: 1)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment
if [ -f "load_env" ]; then
    source load_env
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

# Required arguments (no defaults)
EMBEDDINGS_FILE=""
LABELS_FILE=""
OUTPUT_DIR=""

# Optional arguments with defaults
TASK_NAME="wiki_containment"
TASK_TYPE="regression"
EMBEDDING_TYPE="column"  # Use column-level embeddings for containment
COMBINATION_METHOD="concat"
HIDDEN_DIM=256
NUM_LABELS=1  # Regression outputs single continuous value
BATCH_SIZE=32
MAX_EPOCHS=50
LEARNING_RATE=2e-5
DROPOUT_PROB=0.1
RANDOM_SEED=0
ACCELERATOR="gpu"
DEVICES=1

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --embeddings)
            EMBEDDINGS_FILE="$2"
            shift 2
            ;;
        --labels)
            LABELS_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task_name)
            TASK_NAME="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
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
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num_labels)
            NUM_LABELS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --dropout_prob)
            DROPOUT_PROB="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --accelerator)
            ACCELERATOR="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --embeddings PATH --labels PATH --output_dir PATH [options]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# VALIDATE REQUIRED ARGUMENTS
# ==============================================================================

if [ -z "$EMBEDDINGS_FILE" ]; then
    echo "Error: --embeddings is required"
    echo "Usage: $0 --embeddings PATH --labels PATH --output_dir PATH [options]"
    exit 1
fi

if [ -z "$LABELS_FILE" ]; then
    echo "Error: --labels is required"
    echo "Usage: $0 --embeddings PATH --labels PATH --output_dir PATH [options]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required"
    echo "Usage: $0 --embeddings PATH --labels PATH --output_dir PATH [options]"
    exit 1
fi

# Check if embeddings file exists
if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "Error: Embeddings file not found: $EMBEDDINGS_FILE"
    exit 1
fi

# Check if labels file exists
if [ ! -f "$LABELS_FILE" ]; then
    echo "Error: Labels file not found: $LABELS_FILE"
    exit 1
fi

# Adjust NUM_LABELS based on task type if not explicitly set
if [ "$TASK_TYPE" = "classification" ] && [ "$NUM_LABELS" = "1" ]; then
    NUM_LABELS=2
fi

# ==============================================================================
# DISPLAY CONFIGURATION
# ==============================================================================

echo "======================================================================"
echo "CONTAINMENT ESTIMATION"
echo "======================================================================"
echo "Embeddings:         $EMBEDDINGS_FILE"
echo "Labels:             $LABELS_FILE"
echo "Output:             $OUTPUT_DIR"
echo "Task:               $TASK_NAME"
echo "Task type:          $TASK_TYPE"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Architecture:       input_dim → $HIDDEN_DIM → $NUM_LABELS"
echo "Batch size:         $BATCH_SIZE"
echo "Max epochs:         $MAX_EPOCHS"
echo "Learning rate:      $LEARNING_RATE"
echo "Dropout:            $DROPOUT_PROB"
echo "Random seed:        $RANDOM_SEED"
echo "Accelerator:        $ACCELERATOR"
echo "Devices:            $DEVICES"
echo "======================================================================"
echo ""

# ==============================================================================
# TRAIN REGRESSOR
# ==============================================================================

echo "Training regressor..."
echo ""

python utils/downstream/run_task.py \
    --embeddings "$EMBEDDINGS_FILE" \
    --labels "$LABELS_FILE" \
    --task_name "$TASK_NAME" \
    --task_type "$TASK_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --embedding_type "$EMBEDDING_TYPE" \
    --combination_method "$COMBINATION_METHOD" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_labels "$NUM_LABELS" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --dropout_prob "$DROPOUT_PROB" \
    --random_seed "$RANDOM_SEED" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES"

echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "======================================================================"
echo "CONTAINMENT ESTIMATION COMPLETE!"
echo "======================================================================"
echo "Results saved to:   $OUTPUT_DIR"
echo "Summary file:       $OUTPUT_DIR/results.json"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/results.json"
echo ""
echo "To try different hyperparameters:"
echo "  bash downstream_tasks/join_search/containment/run_regression.sh \\"
echo "    --embeddings $EMBEDDINGS_FILE \\"
echo "    --labels $LABELS_FILE \\"
echo "    --output_dir results/evaluation/join_search/tabsketchfm/wiki_containment_diff \\"
echo "    --combination_method diff \\"
echo "    --hidden_dim 512"
echo ""
echo "To run as classification instead:"
echo "  bash downstream_tasks/join_search/containment/run_regression.sh \\"
echo "    --embeddings $EMBEDDINGS_FILE \\"
echo "    --labels $LABELS_FILE \\"
echo "    --output_dir results/evaluation/join_search/tabsketchfm/wiki_containment_cls \\"
echo "    --task_type classification \\"
echo "    --num_labels 2"
echo "======================================================================"
