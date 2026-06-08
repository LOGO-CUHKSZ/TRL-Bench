#!/bin/bash
# ==============================================================================
# Union Search Regression
# ==============================================================================
# This script trains a regressor for union search using pre-extracted embeddings.
# It assumes embeddings have already been generated and simply trains the
# regression head.
#
# ECB-Union is a regression task where the label represents how many dimensions
# differ between pairs of ECB data slices (ranges from 1 to 12).
#
# Usage:
#   bash downstream_tasks/union_search/regression/run_regression.sh \
#       --embeddings <path> \
#       --labels <path> \
#       --output_dir <path>
#
# Options:
#   --embeddings PATH           Path to embeddings pickle file (required)
#   --labels PATH               Path to labels JSON file (required)
#   --output_dir PATH           Output directory for results (required)
#   --task_name NAME            Task name for logging (default: ecb_union)
#   --embedding_type TYPE       Embedding type: cls, table, column_mean (default: column_mean)
#   --combination_method METHOD How to combine pairs: concat, add, multiply, diff (default: concat)
#   --hidden_dim DIM            Hidden dimension for MLP (default: 256)
#   --batch_size SIZE           Batch size (default: 32)
#   --max_epochs NUM            Max epochs (default: 50)
#   --learning_rate LR          Learning rate (default: 1e-3)
#   --dropout_prob PROB         Dropout probability (default: 0.1)
#   --random_seed SEED          Random seed (default: 42)
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
TASK_NAME="ecb_union"
TASK_TYPE="regression"
EMBEDDING_TYPE="column_mean"
COMBINATION_METHOD="concat"
HIDDEN_DIM=256
NUM_LABELS=1  # Regression outputs single value
BATCH_SIZE=32
MAX_EPOCHS=50
LEARNING_RATE=1e-3
DROPOUT_PROB=0.1
RANDOM_SEED=42
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
        -h|--help)
            echo "Usage: $0 --embeddings PATH --labels PATH --output_dir PATH [options]"
            echo ""
            echo "Union Search Regression (model-agnostic)"
            echo ""
            echo "This script trains a regression model on pre-extracted table embeddings."
            echo "ECB-Union predicts the number of differing dimensions (1-12) between"
            echo "pairs of ECB data slices."
            echo ""
            echo "Required:"
            echo "  --embeddings PATH           Pickled embeddings file"
            echo "  --labels PATH               Labels JSON file (train/valid/test splits)"
            echo "  --output_dir PATH           Output directory for results"
            echo ""
            echo "Optional:"
            echo "  --task_name NAME            Task name (default: ecb_union)"
            echo "  --embedding_type TYPE       cls, table, column_mean (default: column_mean)"
            echo "  --combination_method METHOD concat, add, multiply, diff (default: concat)"
            echo "  --hidden_dim DIM            Hidden dimension (default: 256)"
            echo "  --batch_size SIZE           Batch size (default: 32)"
            echo "  --max_epochs NUM            Max epochs (default: 50)"
            echo "  --learning_rate LR          Learning rate (default: 1e-3)"
            echo "  --dropout_prob PROB         Dropout probability (default: 0.1)"
            echo "  --random_seed SEED          Random seed (default: 42)"
            echo "  --accelerator TYPE          gpu or cpu (default: gpu)"
            echo "  --devices NUM               Number of devices (default: 1)"
            exit 0
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

# ==============================================================================
# DISPLAY CONFIGURATION
# ==============================================================================

echo "======================================================================"
echo "UNION SEARCH REGRESSION"
echo "======================================================================"
echo "Embeddings:         $EMBEDDINGS_FILE"
echo "Labels:             $LABELS_FILE"
echo "Output:             $OUTPUT_DIR"
echo "Task:               $TASK_NAME"
echo "Task type:          $TASK_TYPE"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Architecture:       input_dim -> $HIDDEN_DIM -> 1 (regression)"
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
# CREATE OUTPUT DIRECTORY
# ==============================================================================

mkdir -p "$OUTPUT_DIR"

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
echo "REGRESSION COMPLETE!"
echo "======================================================================"
echo "Results saved to:   $OUTPUT_DIR"
echo "Summary file:       $OUTPUT_DIR/results.json"
echo ""
echo "Metrics:"
echo "  - test_mse: Mean Squared Error on test set"
echo "  - test_r2: R-squared (coefficient of determination)"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/results.json"
echo ""
echo "To try different hyperparameters:"
echo "  bash downstream_tasks/union_search/regression/run_regression.sh \\"
echo "    --embeddings $EMBEDDINGS_FILE \\"
echo "    --labels $LABELS_FILE \\"
echo "    --output_dir ${OUTPUT_DIR}_diff \\"
echo "    --combination_method diff \\"
echo "    --hidden_dim 512"
echo "======================================================================"
