#!/bin/bash
# ==============================================================================
# Embedding Generation Wrapper (Task-Agnostic)
# ==============================================================================
# Simple wrapper for embedding generation using TabSketchFM models.
#
# Usage (from TRL project root):
#   bash models/tabsketchfm/scripts/tasks/generate_embeddings.sh \
#       --model checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
#       --data_dir datasets/tabsketchfm/spider_join_processed_dataset \
#       --output embeddings/spider_join_embeddings.pkl
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

# Default configuration
MODEL_NAME_OR_PATH="checkpoints/tabsketchfm/epoch=10-step=27786.ckpt"
DATA_DIR="datasets/tabsketchfm/spider_join_processed_dataset"
OUTPUT_FILE="embeddings/pretrained_embeddings.pkl"
BATCH_SIZE=256

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH       Model checkpoint path (default: checkpoints/tabsketchfm/epoch=10-step=27786.ckpt)"
            echo "  --data_dir DIR     Directory with preprocessed tables (default: datasets/tabsketchfm/spider_join_processed_dataset)"
            echo "  --output FILE      Output embeddings file (default: embeddings/pretrained_embeddings.pkl)"
            echo "  --batch_size N     Batch size for extraction (default: 256)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "======================================================================"
echo "GENERATING EMBEDDINGS (Task-Agnostic)"
echo "======================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Model:        $MODEL_NAME_OR_PATH"
echo "Data:         $DATA_DIR"
echo "Output:       $OUTPUT_FILE"
echo "Batch size:   $BATCH_SIZE"
echo "======================================================================"

python "$TABSKETCHFM_DIR/scripts/embedding_extraction/extract_embeddings_unified_optimized.py" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --model_type pretrained \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --prefetch_factor 4

echo ""
echo "Embeddings generated: $OUTPUT_FILE"
