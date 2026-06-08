#!/bin/bash
# ==============================================================================
# Wiki-Join-Search Task Runner
# ==============================================================================
# This script demonstrates the fully decoupled approach for wiki-join-search:
#   1. Generate embeddings once (reusable across tasks)
#   2. Run join search task with those embeddings
#
# Usage:
#   bash scripts/tasks/run_wiki_join.sh [--skip_generation] [--model PATH]
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
RAW_TABLES_DIR="wiki-join-search/tables"
PROCESSED_DIR="wiki_join_search_processed"

# Ground truth
LABELS_FILE="wiki-join-search/labels/join_search_jaccard_gt.jsonl"
MIN_SCORE=0.5  # Aligned with TabSketchFM paper (Jaccard threshold for Wiki-Join-Search)

# Embeddings
EMBEDDINGS_FILE="embeddings/wiki_join_search_embeddings.pkl"
# EMBEDDINGS_FILE="embeddings/wiki_union_embeddings.pkl"

# Search parameters
K=10
K_VALUES="1,5,10"

# Output
RESULTS_DIR="results/wiki_join_search"

# Flags
SKIP_GENERATION=false
USE_GPU=false

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
        --k)
            K="$2"
            shift 2
            ;;
        --use_gpu)
            USE_GPU=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip_generation] [--model PATH] [--k N] [--use_gpu]"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "WIKI-JOIN-SEARCH TASK (Decoupled Approach)"
echo "======================================================================"
echo "Model:           $MODEL_PATH"
echo "Data:            $RAW_TABLES_DIR"
echo "Labels:          $LABELS_FILE"
echo "Top-K:           $K"
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
# STEP 2: GENERATE EMBEDDINGS
# ==============================================================================

if [ "$SKIP_GENERATION" = true ]; then
    echo "⏭️  STEP 2: Skipping embedding generation (--skip_generation)"
else
    echo "🔬 STEP 2: Generating embeddings..."
    echo "----------------------------------------------------------------------"

    mkdir -p embeddings

    # Use single-GPU extraction script
    bash scripts/tasks/generate_embeddings.sh \
        --model "$MODEL_PATH" \
        --data_dir "$PROCESSED_DIR" \
        --output "$EMBEDDINGS_FILE" \
        --batch_size 256

    echo "✅ Embeddings generated: $EMBEDDINGS_FILE"
fi

echo ""

# ==============================================================================
# STEP 3: RUN JOIN SEARCH TASK
# ==============================================================================

echo "🔍 STEP 3: Running join search task..."
echo "----------------------------------------------------------------------"

GPU_FLAG=""
if [ "$USE_GPU" = true ]; then
    GPU_FLAG="--use_gpu"
fi

python scripts/tasks/run_join_search.py \
    --embeddings "$EMBEDDINGS_FILE" \
    --ground_truth "$LABELS_FILE" \
    --ground_truth_format jsonl \
    --min_score "$MIN_SCORE" \
    --k "$K" \
    --k_values "$K_VALUES" \
    --output_dir "$RESULTS_DIR" \
    $GPU_FLAG

echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "======================================================================"
echo "WIKI-JOIN-SEARCH TASK COMPLETE!"
echo "======================================================================"
echo "Embeddings:      $EMBEDDINGS_FILE"
echo "Results:         $RESULTS_DIR/search_results.pkl"
echo "Metrics:         $RESULTS_DIR/metrics.json"
echo ""
echo "Key metrics from results/wiki_join_search/metrics.json:"
cat "$RESULTS_DIR/metrics.json"
echo ""
echo "Next steps:"
echo "  1. Try with finetuned model"
echo "  2. Review the Mean F1 / P@10 / R@10 metrics"
echo "  3. Try hybrid approach (TabSketchFM + SBERT)"
echo "======================================================================"
