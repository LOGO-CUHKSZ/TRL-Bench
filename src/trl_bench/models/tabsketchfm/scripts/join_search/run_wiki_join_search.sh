#!/bin/bash
# ==============================================================================
# Wiki-Join-Search Pipeline (Jaccard Similarity)
# ==============================================================================
# This script runs the complete join search pipeline for wiki-join-search dataset.
#
# Steps:
#   1. Preprocess tables
#   2. Extract embeddings from pretrained model
#   3. Convert embeddings to legacy format
#   4. Convert JSONL labels to pickle format
#   5. Run join search
#
# Usage:
#   bash scripts/join_search/run_wiki_join_search.sh [--skip_preprocessing] [--skip_extraction]
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Pretrained model
MODEL_PATH="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"

# Data paths
RAW_TABLES_DIR="wiki-join-search/tables"
LABELS_FILE="wiki-join-search/labels/join_search_jaccard_gt.jsonl"

# Processed data
PROCESSED_DIR="wiki_join_search_processed"
EMBEDDINGS_UNIFIED="embeddings/wiki_join_search_embeddings_unified.pkl"
EMBEDDINGS_LEGACY="embeddings/wiki_join_search_embeddings_legacy.pkl"
GROUND_TRUTH_PKL="embeddings/wiki_join_search_ground_truth.pkl"

# Search parameters
K=10
MIN_SCORE=0.5  # Aligned with TabSketchFM paper (Jaccard threshold for Wiki-Join-Search)

# Output
SEARCH_RESULTS="results/wiki_join_search_results.pkl"

# Flags
SKIP_PREPROCESSING=false
SKIP_EXTRACTION=false

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip_extraction)
            SKIP_EXTRACTION=true
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip_preprocessing] [--skip_extraction] [--model PATH] [--k N]"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "WIKI-JOIN-SEARCH PIPELINE"
echo "======================================================================"
echo "Model:           $MODEL_PATH"
echo "Tables:          $RAW_TABLES_DIR"
echo "Labels:          $LABELS_FILE"
echo "Top-K:           $K"
echo "======================================================================"
echo ""

# ==============================================================================
# STEP 1: PREPROCESS TABLES
# ==============================================================================

if [ "$SKIP_PREPROCESSING" = true ]; then
    echo "⏭️  STEP 1: Skipping preprocessing (--skip_preprocessing)"
else
    echo "📋 STEP 1: Preprocessing tables..."
    echo "----------------------------------------------------------------------"

    mkdir -p "$PROCESSED_DIR"

    python tabsketchfm/batch_fastdata.py \
        --input_dir "$RAW_TABLES_DIR" \
        --output_dir "$PROCESSED_DIR"

    NUM_FILES=$(find "$PROCESSED_DIR" -name "*.json.bz2" | wc -l)
    echo "✅ Preprocessed $NUM_FILES tables"
fi

echo ""

# ==============================================================================
# STEP 2: EXTRACT EMBEDDINGS
# ==============================================================================

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "⏭️  STEP 2: Skipping extraction (--skip_extraction)"
else
    echo "🔬 STEP 2: Extracting embeddings from pretrained model..."
    echo "----------------------------------------------------------------------"

    mkdir -p embeddings

    python scripts/embedding_extraction/extract_embeddings_unified.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_type pretrained \
        --data_dir "$PROCESSED_DIR" \
        --output_file "$EMBEDDINGS_UNIFIED" \
        --batch_size 512

    echo "✅ Embeddings extracted to: $EMBEDDINGS_UNIFIED"
fi

echo ""

# ==============================================================================
# STEP 3: CONVERT EMBEDDINGS TO LEGACY FORMAT
# ==============================================================================

echo "🔄 STEP 3: Converting embeddings to legacy format..."
echo "----------------------------------------------------------------------"

python scripts/utils/convert_embeddings_format.py \
    --input "$EMBEDDINGS_UNIFIED" \
    --output "$EMBEDDINGS_LEGACY"

echo "✅ Legacy embeddings saved to: $EMBEDDINGS_LEGACY"
echo ""

# ==============================================================================
# STEP 4: CONVERT LABELS TO PICKLE FORMAT
# ==============================================================================

echo "🔄 STEP 4: Converting labels to pickle format..."
echo "----------------------------------------------------------------------"

# Note: Don't use --keep_csv_extension for union=False (direct column search)
python scripts/utils/convert_wiki_join_labels.py \
    --input "$LABELS_FILE" \
    --output "$GROUND_TRUTH_PKL" \
    --min_score "$MIN_SCORE"

echo "✅ Ground truth saved to: $GROUND_TRUTH_PKL"
echo ""

# ==============================================================================
# STEP 5: RUN JOIN SEARCH
# ==============================================================================

echo "🔍 STEP 5: Running join search..."
echo "----------------------------------------------------------------------"

mkdir -p results

python embedding_search.py \
    --embeddings "$EMBEDDINGS_LEGACY" \
    --ground_truth "$GROUND_TRUTH_PKL" \
    --data_dir "$RAW_TABLES_DIR" \
    --k "$K" \
    --by_cols True \
    --use_column_based_table_search False \
    --outfile "$SEARCH_RESULTS"

echo "✅ Search results saved to: $SEARCH_RESULTS"
echo ""

# ==============================================================================
# STEP 6: EVALUATE RESULTS
# ==============================================================================

echo "📊 STEP 6: Evaluating search results..."
echo "----------------------------------------------------------------------"

python scripts/join_search/evaluate_join_search.py \
    --results "$SEARCH_RESULTS" \
    --ground_truth "$GROUND_TRUTH_PKL" \
    --k_values "1,5,10,20,50" \
    --output "results/wiki_join_search_metrics.json"

echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "======================================================================"
echo "WIKI-JOIN-SEARCH PIPELINE COMPLETE!"
echo "======================================================================"
echo "Preprocessed:    $PROCESSED_DIR"
echo "Embeddings:      $EMBEDDINGS_LEGACY"
echo "Ground truth:    $GROUND_TRUTH_PKL"
echo "Results:         $SEARCH_RESULTS"
echo "Metrics:         results/wiki_join_search_metrics.json"
echo ""
echo "Next steps:"
echo "  1. Compare with containment metric"
echo "  2. Try with finetuned model"
echo "  3. Review the metrics"
echo "======================================================================"
