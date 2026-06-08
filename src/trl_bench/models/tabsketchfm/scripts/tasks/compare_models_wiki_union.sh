#!/bin/bash
# ==============================================================================
# Compare Different Models on Wiki-Union Task
# ==============================================================================
# This script demonstrates the power of the decoupled approach:
# Extract embeddings from different models, then compare them on the same task.
#
# Usage:
#   bash scripts/tasks/compare_models_wiki_union.sh
#
# This will generate:
#   - embeddings/wiki_union_pretrained.pkl (from pretrained TabSketchFM)
#   - embeddings/wiki_union_raw_bert.pkl (from raw BERT)
#   - results/wiki_union_pretrained/ (task results)
#   - results/wiki_union_raw_bert/ (task results)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
DATA_DIR="wiki_union_processed"
LABELS_FILE="wiki_union/labels.json"
EMBEDDINGS_DIR="embeddings"
PRETRAINED_MODEL="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"

echo "======================================================================"
echo "WIKI-UNION: MODEL COMPARISON"
echo "======================================================================"
echo "This will extract embeddings from two sources and compare them:"
echo "  1. Pretrained TabSketchFM"
echo "  2. Raw BERT (bert-base-uncased)"
echo "======================================================================"
echo ""

# Create directories
mkdir -p "$EMBEDDINGS_DIR"
mkdir -p "results"

# ==============================================================================
# EXTRACT EMBEDDINGS FROM PRETRAINED TABSKETCHFM
# ==============================================================================

echo "======================================================================"
echo "STEP 1: Extracting embeddings from PRETRAINED TabSketchFM"
echo "======================================================================"

if [ -f "$EMBEDDINGS_DIR/wiki_union_pretrained.pkl" ]; then
    echo "⚠️  Embeddings already exist: $EMBEDDINGS_DIR/wiki_union_pretrained.pkl"
    echo "   Skipping extraction. Delete file to re-extract."
else
    python scripts/embedding_extraction/extract_embeddings_unified.py \
        --model_name_or_path "$PRETRAINED_MODEL" \
        --model_type pretrained \
        --data_dir "$DATA_DIR" \
        --output_file "$EMBEDDINGS_DIR/wiki_union_pretrained.pkl" \
        --batch_size 256

    echo "✅ Pretrained embeddings saved"
fi

echo ""

# ==============================================================================
# EXTRACT EMBEDDINGS FROM RAW BERT
# ==============================================================================

echo "======================================================================"
echo "STEP 2: Extracting embeddings from RAW BERT"
echo "======================================================================"

if [ -f "$EMBEDDINGS_DIR/wiki_union_raw_bert.pkl" ]; then
    echo "⚠️  Embeddings already exist: $EMBEDDINGS_DIR/wiki_union_raw_bert.pkl"
    echo "   Skipping extraction. Delete file to re-extract."
else
    python scripts/embedding_extraction/extract_embeddings_unified.py \
        --model_name_or_path bert-base-uncased \
        --model_type pretrained \
        --data_dir "$DATA_DIR" \
        --output_file "$EMBEDDINGS_DIR/wiki_union_raw_bert.pkl" \
        --batch_size 256

    echo "✅ Raw BERT embeddings saved"
fi

echo ""

# ==============================================================================
# TRAIN CLASSIFIER WITH PRETRAINED EMBEDDINGS
# ==============================================================================

echo "======================================================================"
echo "STEP 3: Training classifier with PRETRAINED embeddings"
echo "======================================================================"

python scripts/tasks/run_task.py \
    --embeddings "$EMBEDDINGS_DIR/wiki_union_pretrained.pkl" \
    --labels "$LABELS_FILE" \
    --task_name wiki_union_pretrained \
    --output_dir results/wiki_union_pretrained \
    --embedding_type cls \
    --combination_method concat \
    --hidden_dim 256 \
    --num_labels 2 \
    --batch_size 32 \
    --max_epochs 50 \
    --learning_rate 2e-5 \
    --random_seed 0 \
    --accelerator gpu \
    --devices 1

echo ""
echo "✅ Pretrained results saved to: results/wiki_union_pretrained/"
echo ""

# ==============================================================================
# TRAIN CLASSIFIER WITH RAW BERT EMBEDDINGS
# ==============================================================================

echo "======================================================================"
echo "STEP 4: Training classifier with RAW BERT embeddings"
echo "======================================================================"

python scripts/tasks/run_task.py \
    --embeddings "$EMBEDDINGS_DIR/wiki_union_raw_bert.pkl" \
    --labels "$LABELS_FILE" \
    --task_name wiki_union_raw_bert \
    --output_dir results/wiki_union_raw_bert \
    --embedding_type cls \
    --combination_method concat \
    --hidden_dim 256 \
    --num_labels 2 \
    --batch_size 32 \
    --max_epochs 50 \
    --learning_rate 2e-5 \
    --random_seed 0 \
    --accelerator gpu \
    --devices 1

echo ""
echo "✅ Raw BERT results saved to: results/wiki_union_raw_bert/"
echo ""

# ==============================================================================
# COMPARE RESULTS
# ==============================================================================

echo "======================================================================"
echo "COMPARISON SUMMARY"
echo "======================================================================"
echo ""
echo "Pretrained TabSketchFM results:"
cat results/wiki_union_pretrained/results.json
echo ""
echo "----------------------------------------------------------------------"
echo ""
echo "Raw BERT results:"
cat results/wiki_union_raw_bert/results.json
echo ""
echo "======================================================================"
echo "Complete! Compare the 'test_results' section above."
echo "======================================================================"
