#!/bin/bash
# ==============================================================================
# Wiki-Containment Decoupled Pipeline
# ==============================================================================
# This script demonstrates the fully decoupled approach for wiki-containment:
#   1. Generate labels from containment ground truth (if needed)
#   2. Preprocess tables (if needed)
#   3. Generate embeddings from pretrained model (once, reusable)
#   4. Train regressor on paired embeddings (fast, can iterate)
#
# This is similar to finetuning but WITHOUT updating the pretrained model.
# Instead, embeddings are frozen and a lightweight model is trained on top.
#
# Usage:
#   bash scripts/tasks/run_wiki_containment.sh [OPTIONS]
#
# Options:
#   --skip_generation        Skip embedding extraction if already done
#   --skip_label_generation  Skip label generation if already done
#   --skip_preprocessing     Skip preprocessing if already done
#   --model PATH             Use specific pretrained model
#   --embedding_type TYPE    Type of embedding (cls, mean_pool, last_hidden)
#   --combination_method M   How to combine embeddings (concat, diff, hadamard)
#   --results_dir DIR        Output directory for results
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
MODEL_PATH="logs/tabsketchfm-pretrain-filtered/tabsketchfm-pretrain/bn6oq8sa/checkpoints/epoch=11-step=9084.ckpt"

# Table directories
RAW_TABLES_DIR=""
LABELS_DIR="wiki_containment"

# Processed data
PROCESSED_DIR="wiki_containment_processed"

# Ground truth
GROUND_TRUTH_FILE=""

# Labels
LABELS_FILE="wiki_containment/labels.json"

# Embeddings
EMBEDDINGS_FILE="embeddings/wiki_containment_embeddings.pkl"

# Output
RESULTS_DIR="results/wiki_containment_decoupled"

# Task parameters (REGRESSION for containment scores)
TASK_NAME="wiki_containment"
TASK_TYPE="regression"
NUM_LABELS=1
THRESHOLD=0.05
NEGATIVE_RATIO=1.0

# Training parameters
HIDDEN_DIM=256
MAX_EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=2e-5
RANDOM_SEED=0

# Embedding parameters
# Use 'column' for column-level embeddings (recommended for containment/join tasks)
# This uses the specific column embeddings from join_col_table1/join_col_table2 in labels
EMBEDDING_TYPE="column"
COMBINATION_METHOD="concat"
EXTRACTION_BATCH_SIZE=256

# Flags
SKIP_GENERATION=false
SKIP_LABEL_GENERATION=false
SKIP_PREPROCESSING=false

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_generation)
            SKIP_GENERATION=true
            shift
            ;;
        --skip_label_generation)
            SKIP_LABEL_GENERATION=true
            shift
            ;;
        --skip_preprocessing)
            SKIP_PREPROCESSING=true
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
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Wiki-Containment decoupled pipeline (extract embeddings, train regressor)"
            echo ""
            echo "Options:"
            echo "  --skip_generation        Skip embedding extraction if already done"
            echo "  --skip_label_generation  Skip label generation if already done"
            echo "  --skip_preprocessing     Skip preprocessing if already done"
            echo "  --model PATH             Use specific pretrained model"
            echo "  --embedding_type TYPE    Type of embedding: cls, table, column_mean (table-level),"
            echo "                           column (column-level, uses join_col fields in labels, default)"
            echo "  --combination_method M   Combination: concat, diff, hadamard (default: concat)"
            echo "  --results_dir DIR        Output directory for results"
            echo "  --task_type TYPE         regression or classification (default: regression)"
            echo "  --threshold FLOAT        Threshold for classification (default: 0.05)"
            echo "  -h, --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Adjust NUM_LABELS based on task type
if [ "$TASK_TYPE" = "classification" ]; then
    NUM_LABELS=2
fi

echo "======================================================================"
echo "WIKI-CONTAINMENT DECOUPLED PIPELINE"
echo "======================================================================"
echo "Model:              $MODEL_PATH"
echo "Task type:          $TASK_TYPE"
echo "Num labels:         $NUM_LABELS"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Results:            $RESULTS_DIR"
echo "======================================================================"
echo ""

# ==============================================================================
# STEP 0: VALIDATE AND SELECT DATA SOURCE
# ==============================================================================

echo "📋 STEP 0: Validating data sources..."
echo "----------------------------------------------------------------------"

# Prefer wiki-join-search (full dataset), fallback to wiki_containment (sample)
if [[ -d "wiki-join-search/tables" ]]; then
    RAW_TABLES_DIR="wiki-join-search/tables"
    echo "✅ Using wiki-join-search tables (46K+ tables with containment ground truth)"
elif [[ -d "wiki_containment/tables" ]]; then
    RAW_TABLES_DIR="wiki_containment/tables"
    echo "⚠️  Warning: Using wiki_containment tables (only 47 sample tables)"
    echo "   For better results, download wiki-join-search.tar.bz2"
else
    echo "❌ Error: No tables directory found!"
    echo ""
    echo "Please download and extract wiki-join-search.tar.bz2:"
    echo "  tar -xjf wiki-join-search.tar.bz2"
    echo ""
    echo "Or wiki-containment.tar.bz2 (sample only):"
    echo "  tar -xjf wiki-containment.tar.bz2"
    exit 1
fi

# Check for containment ground truth
if [[ -f "wiki-join-search/labels/join_search_containment_min_gt.jsonl" ]]; then
    GROUND_TRUTH_FILE="wiki-join-search/labels/join_search_containment_min_gt.jsonl"
    echo "✅ Found containment ground truth: $GROUND_TRUTH_FILE"
else
    echo "❌ Error: Could not find containment ground truth file!"
    echo "   Expected: wiki-join-search/labels/join_search_containment_min_gt.jsonl"
    echo ""
    echo "Please download and extract wiki-join-search.tar.bz2 first."
    exit 1
fi

echo ""

# ==============================================================================
# STEP 1: GENERATE LABELS
# ==============================================================================

if [ "$SKIP_LABEL_GENERATION" = false ]; then
    echo "📋 STEP 1: Generating labels from containment ground truth..."
    echo "----------------------------------------------------------------------"

    mkdir -p "$LABELS_DIR"

    python scripts/data_utils/generate_containment_labels.py \
        --input "$GROUND_TRUTH_FILE" \
        --tables_dir "$RAW_TABLES_DIR" \
        --output "$LABELS_FILE" \
        --task_type "$TASK_TYPE" \
        --threshold "$THRESHOLD" \
        --negative_ratio "$NEGATIVE_RATIO" \
        --seed "$RANDOM_SEED"

    if [ $? -ne 0 ]; then
        echo "❌ Label generation failed!"
        exit 1
    fi

    echo "✅ Labels generated: $LABELS_FILE"
    echo ""
else
    echo "⏭️  STEP 1: Skipping label generation (using existing: $LABELS_FILE)"
    echo ""
fi

# Validate labels file
if [[ ! -f "$LABELS_FILE" ]]; then
    echo "❌ Error: Labels file not found: $LABELS_FILE"
    echo "   Run without --skip_label_generation to generate it."
    exit 1
fi

# ==============================================================================
# STEP 2: PREPROCESS TABLES
# ==============================================================================

if [ "$SKIP_PREPROCESSING" = false ]; then
    if [ ! -d "$PROCESSED_DIR" ]; then
        echo "📋 STEP 2: Preprocessing tables..."
        echo "----------------------------------------------------------------------"

        mkdir -p "$PROCESSED_DIR"

        python tabsketchfm/batch_fastdata.py \
            --input_dir "$RAW_TABLES_DIR" \
            --output_dir "$PROCESSED_DIR"

        if [ $? -ne 0 ]; then
            echo "❌ Preprocessing failed!"
            exit 1
        fi

        NUM_FILES=$(find "$PROCESSED_DIR" -name "*.json.bz2" | wc -l)
        echo "✅ Preprocessed $NUM_FILES tables"
        echo ""
    else
        echo "⏭️  STEP 2: Skipping preprocessing (already exists: $PROCESSED_DIR)"
        echo ""
    fi
else
    echo "⏭️  STEP 2: Skipping preprocessing (--skip_preprocessing flag set)"
    echo ""
fi

# ==============================================================================
# STEP 3: EXTRACT EMBEDDINGS
# ==============================================================================

if [ "$SKIP_GENERATION" = true ]; then
    echo "======================================================================"
    echo "SKIPPING EMBEDDING EXTRACTION (--skip_generation flag set)"
    echo "======================================================================"
    echo "Using existing embeddings: $EMBEDDINGS_FILE"
    echo ""
else
    echo "======================================================================"
    echo "PHASE 1: EXTRACTING EMBEDDINGS (FROZEN, REUSABLE)"
    echo "======================================================================"
    echo "Model:           $MODEL_PATH"
    echo "Data:            $PROCESSED_DIR"
    echo "Output:          $EMBEDDINGS_FILE"
    echo "Batch size:      $EXTRACTION_BATCH_SIZE"
    echo ""
    echo "NOTE: This extracts embeddings from the pretrained model WITHOUT"
    echo "      finetuning. Embeddings are frozen and reusable across"
    echo "      different downstream experiments."
    echo "======================================================================"
    echo ""

    mkdir -p embeddings

    # Extract embeddings using unified extraction script
    bash scripts/tasks/generate_embeddings.sh \
        --model "$MODEL_PATH" \
        --data_dir "$PROCESSED_DIR" \
        --output "$EMBEDDINGS_FILE" \
        --batch_size "$EXTRACTION_BATCH_SIZE"

    if [ $? -ne 0 ]; then
        echo "❌ Embedding extraction failed!"
        exit 1
    fi

    echo ""
    echo "✅ Phase 1 complete: Embeddings saved to $EMBEDDINGS_FILE"
    echo ""
fi

# ==============================================================================
# PHASE 2: TRAIN REGRESSOR/CLASSIFIER
# ==============================================================================

echo "======================================================================"
echo "PHASE 2: TRAINING LIGHTWEIGHT MODEL ON FROZEN EMBEDDINGS"
echo "======================================================================"
echo "Embeddings:         $EMBEDDINGS_FILE"
echo "Labels:             $LABELS_FILE"
echo "Task:               $TASK_NAME ($TASK_TYPE)"
echo "Embedding type:     $EMBEDDING_TYPE"
echo "Combination:        $COMBINATION_METHOD"
echo "Architecture:       input_dim → $HIDDEN_DIM → $NUM_LABELS"
echo "Results:            $RESULTS_DIR"
echo ""
echo "NOTE: This trains a simple neural network on frozen embeddings."
echo "      Much faster than full finetuning and allows rapid iteration."
echo "======================================================================"
echo ""

# Train regressor on paired embeddings
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

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "======================================================================"
echo "WIKI-CONTAINMENT DECOUPLED PIPELINE COMPLETE!"
echo "======================================================================"
echo "Embeddings:  $EMBEDDINGS_FILE"
echo "Results:     $RESULTS_DIR"
echo "Summary:     $RESULTS_DIR/results.json"
echo ""
echo "📊 To view results:"
echo "  cat $RESULTS_DIR/results.json"
echo ""
echo "🔄 To run with different hyperparameters (embeddings already extracted):"
echo "  bash scripts/tasks/run_wiki_containment.sh --skip_generation --skip_label_generation --skip_preprocessing"
echo ""
echo "🧪 To try different embedding combinations:"
echo "  bash scripts/tasks/run_wiki_containment.sh --skip_generation --skip_label_generation --skip_preprocessing --combination_method diff"
echo ""
echo "📈 To try classification instead of regression:"
echo "  bash scripts/tasks/run_wiki_containment.sh --skip_generation --skip_label_generation --skip_preprocessing --task_type classification --threshold 0.05"
echo "======================================================================"
