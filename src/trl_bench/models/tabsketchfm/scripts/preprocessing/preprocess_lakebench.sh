#!/bin/bash
# ==============================================================================
# Preprocess LakeBench Data for Finetuning
# ==============================================================================
# This script preprocesses LakeBench benchmark datasets for finetuning.
# LakeBench datasets do NOT have metadata files - only CSV tables with labels.
#
# Prerequisites:
#   1. Download LakeBench from Zenodo: https://doi.org/10.5281/zenodo.8014642
#   2. Extract to create directories like:
#      - spider_join/spider-join/tables/  (CSV files)
#      - spider_join/spider-join/labels.json  (ground truth labels)
#
# Available LakeBench Tasks:
#   - spider_join     : Spider-OpenData join task (binary classification)
#   - tus_santos      : TUS-SANTOS union task
#   - wiki_union      : Wiki Union task
#   - wiki_jaccard    : Wiki Jaccard join task (via wiki-join-search)
#   - wiki_containment: Wiki Containment join task (requires label generation)
#   - ecb_union       : ECB Union task
#   - ecb_join        : ECB Join task (56-class classification)
#   - ckan_subset     : CKAN Subset task
#
# Note: wiki_containment requires running generate_containment_labels.py first
#       to create labels.json from wiki-join-search ground truth.
#
# Usage:
#   bash scripts/preprocessing/preprocess_lakebench.sh [options]
#
# Options:
#   --task TASK       LakeBench task name (default: spider_join)
#   --input_dir DIR   Input directory with tables (auto-detected if not set)
#   --output_dir DIR  Output directory (default: {task}_processed)
#   --help            Show this help message
#
# Examples:
#   # Preprocess Spider-Join (default)
#   bash scripts/preprocessing/preprocess_lakebench.sh
#
#   # Preprocess TUS-SANTOS
#   bash scripts/preprocessing/preprocess_lakebench.sh --task tus_santos
#
#   # Custom directories
#   bash scripts/preprocessing/preprocess_lakebench.sh \
#       --input_dir /path/to/tables \
#       --output_dir /path/to/output
#
# Output:
#   - Processed JSON.bz2 files in output_dir, one per table
#   - Each file contains MinHash sketches, quantiles, and column info
#
# Next Steps:
#   1. Run finetuning: python finetune.py --data_dir {task}_processed ...
#   2. Or use decoupled pipeline: bash scripts/decoupled_pipeline/run_decoupled_pipeline.sh
#
# See PIPELINE.md for complete workflow documentation.
# ==============================================================================

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
TASK="spider_join"
INPUT_DIR=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,52p' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect input directory based on task if not specified
if [[ -z "$INPUT_DIR" ]]; then
    # Try common LakeBench directory structures
    case $TASK in
        spider_join)
            if [[ -d "spider_join/spider-join/tables" ]]; then
                INPUT_DIR="spider_join/spider-join/tables"
            elif [[ -d "spider-join/tables" ]]; then
                INPUT_DIR="spider-join/tables"
            fi
            ;;
        tus_santos)
            if [[ -d "tus_santos/tus-santos/tables" ]]; then
                INPUT_DIR="tus_santos/tus-santos/tables"
            elif [[ -d "tus-santos/tables" ]]; then
                INPUT_DIR="tus-santos/tables"
            fi
            ;;
        wiki_union)
            if [[ -d "wiki_union/wiki-union/tables" ]]; then
                INPUT_DIR="wiki_union/wiki-union/tables"
            elif [[ -d "wiki-union/tables" ]]; then
                INPUT_DIR="wiki-union/tables"
            elif [[ -d "wiki_union/tables" ]]; then
                INPUT_DIR="wiki_union/tables"
            fi
            ;;
        wiki_containment)
            if [[ -d "wiki_containment/tables" ]]; then
                INPUT_DIR="wiki_containment/tables"
            elif [[ -d "wiki-containment/tables" ]]; then
                INPUT_DIR="wiki-containment/tables"
            fi
            ;;
        *)
            # Generic pattern
            if [[ -d "${TASK}/${TASK}/tables" ]]; then
                INPUT_DIR="${TASK}/${TASK}/tables"
            elif [[ -d "${TASK}/tables" ]]; then
                INPUT_DIR="${TASK}/tables"
            fi
            ;;
    esac
fi

# Set default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${TASK}_processed"
fi

echo "======================================================================"
echo "LAKEBENCH PREPROCESSING (for Finetuning)"
echo "======================================================================"
echo "Project root:    $PROJECT_ROOT"
echo "Task:            $TASK"
echo "Input directory: $INPUT_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Validate input directory
if [[ -z "$INPUT_DIR" ]] || [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo ""
    echo "You need to download LakeBench first:"
    echo "  1. Go to: https://doi.org/10.5281/zenodo.8014642"
    echo "  2. Download the dataset for task: $TASK"
    echo "  3. Extract to create the tables directory"
    echo ""
    echo "Expected directory structure:"
    echo "  ${TASK}/${TASK}/tables/*.csv"
    echo "  ${TASK}/${TASK}/labels.json"
    echo ""
    echo "Or specify --input_dir explicitly:"
    echo "  bash $0 --task $TASK --input_dir /path/to/tables"
    echo ""
    echo "See PIPELINE.md for details."
    exit 1
fi

# Count input files
NUM_CSV=$(find "$INPUT_DIR" -name "*.csv" -o -name "*.CSV" 2>/dev/null | wc -l)
echo "Found $NUM_CSV CSV files"
echo ""

if [[ $NUM_CSV -eq 0 ]]; then
    echo "Error: No CSV files found in $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set PYTHONPATH to include tabsketchfm module
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/tabsketchfm:$PYTHONPATH"

# Run preprocessing
echo "Starting preprocessing..."
echo "Progress will be printed every 50 files."
echo ""

python "$PROJECT_ROOT/data_processing/batch_fastdata.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

# Count output files
NUM_OUTPUT=$(find "$OUTPUT_DIR" -name "*.json.bz2" 2>/dev/null | wc -l)

# Try to find labels file
LABELS_FILE=""
if [[ -f "${TASK}/${TASK}/labels.json" ]]; then
    LABELS_FILE="${TASK}/${TASK}/labels.json"
elif [[ -f "${TASK}/labels.json" ]]; then
    LABELS_FILE="${TASK}/labels.json"
fi

echo ""
echo "======================================================================"
echo "PREPROCESSING COMPLETE"
echo "======================================================================"
echo "Output files: $NUM_OUTPUT JSON.bz2 files in $OUTPUT_DIR"
if [[ -n "$LABELS_FILE" ]]; then
    echo "Labels file:  $LABELS_FILE"
fi
echo ""
echo "Next steps:"
echo ""
echo "  Option A - Standard Finetuning:"
echo "    python finetune.py \\"
echo "        --model_name_or_path bert-base-uncased \\"
echo "        --checkpoint <pretrained_checkpoint.ckpt> \\"
echo "        --data_dir $OUTPUT_DIR \\"
echo "        --dataset $LABELS_FILE \\"
echo "        --task_type classification \\"
echo "        --num_labels 2 \\"
echo "        --max_epochs 50"
echo ""
echo "  Option B - Decoupled Pipeline (faster experimentation):"
echo "    bash scripts/decoupled_pipeline/run_decoupled_pipeline.sh \\"
echo "        --data_dir $OUTPUT_DIR \\"
echo "        --dataset $LABELS_FILE \\"
echo "        --output_prefix $TASK"
echo ""
echo "See PIPELINE.md for complete workflow."
echo "======================================================================"
