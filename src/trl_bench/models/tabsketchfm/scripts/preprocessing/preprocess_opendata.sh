#!/bin/bash
# ==============================================================================
# Preprocess OpenData for Pretraining
# ==============================================================================
# This script preprocesses downloaded OpenData tables for pretraining.
# OpenData tables have associated metadata files (.meta) that contain
# table/dataset descriptions used during masked language modeling.
#
# Prerequisites:
#   1. Download OpenData using data_downloader/download_manager.py
#   2. This creates:
#      - opendata/           (CSV files organized by domain)
#      - opendata_metadata/  (corresponding .meta files)
#
# Usage:
#   bash scripts/preprocessing/preprocess_opendata.sh [options]
#
# Options:
#   --input_dir DIR       Input directory with CSV tables (default: opendata)
#   --metadata_dir DIR    Metadata directory with .meta files (default: opendata_metadata)
#   --output_dir DIR      Output directory for processed files (default: opendata_processed)
#   --help                Show this help message
#
# Output:
#   - Processed JSON.bz2 files in output_dir, one per table
#   - Each file contains MinHash sketches, quantiles, and column metadata
#
# Next Steps:
#   1. Create train/val/test splits: python scripts/data_utils/create_data_splits.py
#   2. (Optional) Filter large tables: python scripts/data_utils/filter_large_tables.py
#   3. Run pretraining: python pretrain.py --dataset data_splits.json.bz2
#
# See PIPELINE.md for complete workflow documentation.
# ==============================================================================

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
INPUT_DIR="opendata"
METADATA_DIR="opendata_metadata"
OUTPUT_DIR="opendata_processed"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --metadata_dir)
            METADATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,35p' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "OPENDATA PREPROCESSING (for Pretraining)"
echo "======================================================================"
echo "Project root:    $PROJECT_ROOT"
echo "Input directory: $INPUT_DIR"
echo "Metadata dir:    $METADATA_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Validate directories
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo ""
    echo "You need to download OpenData first. Run:"
    echo "  cd data_downloader"
    echo "  python download_manager.py"
    echo ""
    echo "See PIPELINE.md for details."
    exit 1
fi

if [[ ! -d "$METADATA_DIR" ]]; then
    echo "Error: Metadata directory not found: $METADATA_DIR"
    echo ""
    echo "The data_downloader should create both opendata/ and opendata_metadata/"
    echo "Make sure you ran the download correctly."
    exit 1
fi

# Count input files
NUM_CSV=$(find "$INPUT_DIR" -name "*.csv" -o -name "*.CSV" 2>/dev/null | wc -l)
NUM_META=$(find "$METADATA_DIR" -name "*.meta" 2>/dev/null | wc -l)
echo "Found $NUM_CSV CSV files and $NUM_META metadata files"
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
echo "This may take several hours to days depending on data size."
echo "Progress will be printed every 100 files."
echo ""

python "$PROJECT_ROOT/data_processing/batch_fastdata_opendata.py" \
    --input_dir "$INPUT_DIR" \
    --metadata_dir "$METADATA_DIR" \
    --output_dir "$OUTPUT_DIR"

# Count output files
NUM_OUTPUT=$(find "$OUTPUT_DIR" -name "*.json.bz2" 2>/dev/null | wc -l)

echo ""
echo "======================================================================"
echo "PREPROCESSING COMPLETE"
echo "======================================================================"
echo "Output files: $NUM_OUTPUT JSON.bz2 files in $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Create data splits:"
echo "     python scripts/data_utils/create_data_splits.py \\"
echo "         --opendata_dir $INPUT_DIR \\"
echo "         --metadata_dir $METADATA_DIR \\"
echo "         --processed_dir $OUTPUT_DIR \\"
echo "         --output data_splits.json.bz2"
echo ""
echo "  2. (Optional) Filter large tables:"
echo "     python scripts/data_utils/filter_large_tables.py \\"
echo "         --input data_splits.json.bz2 \\"
echo "         --output data_splits_filtered_256.json.bz2"
echo ""
echo "  3. Run pretraining:"
echo "     python pretrain.py --dataset data_splits.json.bz2 ..."
echo ""
echo "See PIPELINE.md for complete workflow."
echo "======================================================================"
