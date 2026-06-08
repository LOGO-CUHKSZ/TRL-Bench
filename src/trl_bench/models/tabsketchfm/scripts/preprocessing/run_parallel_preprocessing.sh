#!/bin/bash
# Parallel preprocessing script for CKAN Subset dataset
#
# This script runs the parallelized batch preprocessing for LakeBench datasets.
# It provides a convenient wrapper around batch_fastdata_parallel.py with
# proper environment setup and sensible defaults.
#
# Usage:
#   bash scripts/preprocessing/run_parallel_preprocessing.sh [input_dir] [output_dir] [num_workers]
#
# Examples:
#   # Use defaults (ckan_subset/tables -> ckan_subset_processed, 16 workers)
#   bash scripts/preprocessing/run_parallel_preprocessing.sh
#
#   # Custom directories
#   bash scripts/preprocessing/run_parallel_preprocessing.sh \
#       spider_join/tables spider_join_processed
#
#   # Custom worker count
#   bash scripts/preprocessing/run_parallel_preprocessing.sh \
#       ckan_subset/tables ckan_subset_processed 32

set -e  # Exit on error

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================="
echo "Parallel Batch Preprocessing"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Parse arguments with defaults
INPUT_DIR="${1:-ckan_subset/tables}"
OUTPUT_DIR="${2:-ckan_subset_processed}"
NUM_WORKERS="${3:-16}"

echo "Configuration:"
echo "  Input directory: ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Number of workers: ${NUM_WORKERS}"
echo "=========================================="
echo ""

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Input directory not found: ${INPUT_DIR}"
    echo ""
    echo "Available directories:"
    ls -d */tables 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate environment
echo "Activating environment..."
if [ -f "load_env" ]; then
    source load_env
else
    echo "WARNING: load_env not found, using current environment"
fi

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Count input files
echo ""
echo "Counting input files..."
NUM_FILES=$(find "${INPUT_DIR}" -name "*.csv" -o -name "*.csv.gz" -o -name "*.csv.bz2" | wc -l)
echo "Found ${NUM_FILES} CSV files to process"
echo ""

# Estimate time
if [ ${NUM_FILES} -gt 0 ]; then
    # Rough estimate: 0.3 seconds per file sequentially, divided by workers
    # with 70% efficiency
    ESTIMATED_SECONDS=$(echo "${NUM_FILES} * 0.3 / ${NUM_WORKERS} / 0.7" | bc)
    ESTIMATED_MINUTES=$(echo "${ESTIMATED_SECONDS} / 60" | bc)
    echo "Estimated time: ~${ESTIMATED_MINUTES} minutes (very rough estimate)"
    echo ""
fi

# Run parallel preprocessing
echo "=========================================="
echo "Starting parallel preprocessing..."
echo "=========================================="
echo ""

python tabsketchfm/batch_fastdata_parallel.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${NUM_WORKERS}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Preprocessing completed successfully!"
    echo ""

    # Count output files
    NUM_OUTPUT=$(find "${OUTPUT_DIR}" -name "*.json.bz2" | wc -l)
    echo "Output statistics:"
    echo "  Output directory: ${OUTPUT_DIR}"
    echo "  Files created: ${NUM_OUTPUT}"
    echo "  Expected files: ~$((NUM_FILES * 3)) (3 augmentations per input)"

    # Check for error log
    if [ -f "${OUTPUT_DIR}/preprocessing_errors.txt" ]; then
        NUM_ERRORS=$(grep -c "^File:" "${OUTPUT_DIR}/preprocessing_errors.txt" || echo "0")
        echo "  Errors logged: ${NUM_ERRORS}"
        echo "  Error log: ${OUTPUT_DIR}/preprocessing_errors.txt"
    fi
else
    echo "Preprocessing failed with exit code ${EXIT_CODE}"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Check that load_env activates the correct environment"
    echo "  2. Ensure PYTHONPATH is set correctly"
    echo "  3. Review error messages above"
    echo "  4. Try running with fewer workers: --num_workers 4"
fi
echo "=========================================="

exit ${EXIT_CODE}
