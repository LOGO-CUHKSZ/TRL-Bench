#!/bin/bash
# NQT-Retrieval style training pipeline for table retrieval.
#
# This script follows the NQT-Retrieval approach:
# 0. Create hybrid table embeddings (encoder + BERT, default for non-BERT)
# 1. Curate training data by mining hard negatives from full corpus
# 2. Train with curated data (hard negatives from full corpus)
# 3. Evaluate on dev set
#
# IMPORTANT: Query embeddings are always BERT. For table embeddings, we support
# multiple encoders. In hybrid mode (default for non-BERT), table embeddings
# are combined with BERT table embeddings via concatenation.
#
# Usage:
#   bash train_nqt.sh [encoder_type] [experiment_name] [--no-hybrid]
#
# Arguments:
#   encoder_type: bert, tapas, tabert, or tabsketchfm (default: bert)
#   experiment_name: name for experiment (default: nqt_style)
#   --no-hybrid: skip hybrid mode (use pure table embeddings)
#
# Examples:
#   bash train_nqt.sh tapas tapas_v1          # Hybrid: TAPAS + BERT (recommended)
#   bash train_nqt.sh bert bert_v1            # Pure BERT (no hybrid needed)
#   bash train_nqt.sh tapas tapas_pure --no-hybrid  # Pure TAPAS (not recommended)

set -euo pipefail

# Get script directory (downstream_tasks/table_retrieval)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get TRL project root
TRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Arguments
ENCODER_TYPE="${1:-bert}"
EXPERIMENT_NAME="${2:-nqt_style}"
NO_HYBRID=false
for arg in "$@"; do
    if [[ "$arg" == "--no-hybrid" ]]; then
        NO_HYBRID=true
    fi
done

# Query embeddings are always BERT
QUERY_EMBEDDING_DIR="${TRL_ROOT}/embeddings/table_retrieval/bert"
TABLE_EMBEDDING_DIR="${TRL_ROOT}/embeddings/column/${ENCODER_TYPE}"
BERT_TABLE_EMBEDDING_DIR="${TRL_ROOT}/embeddings/column/bert"
HYBRID_TABLE_EMBEDDING_DIR="${TRL_ROOT}/embeddings/column/${ENCODER_TYPE}_bert_hybrid"
DATASET_DIR="${TRL_ROOT}/datasets/nq_tables"
CHECKPOINT_DIR="${TRL_ROOT}/assets/checkpoints/table_retrieval/${ENCODER_TYPE}_${EXPERIMENT_NAME}"
CURATED_DATA_DIR="${TRL_ROOT}/embeddings/table_retrieval/hard_negatives"
TABLE_ID_MAPPING="${DATASET_DIR}/csv/table_id_to_csv.json"

# Determine table embeddings: hybrid by default for non-BERT encoders
if [[ "${ENCODER_TYPE}" == "bert" ]] || [[ "${NO_HYBRID}" == true ]]; then
    TABLE_EMBEDDINGS="${TABLE_EMBEDDING_DIR}/nq_tables.pkl"
    USE_HYBRID=false
else
    TABLE_EMBEDDINGS="${HYBRID_TABLE_EMBEDDING_DIR}/nq_tables.pkl"
    USE_HYBRID=true
fi

# Query embeddings (always BERT)
TRAIN_QUERY_EMBEDDINGS="${QUERY_EMBEDDING_DIR}/queries_train.pkl"
DEV_QUERY_EMBEDDINGS="${QUERY_EMBEDDING_DIR}/queries_dev.pkl"

# Dataset files
TRAIN_QUESTIONS="${DATASET_DIR}/json/train.json"
DEV_QUESTIONS="${DATASET_DIR}/json/dev.json"
TABLES_JSON="${DATASET_DIR}/json/tables.json"

# Curated data output
CURATED_TRAIN="${CURATED_DATA_DIR}/train_${ENCODER_TYPE}_curated.json"

# Training parameters (matching NQT-Retrieval)
BATCH_SIZE=8
EPOCHS=30
LEARNING_RATE=2e-5
NUM_HARD_NEGATIVES=1
NUM_OTHER_NEGATIVES=0
TOP_K_RETRIEVAL=100

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${CURATED_DATA_DIR}"

echo "=============================================="
echo "NQT-Retrieval Style Training Pipeline"
echo "=============================================="
echo "Encoder: ${ENCODER_TYPE}"
echo "Hybrid mode: ${USE_HYBRID}"
echo "Query embeddings: BERT (always)"
echo "Table embeddings: ${TABLE_EMBEDDINGS}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${CHECKPOINT_DIR}"
echo "=============================================="

# Check if base embeddings exist
if [[ ! -f "${TABLE_EMBEDDING_DIR}/nq_tables.pkl" ]]; then
    echo "ERROR: Table embeddings not found at ${TABLE_EMBEDDING_DIR}/nq_tables.pkl"
    exit 1
fi

if [[ ! -f "${TRAIN_QUERY_EMBEDDINGS}" ]]; then
    echo "ERROR: Query embeddings not found at ${TRAIN_QUERY_EMBEDDINGS}"
    exit 1
fi

# Step 0: Create hybrid table embeddings if needed
if [[ "${USE_HYBRID}" == true ]]; then
    echo ""
    echo "[Step 0] Creating hybrid table embeddings (${ENCODER_TYPE} + BERT)..."
    if [[ -f "${TABLE_EMBEDDINGS}" ]]; then
        echo "  Hybrid embeddings already exist at ${TABLE_EMBEDDINGS}, skipping."
        echo "  Delete to re-create."
    else
        python -m downstream_tasks.table_retrieval.create_hybrid_embeddings \
            --base_variant column_mean \
            --base_embeddings "${TABLE_EMBEDDING_DIR}/nq_tables.pkl" \
            --bert_embeddings "${BERT_TABLE_EMBEDDING_DIR}/nq_tables.pkl" \
            --output_path "${TABLE_EMBEDDINGS}" \
            --combination_method concat \
            --table_id_mapping "${TABLE_ID_MAPPING}"
    fi
fi

# Step 1: Curate training data (mine hard negatives from full corpus)
echo ""
echo "[Step 1/3] Curating training data with hard negatives..."
echo "  Mining top-${TOP_K_RETRIEVAL} tables per query from full corpus"
echo "  Output: ${CURATED_TRAIN}"

if [[ -f "${CURATED_TRAIN}" ]]; then
    echo "  Curated data already exists, skipping curation."
    echo "  Delete ${CURATED_TRAIN} to re-run curation."
else
    python -m downstream_tasks.table_retrieval.curate_training_data \
        --table_embeddings "${TABLE_EMBEDDINGS}" \
        --table_id_mapping "${TABLE_ID_MAPPING}" \
        --query_embeddings "${TRAIN_QUERY_EMBEDDINGS}" \
        --questions "${TRAIN_QUESTIONS}" \
        --tables_json "${TABLES_JSON}" \
        --output_path "${CURATED_TRAIN}" \
        --top_k ${TOP_K_RETRIEVAL} \
        --num_hard_negatives ${NUM_HARD_NEGATIVES} \
        --num_other_negatives ${NUM_OTHER_NEGATIVES}
fi

# Step 2: Train with curated data
echo ""
echo "[Step 2/3] Training with curated data..."
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Hard negatives per sample: ${NUM_HARD_NEGATIVES}"

python -m downstream_tasks.table_retrieval.train \
    --table_embeddings "${TABLE_EMBEDDINGS}" \
    --table_id_mapping "${TABLE_ID_MAPPING}" \
    --train_query_embeddings "${TRAIN_QUERY_EMBEDDINGS}" \
    --train_questions "${TRAIN_QUESTIONS}" \
    --train_curated "${CURATED_TRAIN}" \
    --dev_query_embeddings "${DEV_QUERY_EMBEDDINGS}" \
    --dev_questions "${DEV_QUESTIONS}" \
    --output_dir "${CHECKPOINT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --num_hard_negatives ${NUM_HARD_NEGATIVES} \
    --num_other_negatives ${NUM_OTHER_NEGATIVES}

# Step 3: Final evaluation
echo ""
echo "[Step 3/3] Evaluating on dev set..."

python -m downstream_tasks.table_retrieval.evaluate \
    --table_embeddings "${TABLE_EMBEDDINGS}" \
    --table_id_mapping "${TABLE_ID_MAPPING}" \
    --query_embeddings "${DEV_QUERY_EMBEDDINGS}" \
    --questions_path "${DEV_QUESTIONS}" \
    --projection_head "${CHECKPOINT_DIR}/best_model.pt"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Best model saved to: ${CHECKPOINT_DIR}/best_model.pt"
echo ""
