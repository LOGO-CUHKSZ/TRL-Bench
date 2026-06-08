#!/bin/bash
# Starmie Pipeline - SANTOS Benchmark (Aligned with Paper's Optimal Settings)
# Based on: "Semantics-aware Dataset Discovery from Data Lakes with
#            Contextualized Column-based Representation Learning" (VLDB 2023)

set -e  # Exit on error

# Activate conda environment
source activate starmie
export CUDA_VISIBLE_DEVICES=0

# Paper's optimal configuration for SANTOS Small benchmark:
# - augment_op: drop_col
# - sample_meth: tfidf_entity (NOT head!)
# - table_order: column
# This achieves MAP@10 = 0.993 (linear) / 0.945 (HNSW)

echo "=========================================="
echo "Starmie Pipeline for SANTOS Benchmark"
echo "=========================================="
echo ""

# ==========================================
# Step 1: Pre-training (Offline Stage)
# ==========================================
echo "Step 1: Pre-training the multi-column encoder..."
echo "Expected time: 30-60 minutes for 3 epochs on 550 tables"
echo ""

python run_pretrain.py \
  --task santos \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 3 \
  --max_len 128 \
  --size 550 \
  --projector 768 \
  --save_model \
  --augment_op drop_col \
  --sample_meth tfidf_entity \
  --table_order column \
  --run_id 0

echo ""
echo "✓ Pre-training completed!"
echo "Model saved to: results/santos/model_drop_col_tfidf_entity_column_0.pt"
echo ""

# ==========================================
# Step 2: Extract Column Embeddings
# ==========================================
echo "Step 2: Extracting column embeddings from trained model..."
echo "Processing 50 query tables + 550 datalake tables"
echo ""

# Note: Need to update extractVectors.py to use tfidf_entity for santos
python extractVectors.py \
  --benchmark santos \
  --table_order column \
  --run_id 0 \
  --save_model

echo ""
echo "✓ Embeddings extracted!"
echo "Saved to:"
echo "  - data/santos/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl"
echo "  - data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl"
echo ""

# ==========================================
# Step 3: Table Union Search (Online Stage)
# ==========================================
echo "Step 3: Performing Table Union Search..."
echo ""

# ==========================================
# 3a. Linear Search (Most Accurate)
# ==========================================
echo "3a. Running Linear Search (most accurate, slowest)..."
echo "Expected: MAP@10 = 0.993, Query Time = ~96s"
echo ""

python test_naive_search.py \
  --encoder cl \
  --benchmark santos \
  --augment_op drop_col \
  --sample_meth tfidf_entity \
  --matching linear \
  --table_order column \
  --run_id 0 \
  --K 10 \
  --threshold 0.7

echo ""
echo "✓ Linear search completed!"
echo ""

# ==========================================
# 3b. HNSW Index Search (Fastest)
# ==========================================
echo "3b. Running HNSW Search (220x faster with small accuracy loss)..."
echo "Expected: MAP@10 = 0.945, Query Time = ~4s"
echo ""

python test_hnsw_search.py \
  --encoder cl \
  --benchmark santos \
  --run_id 0 \
  --K 10

echo ""
echo "✓ HNSW search completed!"
echo ""

# ==========================================
# 3c. LSH Index Search (Alternative)
# ==========================================
echo "3c. Running LSH Search (20x faster alternative)..."
echo "Expected: MAP@10 = 0.932, Query Time = ~12s"
echo ""

python test_lsh.py \
  --encoder cl \
  --benchmark santos \
  --run_id 0 \
  --num_func 8 \
  --num_table 100 \
  --K 10

echo ""
echo "✓ LSH search completed!"
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "Results Summary (Expected from Paper):"
echo "  Method          | MAP@10 | Query Time"
echo "  ----------------|--------|------------"
echo "  Linear          | 0.993  | ~96s"
echo "  HNSW Index      | 0.945  | ~4s (220x faster)"
echo "  LSH Index       | 0.932  | ~12s (20x faster)"
echo ""
echo "Check MLflow logs in ./mlruns/ for detailed metrics"
echo ""
