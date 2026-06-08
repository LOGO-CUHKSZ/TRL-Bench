#!/bin/bash
# ==============================================================================
# Multi-Node Wiki-Union Finetuning Launch Script
# ==============================================================================
# This script launches distributed finetuning for Wiki-Union across multiple
# nodes using finetune_multinode.py, which orchestrates torchrun on each node.
#
# Prerequisites:
#   1. SSH access to all nodes without password (ssh-key based)
#   2. Same environment setup on all nodes (conda/venv, CUDA, etc.)
#   3. Shared filesystem accessible from all nodes (for data and checkpoints)
#   4. Preprocessed data: wiki_union_processed/
#   5. Labels file: labels.json
#
# Usage:
#   1. Edit the configuration below to match your cluster
#   2. Run: bash scripts/finetuning/multinode_wiki_union.sh
#
# Configuration Parameters:
#   NODES          - Space-separated list of hostnames
#   MASTER_ADDR    - Hostname of the master node (usually first node)
#   GPUS_PER_NODE  - Number of GPUs per node
#   MASTER_PORT    - Port for distributed communication
#   PRE_COMMANDS   - Commands to setup environment on each node
# ==============================================================================

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ==============================================================================
# CLUSTER CONFIGURATION - EDIT THESE FOR YOUR SETUP
# ==============================================================================

# List of node hostnames (space-separated)
NODES="kn087 kn092 kn097 kn098"

# Master node address (usually the first node)
MASTER_ADDR="kn087"

# GPUs per node
GPUS_PER_NODE=4

# Port for distributed communication (change if port conflicts)
MASTER_PORT=12952

# ==============================================================================
# ENVIRONMENT SETUP - EDIT THESE FOR YOUR CLUSTER
# ==============================================================================

# Commands to run on each node before training
# These are executed in order via --pre-cmd flags
PRE_CMD_1="module load python/3.10"
PRE_CMD_2="module load cuda/12.2"
PRE_CMD_3="source venv/bin/activate"

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

# Model checkpoint
MODEL_CHECKPOINT="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"

# Dataset
DATA_DIR="wiki_union_processed"
LABELS_FILE="wiki_union/labels.json"

# Task configuration
TASK_TYPE="classification"
NUM_LABELS=2

# Training hyperparameters
MAX_EPOCHS=50
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
RANDOM_SEED=0

# Learning rate scaling
# Base LR for single-node (4 GPUs) training with batch_size=64
BASE_LEARNING_RATE=2e-5
# Reference batch size for base LR (4 GPUs × 64 = 256)
REFERENCE_BATCH_SIZE=256

# Output directory
OUTPUT_DIR="./wiki_union_finetuned_multinode"

# Resume from checkpoint (leave empty for fresh start)
RESUME_CHECKPOINT=""
# RESUME_CHECKPOINT="wiki_union_finetuned_multinode/checkpoints/epoch=9-step=XXX.ckpt"

# ==============================================================================
# BUILD AND RUN COMMAND
# ==============================================================================

# Count nodes
NUM_NODES=$(echo $NODES | wc -w)

# Calculate scaled learning rate
# Effective batch size = NUM_NODES × GPUS_PER_NODE × TRAIN_BATCH_SIZE
EFFECTIVE_BATCH_SIZE=$((NUM_NODES * GPUS_PER_NODE * TRAIN_BATCH_SIZE))
# Scale factor = EFFECTIVE_BATCH_SIZE / REFERENCE_BATCH_SIZE
SCALE_FACTOR=$(python3 -c "print($EFFECTIVE_BATCH_SIZE / $REFERENCE_BATCH_SIZE)")
# Scaled LR = BASE_LR × SCALE_FACTOR
LEARNING_RATE=$(python3 -c "print($BASE_LEARNING_RATE * $SCALE_FACTOR)")

echo "======================================================================"
echo "MULTI-NODE WIKI-UNION FINETUNING"
echo "======================================================================"
echo "Project root:    $PROJECT_ROOT"
echo "Nodes:           $NODES"
echo "Num nodes:       $NUM_NODES"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "Total GPUs:      $((NUM_NODES * GPUS_PER_NODE))"
echo "Master:          $MASTER_ADDR:$MASTER_PORT"
echo "Model:           $MODEL_CHECKPOINT"
echo "Data directory:  $DATA_DIR"
echo "Labels:          $LABELS_FILE"
echo "Max epochs:      $MAX_EPOCHS"
echo "Batch size:      $TRAIN_BATCH_SIZE (per GPU)"
echo "Effective batch: $EFFECTIVE_BATCH_SIZE"
echo "Learning rate:   $LEARNING_RATE (scaled from base $BASE_LEARNING_RATE by ${SCALE_FACTOR}x)"
echo "Output:          $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Build the command
CMD="python scripts/finetuning/finetune_multinode.py"
CMD="$CMD --nodes $NODES"
CMD="$CMD --master_addr $MASTER_ADDR"
CMD="$CMD --master_port $MASTER_PORT"
CMD="$CMD --nproc_per_node $GPUS_PER_NODE"
CMD="$CMD --pre-cmd \"$PRE_CMD_1\""
CMD="$CMD --pre-cmd \"$PRE_CMD_2\""
CMD="$CMD --pre-cmd \"$PRE_CMD_3\""
CMD="$CMD --"
CMD="$CMD --model_name_or_path bert-base-uncased"
CMD="$CMD --checkpoint $MODEL_CHECKPOINT"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --dataset $LABELS_FILE"
CMD="$CMD --task_type $TASK_TYPE"
CMD="$CMD --num_labels $NUM_LABELS"
CMD="$CMD --accelerator gpu"
CMD="$CMD --devices $GPUS_PER_NODE"
CMD="$CMD --num_nodes $NUM_NODES"
CMD="$CMD --strategy ddp"
CMD="$CMD --max_epochs $MAX_EPOCHS"
CMD="$CMD --train_batch_size $TRAIN_BATCH_SIZE"
CMD="$CMD --val_batch_size $VAL_BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --default_root_dir $OUTPUT_DIR"
CMD="$CMD --random_seed $RANDOM_SEED"

# Add checkpoint resume if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --ckpt_path $RESUME_CHECKPOINT"
    echo "Resuming from: $RESUME_CHECKPOINT"
fi

echo "Launching finetuning..."
echo ""

# Execute
eval $CMD

echo ""
echo "======================================================================"
echo "Finetuning complete!"
echo "======================================================================"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Extract embeddings for search"
echo "  2. Evaluate on test set"
echo "======================================================================"
