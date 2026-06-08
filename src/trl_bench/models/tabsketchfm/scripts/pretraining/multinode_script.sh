#!/bin/bash
# ==============================================================================
# Multi-Node Pretraining Launch Script
# ==============================================================================
# This script launches distributed pretraining across multiple nodes using
# pretrain_multinode.py, which orchestrates torchrun on each node via SSH.
#
# Prerequisites:
#   1. SSH access to all nodes without password (ssh-key based)
#   2. Same environment setup on all nodes (conda/venv, CUDA, etc.)
#   3. Shared filesystem accessible from all nodes (for data and checkpoints)
#   4. Preprocessed data splits: data_splits.json.bz2 (see PIPELINE.md)
#
# Usage:
#   1. Edit the configuration below to match your cluster
#   2. Run from project root: bash models/tabsketchfm/scripts/pretraining/multinode_script.sh
#
# Configuration Parameters:
#   NODES          - Space-separated list of hostnames
#   MASTER_ADDR    - Hostname of the master node (usually first node)
#   GPUS_PER_NODE  - Number of GPUs per node
#   MASTER_PORT    - Port for distributed communication
#   PRE_COMMANDS   - Commands to setup environment on each node
#
# See TRAINING.md for complete documentation.
# ==============================================================================

set -e

# Get project root (4 levels up from this script: pretraining -> scripts -> tabsketchfm -> models -> root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

# TabSketchFM module path
TABSKETCHFM_DIR="$PROJECT_ROOT/models/tabsketchfm"

# ==============================================================================
# CLUSTER CONFIGURATION - EDIT THESE FOR YOUR SETUP
# ==============================================================================

# List of node hostnames (space-separated)
NODES="kn086 kn087 kn089 kn091 kn092 kn095"

# Master node address (usually the first node)
MASTER_ADDR="kn086"

# GPUs per node
GPUS_PER_NODE=4

# Port for distributed communication (change if port conflicts)
MASTER_PORT=12951

# ==============================================================================
# ENVIRONMENT SETUP - EDIT THESE FOR YOUR CLUSTER
# ==============================================================================

# Commands to run on each node before training
# These are executed in order via --pre-cmd flags
# NOTE: Uses the original tabsketchfm venv (Python 3.12, PyTorch 2.9.0) for compatibility
# The TRL venv has PyTorch 2.6.0 which has NCCL synchronization issues in multinode training
PRE_CMD_1="module load python/3.12"
PRE_CMD_2="module load cuda/12.2"
PRE_CMD_3="source /path/to/tabsketchfm/venv/bin/activate"
PRE_CMD_4="export PYTHONPATH=\$PWD/models/tabsketchfm:\$PWD/models/tabsketchfm/tabsketchfm:\$PYTHONPATH"

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

# Dataset (paths relative to project root)
DATASET="datasets/tabsketchfm/data_splits_filtered_256.json.bz2"
DATA_DIR="datasets/tabsketchfm"

# Training hyperparameters
MAX_EPOCHS=40
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
LEARNING_RATE=1e-4
RANDOM_SEED=0

# Output directories (relative to project root)
OUTPUT_DIR="logs/tabsketchfm-pretrain"
MODEL_SAVE_PATH="checkpoints/tabsketchfm/bert_model"

# Weights & Biases (optional - comment out to disable)
USE_WANDB=true
WANDB_PROJECT="tabsketchfm-pretrain"
WANDB_RUN_NAME="multinode-$(date +%Y%m%d-%H%M%S)"

# Resume from checkpoint (leave empty for fresh start)
RESUME_CHECKPOINT=""
# RESUME_CHECKPOINT="checkpoints/tabsketchfm/epoch=10-step=27786.ckpt"

# ==============================================================================
# BUILD AND RUN COMMAND
# ==============================================================================

# Count nodes
NUM_NODES=$(echo $NODES | wc -w)

echo "======================================================================"
echo "MULTI-NODE PRETRAINING"
echo "======================================================================"
echo "Project root:    $PROJECT_ROOT"
echo "Nodes:           $NODES"
echo "Num nodes:       $NUM_NODES"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "Total GPUs:      $((NUM_NODES * GPUS_PER_NODE))"
echo "Master:          $MASTER_ADDR:$MASTER_PORT"
echo "Dataset:         $DATASET"
echo "Max epochs:      $MAX_EPOCHS"
echo "Batch size:      $TRAIN_BATCH_SIZE (per GPU)"
echo "Learning rate:   $LEARNING_RATE"
echo "Output:          $OUTPUT_DIR"
echo "======================================================================"
echo ""

# Build the command
CMD="python models/tabsketchfm/scripts/pretraining/pretrain_multinode.py"
CMD="$CMD --nodes $NODES"
CMD="$CMD --master_addr $MASTER_ADDR"
CMD="$CMD --master_port $MASTER_PORT"
CMD="$CMD --nproc_per_node $GPUS_PER_NODE"
CMD="$CMD --pre-cmd \"$PRE_CMD_1\""
CMD="$CMD --pre-cmd \"$PRE_CMD_2\""
CMD="$CMD --pre-cmd \"$PRE_CMD_3\""
CMD="$CMD --pre-cmd \"$PRE_CMD_4\""
CMD="$CMD --"
CMD="$CMD --accelerator gpu"
CMD="$CMD --devices $GPUS_PER_NODE"
CMD="$CMD --num_nodes $NUM_NODES"
CMD="$CMD --strategy ddp"
CMD="$CMD --max_epochs $MAX_EPOCHS"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --train_batch_size $TRAIN_BATCH_SIZE"
CMD="$CMD --val_batch_size $VAL_BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --random_seed $RANDOM_SEED"
CMD="$CMD --save_bert_model"
CMD="$CMD --bert_model_path $MODEL_SAVE_PATH"
CMD="$CMD --default_root_dir $OUTPUT_DIR"

# Add W&B logging if enabled
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
    CMD="$CMD --wandb_project $WANDB_PROJECT"
    CMD="$CMD --wandb_run_name $WANDB_RUN_NAME"
fi

# Add checkpoint resume if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --ckpt_path $RESUME_CHECKPOINT"
    echo "Resuming from: $RESUME_CHECKPOINT"
fi

echo "Launching training..."
echo ""

# Execute
eval $CMD

echo ""
echo "======================================================================"
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
echo "Model saved to: $MODEL_SAVE_PATH"
echo "======================================================================"
