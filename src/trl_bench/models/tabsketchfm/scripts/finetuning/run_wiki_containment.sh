#!/bin/bash
# ==============================================================================
# Wiki-Containment Finetuning Workflow
# ==============================================================================
# This script runs the complete finetuning workflow for Wiki-Containment task.
# Wiki-Containment uses containment similarity metric for join discovery.
#
# NOTE: This script uses the wiki-join-search dataset for containment finetuning.
# The wiki-containment.tar.bz2 archive only contains 47 sample tables, which is
# insufficient for finetuning. The wiki-join-search dataset has 46K+ tables and
# containment ground truth in join_search_containment_min_gt.jsonl.
#
# Prerequisites:
#   1. Download wiki-join-search.tar.bz2 and extract
#   2. (Optional) Download wiki-containment.tar.bz2 for sample tables
#
# Usage (from project root):
#   # With pretrained TabSketchFM checkpoint (default)
#   bash scripts/finetuning/run_wiki_containment.sh
#
#   # With raw BERT (no tabular pretraining - baseline)
#   bash scripts/finetuning/run_wiki_containment.sh --model_name_or_path bert-base-uncased
#
#   # With custom checkpoint
#   bash scripts/finetuning/run_wiki_containment.sh --model_name_or_path path/to/checkpoint.ckpt
#
#   # Skip preprocessing (if already done)
#   bash scripts/finetuning/run_wiki_containment.sh --skip_preprocessing
#
#   # Skip label generation (if labels.json already exists)
#   bash scripts/finetuning/run_wiki_containment.sh --skip_label_generation
# ==============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is two levels up from scripts/finetuning/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default parameters
# MODEL_NAME_OR_PATH can be:
#   - a .ckpt file (pretrained TabSketchFM checkpoint)
#   - a model name (e.g., bert-base-uncased for raw BERT baseline)
# MODEL_NAME_OR_PATH="logs/tabsketchfm-pretrain/tabsketchfm-pretrain/tem0b5h7/checkpoints/epoch=10-step=27786.ckpt"
MODEL_NAME_OR_PATH="logs/tabsketchfm-pretrain-filtered/tabsketchfm-pretrain/bn6oq8sa/checkpoints/epoch=11-step=9084.ckpt"
DATA_DIR="wiki_containment_processed"
# OUTPUT_DIR="./wiki_containment_finetuned"
OUTPUT_DIR="./wiki_containment_finetuned_filtered"
DEVICES=4                # GPUs per node
NUM_NODES=10              # Number of nodes (default: 1 for single-node)
STRATEGY="auto"          # Training strategy (auto, ddp, ddp_find_unused_parameters_true)
MAX_EPOCHS=50
BATCH_SIZE=128
RANDOM_SEED=0
TASK_TYPE="regression"   # Task type: regression or classification
NUM_LABELS=1             # Number of labels: 1 for regression, 2 for binary classification
THRESHOLD=0.05            # Threshold for classification (score >= threshold → label=1)
SKIP_PREPROCESSING=""
SKIP_LABEL_GENERATION=""

# ==============================================================================
# SSH-BASED MULTI-NODE CONFIGURATION (Optional)
# ==============================================================================
# If NODES is set, the script will use finetune_multinode.py for SSH-based
# distributed training across multiple nodes. Otherwise, uses regular finetune.py
# with PyTorch Lightning's native multi-node support (via SLURM or torchrun).
#
# Example for SSH-based multi-node:
#   NODES="kn087 kn092 kn097 kn098"
#   MASTER_ADDR="kn087"
#   MASTER_PORT=12952
#   PRE_CMD_1="module load python/3.10"
#   PRE_CMD_2="module load cuda/12.2"
#   PRE_CMD_3="source venv/bin/activate"
# ==============================================================================

# List of node hostnames (space-separated) - leave empty for single-node or SLURM/torchrun
NODES="kn091 kn092 kn093 kn085 kn086 kn088 kn099 kn100 kn169 kn170"

# Master node address (usually the first node)
MASTER_ADDR="kn091"

# Port for distributed communication (change if port conflicts)
MASTER_PORT=12952

# Commands to run on each node before training (for SSH-based multi-node)
PRE_CMD_1="source load_env"
PRE_CMD_2=""
PRE_CMD_3=""

# Learning rate auto-scaling configuration
# Scales LR proportionally with effective batch size (batch_size × num_gpus × num_nodes)
# Formula: scaled_lr = base_lr × (effective_batch_size / base_batch_size)
AUTO_SCALE_LR=true
BASE_BATCH_SIZE=1024       # Reference batch size for base LR
BASE_LR=2e-5             # Base learning rate (2e-5 is BERT default)
LEARNING_RATE=""         # Will be calculated if AUTO_SCALE_LR=true

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --num_nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --num_labels)
            NUM_LABELS="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --skip_preprocessing)
            SKIP_PREPROCESSING=1
            shift
            ;;
        --skip_label_generation)
            SKIP_LABEL_GENERATION=1
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            AUTO_SCALE_LR=false  # Disable auto-scaling if manually set
            shift 2
            ;;
        --no_auto_scale_lr)
            AUTO_SCALE_LR=false
            shift
            ;;
        --base_lr)
            BASE_LR="$2"
            shift 2
            ;;
        --base_batch_size)
            BASE_BATCH_SIZE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --pre_cmd_1)
            PRE_CMD_1="$2"
            shift 2
            ;;
        --pre_cmd_2)
            PRE_CMD_2="$2"
            shift 2
            ;;
        --pre_cmd_3)
            PRE_CMD_3="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Wiki-Containment finetuning workflow."
            echo ""
            echo "Options:"
            echo "  --model_name_or_path PATH  .ckpt checkpoint OR model name (e.g., bert-base-uncased)"
            echo "                             (default: pretrained TabSketchFM checkpoint)"
            echo "  --data_dir PATH            Directory with preprocessed data"
            echo "  --output_dir PATH          Output directory for finetuned model"
            echo "  --devices INT              Number of GPUs per node (default: 4)"
            echo "  --num_nodes INT            Number of nodes for distributed training (default: 1)"
            echo "  --strategy STR             Training strategy: auto, ddp, ddp_find_unused_parameters_true (default: auto)"
            echo "  --max_epochs INT           Maximum epochs (default: 50)"
            echo "  --batch_size INT           Batch size per GPU (default: 128)"
            echo "  --random_seed INT          Random seed (default: 0)"
            echo "  --task_type TYPE           Task type: regression or classification (default: regression)"
            echo "  --num_labels INT           Number of labels: 1 for regression, 2 for binary classification"
            echo "  --threshold FLOAT          Threshold for classification (default: 0.5)"
            echo ""
            echo "Learning Rate Options:"
            echo "  --learning_rate FLOAT      Manual learning rate (disables auto-scaling)"
            echo "  --no_auto_scale_lr         Disable automatic LR scaling (use base_lr)"
            echo "  --base_lr FLOAT            Base learning rate for scaling (default: 2e-5)"
            echo "  --base_batch_size INT      Reference batch size for scaling (default: 256)"
            echo "                             Auto-scaled LR = base_lr × (batch_size × devices × num_nodes / base_batch_size)"
            echo "                             IMPORTANT: Use 256+ to avoid excessive scaling with multi-node training"
            echo ""
            echo "Other Options:"
            echo "  --skip_preprocessing       Skip data preprocessing step"
            echo "  --skip_label_generation    Skip label generation step"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "SSH-Based Multi-Node Options (uses finetune_multinode.py):"
            echo "  --nodes \"node1 node2...\"    Space-separated list of node hostnames"
            echo "  --master_addr HOSTNAME     Master node address (default: first node)"
            echo "  --master_port PORT         Communication port (default: 12952)"
            echo "  --pre_cmd_1 \"COMMAND\"      First pre-command to run on each node"
            echo "  --pre_cmd_2 \"COMMAND\"      Second pre-command to run on each node"
            echo "  --pre_cmd_3 \"COMMAND\"      Third pre-command to run on each node"
            echo ""
            echo "Multi-Node Usage:"
            echo "  1. SSH-based (requires passwordless SSH):"
            echo "     bash $0 --nodes \"kn087 kn092\" --devices 8 --batch_size 128 \\"
            echo "             --pre_cmd_1 \"module load cuda/12.2\" \\"
            echo "             --pre_cmd_2 \"source venv/bin/activate\""
            echo ""
            echo "  2. SLURM/torchrun (PyTorch Lightning native):"
            echo "     bash $0 --num_nodes 4 --devices 8 --strategy ddp --batch_size 128"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# LEARNING RATE AUTO-SCALING
# ==============================================================================
# Calculate learning rate based on effective batch size if not manually set
if [ "$AUTO_SCALE_LR" = true ] && [ -z "$LEARNING_RATE" ]; then
    EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * DEVICES * NUM_NODES))

    # Calculate scaled LR using Python (handles scientific notation correctly)
    # Formula: scaled_lr = base_lr × (effective_batch_size / base_batch_size)
    LEARNING_RATE=$(python3 -c "
base_lr = float('$BASE_LR')
effective_batch = $EFFECTIVE_BATCH_SIZE
base_batch = $BASE_BATCH_SIZE
scaled_lr = base_lr * (effective_batch / base_batch)
print(f'{scaled_lr:.10f}')
")

    SCALE_FACTOR=$(python3 -c "print($EFFECTIVE_BATCH_SIZE / $BASE_BATCH_SIZE)")

    echo "================================================"
    echo "AUTO-SCALING LEARNING RATE"
    echo "================================================"
    echo "Base LR:              $BASE_LR"
    echo "Base batch size:      $BASE_BATCH_SIZE"
    echo "Effective batch size: $EFFECTIVE_BATCH_SIZE (batch_size=$BATCH_SIZE × devices=$DEVICES × nodes=$NUM_NODES)"
    echo "Scale factor:         $SCALE_FACTOR"
    echo "Scaled LR:            $LEARNING_RATE"

    # Warn if scaling is too aggressive
    if (( $(python3 -c "print(1 if $SCALE_FACTOR > 20 else 0)") )); then
        echo ""
        echo "⚠️  WARNING: Scale factor >20x may be too aggressive for regression!"
        echo "   Consider increasing BASE_BATCH_SIZE or using --no_auto_scale_lr"
        echo "   Recommended: BASE_BATCH_SIZE >= 256 for multi-node training"
    fi

    echo "================================================"
    echo ""
elif [ -n "$LEARNING_RATE" ]; then
    echo "Using manually set learning rate: $LEARNING_RATE"
    echo ""
else
    # Auto-scaling disabled, use base LR
    LEARNING_RATE=$BASE_LR
    echo "Auto-scaling disabled. Using base LR: $LEARNING_RATE"
    echo ""
fi

# ==============================================================================
# DETECT SSH-BASED MULTI-NODE MODE
# ==============================================================================
# If NODES is set, use SSH-based multi-node via finetune_multinode.py
# Otherwise, use regular finetune.py (supports single-node or SLURM/torchrun multi-node)

USE_SSH_MULTINODE=false

if [ -n "$NODES" ]; then
    USE_SSH_MULTINODE=true

    # Count number of nodes
    NUM_NODES=$(echo $NODES | wc -w)

    # Set master address to first node if not specified
    if [ -z "$MASTER_ADDR" ]; then
        MASTER_ADDR=$(echo $NODES | awk '{print $1}')
    fi

    # Recalculate LR with updated NUM_NODES (if auto-scaling)
    if [ "$AUTO_SCALE_LR" = true ]; then
        EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * DEVICES * NUM_NODES))
        LEARNING_RATE=$(python3 -c "
base_lr = float('$BASE_LR')
effective_batch = $EFFECTIVE_BATCH_SIZE
base_batch = $BASE_BATCH_SIZE
scaled_lr = base_lr * (effective_batch / base_batch)
print(f'{scaled_lr:.10f}')
")
    fi

    # Force strategy to ddp for SSH-based multi-node
    STRATEGY="ddp"

    echo "================================================"
    echo "SSH-BASED MULTI-NODE MODE DETECTED"
    echo "================================================"
    echo "Nodes:           $NODES"
    echo "Master address:  $MASTER_ADDR:$MASTER_PORT"
    echo "Num nodes:       $NUM_NODES"
    echo "Total GPUs:      $((NUM_NODES * DEVICES))"
    echo "Strategy:        $STRATEGY (forced for SSH-based multi-node)"
    if [ "$AUTO_SCALE_LR" = true ]; then
        echo "LR recalculated: $LEARNING_RATE (for effective batch: $EFFECTIVE_BATCH_SIZE)"
    fi
    echo "================================================"
    echo ""
fi

# Determine if using checkpoint or raw model
IS_CHECKPOINT=false
CHECKPOINT_ARG=""
BASE_MODEL="bert-base-uncased"

if [[ "$MODEL_NAME_OR_PATH" == *.ckpt ]] && [[ -f "$MODEL_NAME_OR_PATH" ]]; then
    IS_CHECKPOINT=true
    CHECKPOINT_ARG="--checkpoint $MODEL_NAME_OR_PATH"
    echo "Mode: Pretrained TabSketchFM checkpoint"
else
    BASE_MODEL="$MODEL_NAME_OR_PATH"
    echo "Mode: Raw model ($MODEL_NAME_OR_PATH) - no tabular pretraining"
fi

echo "================================================"
echo "TabSketchFM: Wiki-Containment Finetuning Workflow"
echo "================================================"
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Num nodes: $NUM_NODES"
echo "Devices per node: $DEVICES"
echo "Total GPUs: $((NUM_NODES * DEVICES))"
echo "Strategy: $STRATEGY"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size (per GPU): $BATCH_SIZE"
echo "Effective batch size: $((BATCH_SIZE * DEVICES * NUM_NODES))"
echo "Learning rate: $LEARNING_RATE"
echo "Random seed: $RANDOM_SEED"
echo "Task type: $TASK_TYPE"
if [ "$TASK_TYPE" = "classification" ]; then
    echo "Num labels: $NUM_LABELS"
    echo "Classification threshold: $THRESHOLD"
fi
echo "================================================"
echo ""

# Validate input directories - prefer wiki-join-search (full dataset) over wiki_containment (sample)
TABLES_DIR=""
LABELS_DIR="wiki_containment"  # Where to save generated labels

if [[ -d "wiki-join-search/tables" ]]; then
    TABLES_DIR="wiki-join-search/tables"
    echo "Using wiki-join-search tables (46K+ tables with containment ground truth)"
elif [[ -d "wiki_containment/tables" ]]; then
    TABLES_DIR="wiki_containment/tables"
    echo "Warning: Using wiki_containment tables (only 47 sample tables)"
    echo "For better results, download wiki-join-search.tar.bz2"
else
    echo "Error: No tables directory found!"
    echo ""
    echo "Please download and extract wiki-join-search.tar.bz2:"
    echo "  tar -xjf wiki-join-search.tar.bz2"
    echo ""
    echo "Or wiki-containment.tar.bz2 (sample only):"
    echo "  tar -xjf wiki-containment.tar.bz2"
    echo "  mkdir -p wiki_containment && mv tables README.txt wiki_containment/"
    exit 1
fi

# Step 0: Generate labels (if needed)
if [[ -z "$SKIP_LABEL_GENERATION" ]]; then
    echo ""
    echo "Step 0: Generating finetuning labels from containment ground truth..."
    echo "-------------------------------------------------------------------"

    # Check for containment ground truth
    if [[ -f "wiki-join-search/labels/join_search_containment_min_gt.jsonl" ]]; then
        GROUND_TRUTH_FILE="wiki-join-search/labels/join_search_containment_min_gt.jsonl"
    else
        echo "Error: Could not find containment ground truth file!"
        echo "Expected: wiki-join-search/labels/join_search_containment_min_gt.jsonl"
        echo ""
        echo "Please download and extract wiki-join-search.tar.bz2 first."
        exit 1
    fi

    mkdir -p "$LABELS_DIR"

    python scripts/data_utils/generate_containment_labels.py \
        --input "$GROUND_TRUTH_FILE" \
        --tables_dir "$TABLES_DIR" \
        --output "$LABELS_DIR/labels.json" \
        --task_type "$TASK_TYPE" \
        --threshold "$THRESHOLD" \
        --negative_ratio 1.0 \
        --seed "$RANDOM_SEED"

    if [ $? -ne 0 ]; then
        echo "Label generation failed!"
        exit 1
    fi
    echo ""
else
    echo "Skipping Step 0: Using existing labels in $LABELS_DIR/labels.json"
    echo ""
fi

# Validate labels file
if [[ ! -f "$LABELS_DIR/labels.json" ]]; then
    echo "Error: $LABELS_DIR/labels.json not found!"
    echo "Run without --skip_label_generation to generate it."
    exit 1
fi

# Step 1: Preprocessing (optional)
if [[ -z "$SKIP_PREPROCESSING" ]]; then
    echo "Step 1: Preprocessing tables from $TABLES_DIR..."
    echo "-------------------------------------------"
    mkdir -p "$DATA_DIR"

    python tabsketchfm/batch_fastdata.py \
        --input_dir "$TABLES_DIR" \
        --output_dir "$DATA_DIR"

    if [ $? -ne 0 ]; then
        echo "Preprocessing failed!"
        exit 1
    fi

    # Check if preprocessing produced files
    NUM_FILES=$(find "$DATA_DIR" -name "*.json.bz2" | wc -l)
    echo "Preprocessing complete: $NUM_FILES files generated"
    echo ""
else
    echo "Skipping Step 1: Using existing preprocessed data in $DATA_DIR"
    echo ""
fi

# Step 2: Finetuning
echo "Step 2: Finetuning on wiki-containment task..."
echo "-------------------------------------------"

# Build the command based on multi-node mode
if [ "$USE_SSH_MULTINODE" = true ]; then
    # SSH-based multi-node using finetune_multinode.py
    CMD="python scripts/finetuning/finetune_multinode.py"
    CMD="$CMD --nodes $NODES"
    CMD="$CMD --master_addr $MASTER_ADDR"
    CMD="$CMD --master_port $MASTER_PORT"
    CMD="$CMD --nproc_per_node $DEVICES"

    # Add pre-commands if specified
    if [ -n "$PRE_CMD_1" ]; then
        CMD="$CMD --pre-cmd \"$PRE_CMD_1\""
    fi
    if [ -n "$PRE_CMD_2" ]; then
        CMD="$CMD --pre-cmd \"$PRE_CMD_2\""
    fi
    if [ -n "$PRE_CMD_3" ]; then
        CMD="$CMD --pre-cmd \"$PRE_CMD_3\""
    fi

    # Separator between multinode args and finetune.py args
    CMD="$CMD --"

    # Add finetune.py arguments
    CMD="$CMD --model_name_or_path $BASE_MODEL"
    CMD="$CMD $CHECKPOINT_ARG"
    CMD="$CMD --data_dir $DATA_DIR"
    CMD="$CMD --dataset $LABELS_DIR/labels.json"
    CMD="$CMD --task_type $TASK_TYPE"
    CMD="$CMD --num_labels $NUM_LABELS"
    CMD="$CMD --accelerator gpu"
    CMD="$CMD --devices $DEVICES"
    CMD="$CMD --num_nodes $NUM_NODES"
    CMD="$CMD --strategy $STRATEGY"
    CMD="$CMD --max_epochs $MAX_EPOCHS"
    CMD="$CMD --train_batch_size $BATCH_SIZE"
    CMD="$CMD --val_batch_size $BATCH_SIZE"
    CMD="$CMD --learning_rate $LEARNING_RATE"
    CMD="$CMD --default_root_dir $OUTPUT_DIR"
    CMD="$CMD --random_seed $RANDOM_SEED"
else
    # Regular single-node or SLURM/torchrun multi-node
    CMD="python finetune.py"
    CMD="$CMD --model_name_or_path $BASE_MODEL"
    CMD="$CMD $CHECKPOINT_ARG"
    CMD="$CMD --data_dir $DATA_DIR"
    CMD="$CMD --dataset $LABELS_DIR/labels.json"
    CMD="$CMD --task_type $TASK_TYPE"
    CMD="$CMD --num_labels $NUM_LABELS"
    CMD="$CMD --accelerator gpu"
    CMD="$CMD --devices $DEVICES"
    CMD="$CMD --num_nodes $NUM_NODES"
    CMD="$CMD --strategy $STRATEGY"
    CMD="$CMD --max_epochs $MAX_EPOCHS"
    CMD="$CMD --train_batch_size $BATCH_SIZE"
    CMD="$CMD --val_batch_size $BATCH_SIZE"
    CMD="$CMD --default_root_dir $OUTPUT_DIR"
    CMD="$CMD --random_seed $RANDOM_SEED"
    CMD="$CMD --learning_rate $LEARNING_RATE"
fi

echo "Running: $CMD"
eval $CMD

if [ $? -ne 0 ]; then
    echo "Finetuning failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "Workflow complete!"
echo "================================================"
echo ""
echo "Finetuned model saved in: $OUTPUT_DIR/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Extract embeddings: python extract_embeddings.py --checkpoint $OUTPUT_DIR/checkpoints/best.ckpt"
echo "  2. Perform join search: python embedding_search.py --embeddings embeddings.pkl --k 10 --by_cols True"
echo ""
