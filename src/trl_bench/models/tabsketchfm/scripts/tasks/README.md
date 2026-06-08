# Task-Agnostic Training Pipeline

This directory contains scripts for **fully decoupled** embedding generation and task execution.

## Philosophy

**Complete separation of concerns:**
- Embedding generation doesn't know about tasks
- Task execution doesn't know about embedding sources

## Two-Part Workflow

### Part 1: Generate Embeddings (Once)

**Script:** `generate_embeddings.sh`

```bash
# Generate embeddings from pretrained model
bash scripts/tasks/generate_embeddings.sh \
    --model logs/.../epoch=10.ckpt \
    --data_dir spider_join_processed_dataset \
    --output embeddings/pretrained.pkl

# Generate embeddings from raw BERT
bash scripts/tasks/generate_embeddings.sh \
    --model bert-base-uncased \
    --data_dir spider_join_processed_dataset \
    --output embeddings/raw_bert.pkl
```

**Output:** Individual table embeddings (task-agnostic)
```python
[
    {
        'table': 'customers.csv',
        'cls_embedding': [768],
        'table_embedding': [768],
        'column_embedding': {0: [768], 1: [768], ...}
    },
    ...
]
```

### Part 2: Run Task (Embedding-Agnostic)

**Script:** `run_task.py`

```bash
# Run Spider-Join task with pretrained embeddings
python scripts/tasks/run_task.py \
    --embeddings embeddings/pretrained.pkl \
    --labels spider_join/spider-join/labels.json \
    --task_name spider_join_pretrained \
    --output_dir results/spider_join_pretrained

# Run same task with raw BERT embeddings
python scripts/tasks/run_task.py \
    --embeddings embeddings/raw_bert.pkl \
    --labels spider_join/spider-join/labels.json \
    --task_name spider_join_raw_bert \
    --output_dir results/spider_join_raw_bert
```

## Complete Example: Baseline Comparison

```bash
# Step 1: Generate embeddings from different sources
bash scripts/tasks/generate_embeddings.sh \
    --model logs/.../epoch=10.ckpt \
    --output embeddings/pretrained.pkl

bash scripts/tasks/generate_embeddings.sh \
    --model bert-base-uncased \
    --output embeddings/raw_bert.pkl

# Step 2: Run same task with both embeddings
python scripts/tasks/run_task.py \
    --embeddings embeddings/pretrained.pkl \
    --labels spider_join/labels.json \
    --task_name spider_join_pretrained \
    --output_dir results/spider_join_pretrained

python scripts/tasks/run_task.py \
    --embeddings embeddings/raw_bert.pkl \
    --labels spider_join/labels.json \
    --task_name spider_join_raw_bert \
    --output_dir results/spider_join_raw_bert

# Step 3: Compare results
cat results/spider_join_pretrained/results.json
cat results/spider_join_raw_bert/results.json
```

## Multiple Tasks with Same Embeddings

```bash
# Generate embeddings once
bash scripts/tasks/generate_embeddings.sh --output embeddings/model.pkl

# Run different tasks
python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels spider_join/labels.json \
    --task_name spider_join \
    --output_dir results/spider_join

python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels wiki_union/labels.json \
    --task_name wiki_union \
    --output_dir results/wiki_union

python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels ckan_subset/labels.json \
    --task_name ckan_subset \
    --output_dir results/ckan_subset
```

## Advanced Options

### Embedding Types

```bash
# Use different embedding types
python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels task.json \
    --embedding_type cls \
    --output_dir results/cls

python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels task.json \
    --embedding_type table \
    --output_dir results/table

python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels task.json \
    --embedding_type column_mean \
    --output_dir results/column_mean
```

### Combination Methods

```bash
# Try different ways to combine table pair embeddings
python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels task.json \
    --combination_method concat \
    --output_dir results/concat

python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels task.json \
    --combination_method add \
    --output_dir results/add

python scripts/tasks/run_task.py \
    --embeddings embeddings/model.pkl \
    --labels task.json \
    --combination_method multiply \
    --output_dir results/multiply
```

## Output Format

Each task execution produces:
```
results/task_name/
├── checkpoints/
│   └── best.ckpt                # Best model checkpoint
├── lightning_logs/              # Training logs
└── results.json                 # Summary with test metrics
```

`results.json`:
```json
{
  "task_name": "spider_join_pretrained",
  "embedding_type": "cls",
  "combination_method": "concat",
  "input_dim": 1536,
  "hidden_dim": 256,
  "test_results": {
    "test_loss": 0.123,
    "total_test_accuracy": 0.95,
    "f1": 0.94
  },
  "data_stats": {
    "train": 5146,
    "valid": 742,
    "test": 1474
  }
}
```

## Advantages

1. **Generate once, reuse forever:** Extract embeddings once, run unlimited tasks
2. **Compare embedding sources:** Test different models on same task
3. **Transfer learning:** Test embeddings across different tasks
4. **Fast iteration:** No BERT forward passes during task training
5. **Modular:** Replace either component independently

## Workflow Comparison

### Old: Monolithic
```
finetune.py → Checkpoint → extract → search
(Everything coupled, must re-run for each experiment)
```

### New: Fully Decoupled
```
generate_embeddings.sh → embeddings.pkl
                              ↓
run_task.py (task1) ← embeddings.pkl → run_task.py (task2)
    ↓                                        ↓
results1/                                results2/
```

## Integration with Existing Scripts

This complements the `decoupled_pipeline/` directory:
- **tasks/**: Fully decoupled (embeddings → tasks)
- **decoupled_pipeline/**: Integrated pipeline (runs all phases)

Use `tasks/` for maximum flexibility and experimentation!
