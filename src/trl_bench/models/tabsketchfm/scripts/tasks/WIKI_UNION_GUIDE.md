# Wiki-Union Decoupled Pipeline Guide

This guide explains how to use the decoupled approach for the Wiki-Union table union search task.

## What is the Decoupled Approach?

The decoupled approach separates embedding extraction from classifier training:

```
┌─────────────────────────────────────────────────────────────┐
│ Traditional Finetuning (scripts/finetuning/)               │
│                                                              │
│  Pretrained Model → Full Finetuning → Checkpoint → Results  │
│  (slow, must re-run entire pipeline for experiments)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Decoupled Approach (scripts/tasks/)                         │
│                                                              │
│  Pretrained Model → Extract Embeddings (once)               │
│                              ↓                               │
│                       embeddings.pkl                         │
│                              ↓                               │
│                     Train Classifier (fast)                  │
│  (can iterate quickly, compare models, test hyperparameters) │
└─────────────────────────────────────────────────────────────┘
```

## Advantages

1. **Extract once, reuse forever**: Embeddings are model-specific, not task-specific
2. **Fast iteration**: Train lightweight classifier in minutes instead of hours
3. **Easy comparison**: Test different models (pretrained vs raw BERT) on the same task
4. **Flexible experimentation**: Try different embedding types, combination methods, architectures

## Dataset Overview

- **Task**: Table Union Search (binary classification)
- **Tables**: 40,752 Wikipedia tables
- **Pairs**: 376,384 (50% positive, 50% negative)
- **Splits**:
  - Train: 301,108 pairs
  - Valid: 37,638 pairs
  - Test: 37,638 pairs

## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run both phases (extraction + training)
bash scripts/tasks/run_wiki_union.sh
```

This will:
1. Extract embeddings from pretrained TabSketchFM → `embeddings/wiki_union_embeddings.pkl`
2. Train classifier on paired embeddings → `results/wiki_union_decoupled/`

### Option 2: Run Phases Separately

```bash
# Phase 1: Extract embeddings (once)
bash scripts/tasks/generate_embeddings.sh \
    --model logs/tabsketchfm-pretrain/.../epoch=10.ckpt \
    --data_dir wiki_union_processed \
    --output embeddings/wiki_union_embeddings.pkl

# Phase 2: Train classifier (can iterate)
python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_embeddings.pkl \
    --labels labels.json \
    --task_name wiki_union \
    --output_dir results/wiki_union_decoupled
```

### Option 3: Compare Models

```bash
# Compare pretrained TabSketchFM vs raw BERT
bash scripts/tasks/compare_models_wiki_union.sh
```

This extracts embeddings from both models and compares their performance.

## Workflow Details

### Phase 1: Extract Embeddings

**Input**:
- Pretrained model checkpoint
- Preprocessed tables (`wiki_union_processed/*.json.bz2`)

**Process**:
```bash
python scripts/embedding_extraction/extract_embeddings_unified.py \
    --model_name_or_path CHECKPOINT \
    --model_type pretrained \
    --data_dir wiki_union_processed \
    --output_file embeddings/wiki_union_embeddings.pkl \
    --batch_size 256
```

**Output**: Individual table embeddings (task-agnostic)
```python
[
    {
        'table': 'table_123.csv',
        'cls_embedding': [768],        # CLS token representation
        'table_embedding': [768],      # Mean-pooled table tokens
        'column_embedding': {          # Per-column embeddings
            0: [768],
            1: [768],
            ...
        }
    },
    ...
]
```

### Phase 2: Train Classifier

**Input**:
- Individual table embeddings
- Labels file defining table pairs

**Process**:
1. Load embeddings and labels
2. Create paired embeddings (concatenate table1 + table2)
3. Train 2-layer MLP classifier: `input_dim → hidden_dim → num_labels`

```bash
python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_embeddings.pkl \
    --labels labels.json \
    --task_name wiki_union \
    --output_dir results/wiki_union_decoupled \
    --embedding_type cls \
    --combination_method concat \
    --hidden_dim 256 \
    --num_labels 2 \
    --max_epochs 50 \
    --batch_size 32 \
    --learning_rate 2e-5
```

**Output**:
```
results/wiki_union_decoupled/
├── checkpoints/
│   └── best.ckpt              # Best model checkpoint
├── lightning_logs/            # Training logs
└── results.json               # Test metrics summary
```

## Advanced Usage

### Try Different Embedding Types

```bash
# CLS token embedding (default)
python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_embeddings.pkl \
    --labels labels.json \
    --embedding_type cls \
    --output_dir results/wiki_union_cls

# Table-level mean pooling
python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_embeddings.pkl \
    --labels labels.json \
    --embedding_type table \
    --output_dir results/wiki_union_table

# Mean of all column embeddings
python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_embeddings.pkl \
    --labels labels.json \
    --embedding_type column_mean \
    --output_dir results/wiki_union_column_mean
```

### Try Different Combination Methods

How to combine table pair embeddings:

```bash
# Concatenate [emb1; emb2] (default, input_dim = 1536)
--combination_method concat

# Element-wise addition emb1 + emb2 (input_dim = 768)
--combination_method add

# Element-wise multiplication emb1 * emb2 (input_dim = 768)
--combination_method multiply

# Absolute difference |emb1 - emb2| (input_dim = 768)
--combination_method diff
```

### Compare Model Sources

```bash
# Generate embeddings from different sources
bash scripts/tasks/generate_embeddings.sh \
    --model logs/.../pretrained.ckpt \
    --data_dir wiki_union_processed \
    --output embeddings/wiki_union_pretrained.pkl

bash scripts/tasks/generate_embeddings.sh \
    --model bert-base-uncased \
    --data_dir wiki_union_processed \
    --output embeddings/wiki_union_raw_bert.pkl

# Train on same task with both
python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_pretrained.pkl \
    --labels labels.json \
    --output_dir results/pretrained

python scripts/tasks/run_task.py \
    --embeddings embeddings/wiki_union_raw_bert.pkl \
    --labels labels.json \
    --output_dir results/raw_bert

# Compare results
cat results/pretrained/results.json
cat results/raw_bert/results.json
```

## Iteration Examples

Since embeddings are pre-extracted, you can quickly iterate:

```bash
# Already extracted embeddings once
bash scripts/tasks/run_wiki_union.sh  # Initial run

# Now iterate quickly with --skip_extraction
bash scripts/tasks/run_wiki_union.sh --skip_extraction
```

This is useful for:
- Testing different hyperparameters
- Trying different architectures
- Debugging training issues
- Re-running experiments

## Output Interpretation

`results.json` contains:
```json
{
  "task_name": "wiki_union",
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
    "train": 301108,
    "valid": 37638,
    "test": 37638
  }
}
```

## Comparison with Full Finetuning

| Aspect | Full Finetuning | Decoupled |
|--------|-----------------|-----------|
| **Time to first result** | Hours (full training) | Minutes (classifier only) |
| **Model comparison** | Re-run full pipeline | Reuse embeddings |
| **Hyperparameter tuning** | Slow | Fast |
| **Memory usage** | High (full BERT) | Low (MLP only) |
| **Flexibility** | Limited | High |

## Integration with Existing Workflows

The decoupled approach complements full finetuning:

1. **Use full finetuning** for:
   - Final production models
   - When you need end-to-end training

2. **Use decoupled approach** for:
   - Rapid experimentation
   - Model comparison
   - Hyperparameter search
   - Understanding embedding quality

## Troubleshooting

### Embeddings extraction is slow
- Reduce `--batch_size` if running out of memory
- Use GPU acceleration (automatically detected)

### Training fails with dimension mismatch
- Check `--embedding_type` and `--combination_method`
- `concat` doubles the dimension (768 → 1536)
- Other methods keep dimension (768)

### Missing tables in output
- The script reports skipped pairs due to missing tables
- This is normal if some tables failed preprocessing
- Check preprocessing logs if too many are skipped

## Scripts Reference

- `run_wiki_union.sh`: Complete pipeline (extraction + training)
- `compare_models_wiki_union.sh`: Compare pretrained vs raw BERT
- `generate_embeddings.sh`: Extract embeddings only
- `run_task.py`: Train classifier only

## Next Steps

After running the decoupled pipeline:

1. Compare results with full finetuning (scripts/finetuning/multinode_wiki_union.sh)
2. Try different embedding sources (finetuned models, raw BERT)
3. Experiment with classifier architectures
4. Apply same embeddings to other tasks (if you have multi-task data)
