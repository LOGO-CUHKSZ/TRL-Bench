# Union Search Classification

Binary classification for table union search using pre-extracted embeddings.

## Overview

This directory contains scripts for training a binary classifier that predicts whether two tables are unionable. The classifier operates on pre-extracted table embeddings.

**Task:** Given two tables, predict if they can be unioned (same schema)
**Labels:** 0 = non-unionable, 1 = unionable
**Dataset:** wiki_union (301K train, 37K test pairs)

## Quick Start

### Basic Usage

```bash
bash downstream_tasks/union_search/classification/run_classification.sh \
    --embeddings embeddings/union_search/wiki_union_embeddings.pkl \
    --labels datasets/wiki_union/labels.json \
    --output_dir results/evaluation/union_search/model_name/wiki_union
```

### With Custom Hyperparameters

```bash
bash downstream_tasks/union_search/classification/run_classification.sh \
    --embeddings embeddings/union_search/wiki_union_embeddings.pkl \
    --labels datasets/wiki_union/labels.json \
    --output_dir results/evaluation/union_search/model_name/wiki_union_diff_512 \
    --combination_method diff \
    --hidden_dim 512 \
    --max_epochs 30
```

## Configuration

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--embeddings PATH` | Path to pickled embeddings file |
| `--labels PATH` | Path to labels JSON file |
| `--output_dir PATH` | Output directory for results |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_name` | `union_search` | Task name for logging |
| `--embedding_type` | `column_mean` | Embedding type: `cls`, `table`, `column_mean` |
| `--combination_method` | `concat` | Pair combination: `concat`, `add`, `multiply`, `diff` |
| `--hidden_dim` | `256` | Hidden layer dimension |
| `--num_labels` | `2` | Number of output classes |
| `--batch_size` | `32` | Training batch size |
| `--max_epochs` | `50` | Maximum training epochs |
| `--learning_rate` | `1e-3` | Learning rate |
| `--dropout_prob` | `0.1` | Dropout probability |
| `--random_seed` | `42` | Random seed |
| `--accelerator` | `gpu` | Device type: `gpu`, `cpu` |
| `--devices` | `1` | Number of devices |

## Input Formats

### Embeddings Format

Pickled list of dicts:
```python
[
    {
        'table': 'table1.csv',
        'cls_embedding': [768-dim vector],      # for embedding_type=cls
        'table_embedding': [768-dim vector],    # for embedding_type=table
        'column_embedding': {0: [...], 1: [...]}  # for embedding_type=column_mean
    },
    ...
]
```

### Labels Format

JSON with train/valid/test splits:
```json
{
    "train": [
        {
            "table1": {"filename": "table1.csv"},
            "table2": {"filename": "table2.csv"},
            "label": 1
        },
        ...
    ],
    "valid": [...],
    "test": [...]
}
```

## Output

Results are saved to `{output_dir}/results.json`:

```json
{
    "task_name": "union_search",
    "embedding_type": "column_mean",
    "combination_method": "concat",
    "input_dim": 1536,
    "hidden_dim": 256,
    "test_results": {
        "test_loss": 0.234,
        "total_test_accuracy": 0.892,
        "f1": 0.875
    },
    "data_stats": {
        "train": 301108,
        "valid": 21506,
        "test": 37638
    }
}
```

## Embedding Types

### `column_mean` (Default)
Computes mean of all column embeddings on-the-fly. Works with any model that produces column embeddings.

```bash
--embedding_type column_mean
```

### `cls`
Uses the [CLS] token embedding from the model. Best for models pretrained with CLS pooling.

```bash
--embedding_type cls
```

### `table`
Uses table-level embedding (typically mean of column embeddings). Best for Starmie models.

```bash
--embedding_type table
```

## Combination Methods

How to combine two table embeddings into a pair representation:

| Method | Description | Input Dim |
|--------|-------------|-----------|
| `concat` | Concatenate [emb1; emb2] | 2 × emb_dim |
| `add` | Element-wise add emb1 + emb2 | emb_dim |
| `multiply` | Element-wise multiply emb1 * emb2 | emb_dim |
| `diff` | Absolute difference \|emb1 - emb2\| | emb_dim |

## Examples

### Using Different Embedding Types

```bash
# With CLS embeddings (TabSketchFM)
bash downstream_tasks/union_search/classification/run_classification.sh \
    --embeddings embeddings/union_search/tabsketchfm_embeddings.pkl \
    --labels datasets/wiki_union/labels.json \
    --output_dir results/evaluation/union_search/tabsketchfm/wiki_union \
    --embedding_type cls

# With table embeddings (Starmie)
bash downstream_tasks/union_search/classification/run_classification.sh \
    --embeddings embeddings/union_search/starmie_embeddings.pkl \
    --labels datasets/wiki_union/labels.json \
    --output_dir results/evaluation/union_search/starmie/wiki_union \
    --embedding_type table
```

### Hyperparameter Search

```bash
# Try different architectures
for hidden in 128 256 512; do
    bash downstream_tasks/union_search/classification/run_classification.sh \
        --embeddings embeddings/union_search/embeddings.pkl \
        --labels datasets/wiki_union/labels.json \
        --output_dir results/evaluation/union_search/starmie/wiki_union_h${hidden} \
        --hidden_dim $hidden
done

# Try different combination methods
for method in concat add multiply diff; do
    bash downstream_tasks/union_search/classification/run_classification.sh \
        --embeddings embeddings/union_search/embeddings.pkl \
        --labels datasets/wiki_union/labels.json \
        --output_dir results/evaluation/union_search/starmie/wiki_union_${method} \
        --combination_method $method
done
```

## Troubleshooting

### Embeddings file not found
```
Error: Embeddings file not found: ...
```
**Solution:** Extract embeddings first using your embedding extraction pipeline.

### Labels file not found
```
Error: Labels file not found: ...
```
**Solution:** Ensure `datasets/wiki_union/labels.json` exists.

### Table filename mismatch
```
Warning: Table XYZ.csv not found in embeddings
```
**Solution:** Table filenames in embeddings must match those in labels exactly. Check that:
- Embeddings cover all tables referenced in labels
- Filenames match exactly (case-sensitive)

### Out of memory
```
CUDA out of memory
```
**Solution:** Reduce batch size:
```bash
--batch_size 16
```

## Related Scripts

- `scripts/starmie/union_search_santos.sh` - Union search with Starmie (HNSW ranking)
- `downstream_tasks/join_search/classification/` - Similar pipeline for join search
- `utils/downstream/run_task.py` - Underlying classifier implementation
