# Table Subset

## Overview

Binary classification task to determine whether one table is a subset of another. Uses table-level embeddings to predict subset relationships between table pairs.

## Embeddings Consumed

> **Embedding Level:** Table
> **Primary Embedding:** `cls_embedding` or `table_embedding`
> **Pair Input:** Yes (table pairs)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `table_embedding` | **Yes** | `dict (v2.0)` | Table-level embeddings | TabSketchFM, TaBERT, TAPAS, TURL, Doduo |
| `column_embeddings` | Optional | `{col_idx: (dim,)}` | For column_mean | Any column model |

**Note:** Use `--embedding_type cls` for models with CLS embeddings (TAPAS, TabSketchFM), or `--embedding_type column_mean` for all models.

**Input format (v2.0):** Pickle file (`.pkl`) containing:
```python
[
    {
        'table': 'table1.csv',
        'column_embeddings': {0: [...], 1: [...]},
        'table_embedding': {
            'cls_embedding': array or None,  # for embedding_type=cls
            'table_embedding': None,
            'column_mean': array,            # for embedding_type=column_mean
        }
    },
    ...
]
```

**Helper:** Use `get_table_level_embedding(item, variant='column_mean')` to extract the desired variant.

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Binary Classification |
| **Embedding Level** | Table |
| **Pair Input** | Yes |
| **Labels** | 0=not subset, 1=subset |

## Evaluation Metrics

| Metric | Primary | Description |
|--------|---------|-------------|
| Accuracy | Yes | Classification accuracy |
| F1 | No | F1 score |

## Input Data

**Embeddings:** `embeddings/table/<model>/<dataset>.pkl`

**Labels:** JSON with train/valid/test splits:
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

## Example Commands

### Basic Classification

```bash
bash downstream_tasks/table_subset/classification/run_classification.sh \
    --embeddings embeddings/table_subset/tabsketchfm/wiki_subset.pkl \
    --labels datasets/wiki_subset/labels.json \
    --output_dir results/table_subset/tabsketchfm
```

### With Custom Hyperparameters

```bash
bash downstream_tasks/table_subset/classification/run_classification.sh \
    --embeddings embeddings/table_subset/starmie/wiki_subset.pkl \
    --labels datasets/wiki_subset/labels.json \
    --output_dir results/table_subset/starmie_diff \
    --embedding_type table \
    --combination_method diff \
    --hidden_dim 512
```

### Using Different Embedding Types

```bash
# CLS embedding (TabSketchFM)
bash downstream_tasks/table_subset/classification/run_classification.sh \
    --embeddings embeddings.pkl \
    --labels labels.json \
    --output_dir results/cls \
    --embedding_type cls

# Table embedding (Starmie)
bash downstream_tasks/table_subset/classification/run_classification.sh \
    --embeddings embeddings.pkl \
    --labels labels.json \
    --output_dir results/table \
    --embedding_type table

# Mean of column embeddings
bash downstream_tasks/table_subset/classification/run_classification.sh \
    --embeddings embeddings.pkl \
    --labels labels.json \
    --output_dir results/column_mean \
    --embedding_type column_mean
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--embeddings` | Yes | - | Path to embeddings pickle file |
| `--labels` | Yes | - | Path to labels JSON file |
| `--output_dir` | Yes | - | Output directory for results |
| `--task_name` | No | table_subset | Task name for logging |
| `--embedding_type` | No | column_mean | Embedding: cls, table, column_mean |
| `--combination_method` | No | concat | Pair combination: concat, add, multiply, diff |
| `--hidden_dim` | No | 256 | Hidden layer dimension |
| `--num_labels` | No | 2 | Number of output classes |
| `--batch_size` | No | 32 | Training batch size |
| `--max_epochs` | No | 50 | Maximum training epochs |
| `--learning_rate` | No | 1e-3 | Learning rate |
| `--dropout_prob` | No | 0.1 | Dropout probability |
| `--random_seed` | No | 0 | Random seed |
| `--accelerator` | No | gpu | Device: gpu, cpu |
| `--devices` | No | 1 | Number of devices |

## Embedding Types

| Type | Description | Best For |
|------|-------------|----------|
| `cls` | [CLS] token embedding | TabSketchFM |
| `table` | Table-level embedding | Starmie |
| `column_mean` | Mean of column embeddings | Any model |

## Combination Methods

| Method | Description | Input Dim |
|--------|-------------|-----------|
| `concat` | Concatenate [emb1; emb2] | 2 x emb_dim |
| `add` | Element-wise emb1 + emb2 | emb_dim |
| `multiply` | Element-wise emb1 * emb2 | emb_dim |
| `diff` | Absolute difference \|emb1 - emb2\| | emb_dim |

## Output

Results saved to `{output_dir}/results.json`:
```json
{
    "task_name": "table_subset",
    "embedding_type": "cls",
    "combination_method": "concat",
    "input_dim": 1536,
    "hidden_dim": 256,
    "test_results": {
        "loss": 0.312,
        "accuracy": 0.845,
        "weighted_f1": 0.832,
        "macro_f1": 0.821
    },
    "data_stats": {
        "train": 15000,
        "valid": 2000,
        "test": 3000
    }
}
```

---

## Subtasks

### Subtask: Classification

The main (and only) subtask for table subset detection.

**Location:** `downstream_tasks/table_subset/classification/`

This directory contains the shell script wrapper for the classification pipeline.

---

## Troubleshooting

### Table not found in embeddings
```
Warning: Table XYZ.csv not found in embeddings
```
**Solution:** Ensure all tables in labels are included in embedding generation.

### Low accuracy
- Try different `--combination_method` (diff often works well)
- Increase `--hidden_dim`
- Try different embedding types

### Class imbalance
If accuracy is low, check label distribution. Consider:
- Weighted loss
- Data augmentation
- Undersampling majority class

## Related

- Embedding generation: `models/tabsketchfm/USAGE.md`, `models/starmie/USAGE.md`
- Similar tasks: `downstream_tasks/join_search/classification/`, `downstream_tasks/union_search/classification/`
