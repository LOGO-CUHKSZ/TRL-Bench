# Join Search Containment

## Overview

Estimates the containment (overlap) between column pairs. Given two columns from different tables, predicts how much one column's values are contained in another. Can be formulated as regression (continuous containment score) or classification (binary joinable/not).

Reference: TabSketchFM paper (arXiv:2407.01619)

## Embeddings Consumed

> **Embedding Level:** Column (specific columns from pairs)
> **Primary Embedding:** `column_embeddings[col_idx]`
> **Pair Input:** Yes (column pairs with join_col_table1, join_col_table2)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `column_embeddings` | **Yes** | `{col_idx: (dim,)}` | Per-column embedding | TabSketchFM, Starmie, TaBERT, TAPAS |
| `table_embedding` | Optional | `dict (v2.0)` | Table-level embeddings | All models |

**Input format (v2.0):** Pickle file (`.pkl`) containing:
```python
[
    {
        'table': 'path/to/table.csv',
        'column_embeddings': {0: array, 1: array, ...},  # Required
        'table_embedding': {
            'cls_embedding': array or None,
            'table_embedding': None,
            'column_mean': array,
        }
    },
    ...
]
```

**Helper:** Use `get_table_level_embedding(item, variant='column_mean')` to extract the desired variant.

**Note:** For containment tasks, column embeddings are essential since the task operates at the column level. The script uses specific columns specified in `join_col_table1` and `join_col_table2` from the labels.

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Regression (default) / Classification |
| **Embedding Level** | Column |
| **Pair Input** | Yes |
| **Primary Metric** | R² (regression) / F1 (classification) |

## Evaluation Metrics

### Regression
| Metric | Primary | Description |
|--------|---------|-------------|
| R² | Yes | Coefficient of determination |
| MSE | No | Mean squared error |

### Classification
| Metric | Primary | Description |
|--------|---------|-------------|
| F1 | Yes | F1 score |
| Accuracy | No | Classification accuracy |

## Input Data

**Embeddings:** `embeddings/join_search/containment/<model>/<dataset>.pkl`

**Labels format:** JSON with train/valid/test splits:

### Regression Labels
```json
{
    "train": [
        {
            "table1": {"filename": "table1.csv"},
            "table2": {"filename": "table2.csv"},
            "label": 0.85,
            "join_col_table1": "0",
            "join_col_table2": "2"
        },
        ...
    ],
    "valid": [...],
    "test": [...]
}
```

### Classification Labels
```json
{
    "train": [
        {
            "table1": {"filename": "table1.csv"},
            "table2": {"filename": "table2.csv"},
            "label": 1,
            "join_col_table1": "0",
            "join_col_table2": "2"
        },
        ...
    ],
    "valid": [...],
    "test": [...]
}
```

**Important:** `join_col_table1` and `join_col_table2` specify which columns to use for the containment comparison.

## Example Commands

### Regression (Default)

```bash
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings embeddings/join_search/containment/starmie/wiki_containment.pkl \
    --labels datasets/wiki_containment/labels.json \
    --output_dir results/wiki_containment_regression \
    --task_type regression \
    --embedding_type column \
    --combination_method concat
```

### Classification

```bash
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings embeddings/join_search/containment/starmie/wiki_containment.pkl \
    --labels datasets/wiki_containment/labels.json \
    --output_dir results/wiki_containment_classification \
    --task_type classification \
    --num_labels 2 \
    --embedding_type column
```

### Advanced Configuration

```bash
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings embeddings.pkl \
    --labels labels.json \
    --output_dir results/custom \
    --task_name wiki_containment \
    --task_type regression \
    --embedding_type column \
    --combination_method diff \
    --hidden_dim 512 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --batch_size 64
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--embeddings` | Yes | - | Path to embeddings pickle file |
| `--labels` | Yes | - | Path to labels JSON file |
| `--output_dir` | Yes | - | Directory to save results |
| `--task_name` | No | wiki_containment | Task name for logging |
| `--task_type` | No | regression | Task type: regression, classification |
| `--embedding_type` | No | column | Embedding: cls, table, column_mean, column |
| `--combination_method` | No | concat | Pair combination: concat, add, multiply, diff |
| `--hidden_dim` | No | 256 | Hidden layer dimension |
| `--num_labels` | No | 1 | Number of outputs (1 for regression, 2 for classification) |
| `--batch_size` | No | 32 | Training batch size |
| `--max_epochs` | No | 50 | Maximum training epochs |
| `--learning_rate` | No | 2e-5 | Learning rate |
| `--dropout_prob` | No | 0.1 | Dropout probability |
| `--random_seed` | No | 0 | Random seed |
| `--accelerator` | No | gpu | Hardware: gpu, cpu |
| `--devices` | No | 1 | Number of devices |

## Embedding Types

| Type | Description | Use When |
|------|-------------|----------|
| `column` | Uses specific column embeddings from labels | **Recommended** for containment |
| `cls` | [CLS] token embedding | Models with CLS pooling |
| `table` | Table-level embedding | Table-level comparison |
| `column_mean` | Mean of all column embeddings | No specific columns |

**Recommendation:** Always use `--embedding_type column` for containment tasks, as containment operates at the column level.

## Combination Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `concat` | Concatenate [emb1; emb2] | Default, general purpose |
| `diff` | Absolute difference \|emb1 - emb2\| | Regression, similarity |
| `add` | Element-wise emb1 + emb2 | Symmetric relationships |
| `multiply` | Element-wise emb1 * emb2 | Interaction features |

## Output

Results are saved to `{output_dir}/`:
- `checkpoints/best.ckpt`: Best model checkpoint
- `results.json`: Test metrics and configuration
- `lightning_logs/`: Training logs

### Regression Metrics
```json
{
    "test_mse": 0.043,
    "test_r2": 0.892
}
```

### Classification Metrics
```json
{
    "test_accuracy": 0.875,
    "test_f1": 0.862
}
```

## Generating Labels

To generate labels from containment ground truth:

### For Regression
```bash
python utils/data_utils/generate_containment_labels.py \
    --input datasets/wiki-join-search/labels/join_search_containment_min_gt.jsonl \
    --tables_dir datasets/wiki-join-search/tables \
    --output datasets/wiki_containment/labels.json \
    --task_type regression \
    --negative_ratio 1.0
```

### For Classification
```bash
python utils/data_utils/generate_containment_labels.py \
    --input datasets/wiki-join-search/labels/join_search_containment_min_gt.jsonl \
    --tables_dir datasets/wiki-join-search/tables \
    --output datasets/wiki_containment/labels.json \
    --task_type classification \
    --threshold 0.05 \
    --negative_ratio 1.0
```

---

## Troubleshooting

### Column ID out of bounds
```
Warning: table_id col_id X >= Y embeddings
```
**Solution:** Ensure embeddings were generated for all columns in the tables.

### Poor R² score
- Try `--combination_method diff` (often better for regression)
- Increase `--hidden_dim`
- Check embedding quality with different models
- Verify labels are correct (0-1 range for containment)

### Table not found in embeddings
```
Warning: Table XYZ.csv not found in embeddings
```
**Solution:** Ensure all tables in labels are included in embedding generation.

## Related

- Parent task: `downstream_tasks/join_search/USAGE.md`
- Join classification: `downstream_tasks/join_search/classification/README.md`
- Embedding generation: `models/starmie/USAGE.md`, `models/tabsketchfm/USAGE.md`
