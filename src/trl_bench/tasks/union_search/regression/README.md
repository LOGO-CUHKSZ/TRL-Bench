# Union Search Regression

Regression for table union search using pre-extracted embeddings.

## Overview

This directory contains scripts for training a regression model that predicts the similarity/difference between pairs of tables. The regressor operates on pre-extracted table embeddings.

**Task:** Given two tables, predict a continuous value representing their relationship
**Primary Dataset:** ECB-Union (dimension difference between ECB data slices, labels 1-12)
**Metrics:** MSE (Mean Squared Error), R² (coefficient of determination)

## Quick Start

### Basic Usage

```bash
bash downstream_tasks/union_search/regression/run_regression.sh \
    --embeddings embeddings/union_search/ecb_union_embeddings.pkl \
    --labels datasets/ecb_union/labels.json \
    --output_dir results/evaluation/union_search/model_name/ecb_union
```

### With Custom Hyperparameters

```bash
bash downstream_tasks/union_search/regression/run_regression.sh \
    --embeddings embeddings/union_search/ecb_union_embeddings.pkl \
    --labels datasets/ecb_union/labels.json \
    --output_dir results/evaluation/union_search/model_name/ecb_union_diff_512 \
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
| `--task_name` | `ecb_union` | Task name for logging |
| `--embedding_type` | `column_mean` | Embedding type: `cls`, `table`, `column_mean` |
| `--combination_method` | `concat` | Pair combination: `concat`, `add`, `multiply`, `diff` |
| `--hidden_dim` | `256` | Hidden layer dimension |
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

JSON with train/valid/test splits (continuous labels for regression):
```json
{
    "train": [
        {
            "table1": {"filename": "table1.csv"},
            "table2": {"filename": "table2.csv"},
            "label": 3.0
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
    "task_name": "ecb_union",
    "task_type": "regression",
    "embedding_type": "column_mean",
    "combination_method": "concat",
    "input_dim": 1536,
    "hidden_dim": 256,
    "num_labels": 1,
    "test_results": {
        "test_loss": 0.143,
        "test_mse": 0.143,
        "test_r2": 0.945
    },
    "data_stats": {
        "train": 15344,
        "valid": 4270,
        "test": 4270
    }
}
```

## ECB-Union Task Details

The European Central Bank (ECB) organizes economic data into distinct datasets with multiple dimensions. The ECB-Union task measures how many dimensions differ between pairs of data slices:

- **Label range:** 1-12 (continuous)
- **Label meaning:** Number of dimensions that differ between two ECB data slices
- **Lower values:** Tables are more similar (share more dimensions)
- **Higher values:** Tables are more different (share fewer dimensions)

This is useful for:
- Finding comparable economic datasets
- Identifying related data slices for analysis
- Ranking tables by similarity

## Embedding Types

### `column_mean` (Default)
Computes mean of all column embeddings on-the-fly. Works with any model that produces column embeddings.

```bash
--embedding_type column_mean
```

### `cls`
Uses the [CLS] token embedding from the model. Best for models pretrained with CLS pooling (e.g., TabSketchFM).

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
| `concat` | Concatenate [emb1; emb2] | 2 x emb_dim |
| `add` | Element-wise add emb1 + emb2 | emb_dim |
| `multiply` | Element-wise multiply emb1 * emb2 | emb_dim |
| `diff` | Absolute difference \|emb1 - emb2\| | emb_dim |

For regression tasks, `diff` often works well as it directly captures the "distance" between embeddings.

## Examples

### Using Different Embedding Types

```bash
# With CLS embeddings (TabSketchFM)
bash downstream_tasks/union_search/regression/run_regression.sh \
    --embeddings embeddings/union_search/tabsketchfm_ecb_embeddings.pkl \
    --labels datasets/ecb_union/labels.json \
    --output_dir results/evaluation/union_search/tabsketchfm/ecb_union \
    --embedding_type cls

# With table embeddings (Starmie)
bash downstream_tasks/union_search/regression/run_regression.sh \
    --embeddings embeddings/union_search/starmie_ecb_embeddings.pkl \
    --labels datasets/ecb_union/labels.json \
    --output_dir results/evaluation/union_search/starmie/ecb_union \
    --embedding_type table
```

### Hyperparameter Search

```bash
# Try different architectures
for hidden in 128 256 512; do
    bash downstream_tasks/union_search/regression/run_regression.sh \
        --embeddings embeddings/union_search/embeddings.pkl \
        --labels datasets/ecb_union/labels.json \
        --output_dir results/evaluation/union_search/model/ecb_union_h${hidden} \
        --hidden_dim $hidden
done

# Try different combination methods
for method in concat add multiply diff; do
    bash downstream_tasks/union_search/regression/run_regression.sh \
        --embeddings embeddings/union_search/embeddings.pkl \
        --labels datasets/ecb_union/labels.json \
        --output_dir results/evaluation/union_search/model/ecb_union_${method} \
        --combination_method $method
done
```

## Interpreting Results

### R² (Coefficient of Determination)
- **Range:** -inf to 1.0
- **1.0:** Perfect predictions
- **0.0:** Model predicts the mean for all samples
- **< 0:** Model is worse than predicting the mean

### MSE (Mean Squared Error)
- **Range:** 0 to inf
- **Lower is better**
- For ECB-Union (labels 1-12), MSE < 1.0 indicates predictions within ~1 dimension on average

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
**Solution:** Ensure `datasets/ecb_union/labels.json` exists.

### Poor R² score
If R² is very low or negative:
1. Try different combination methods (`diff` often works well for regression)
2. Increase hidden_dim
3. Check that embeddings were extracted correctly
4. Try different embedding types

## Related Scripts

- `downstream_tasks/union_search/classification/` - Classification version for binary union tasks
- `scripts/tabsketchfm/union_search_regression_ecb_union.sh` - Full TabSketchFM pipeline
- `utils/downstream/run_task.py` - Underlying regressor implementation
