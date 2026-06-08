# Join Search Containment

This directory contains scripts for training regression/classification models on containment estimation tasks using pre-extracted embeddings.

## Overview

The containment task estimates how much one table's column is contained in another table's column. Per the TabSketchFM paper, this is formulated as a regression task to estimate containment values (R² metric). The script can also run as a binary classification task.

The regression script assumes embeddings have already been generated and focuses solely on training the regression/classification head. This allows for fast iteration on hyperparameters without re-extracting embeddings.

## Usage

### Basic Usage (Regression)

```bash
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings <path_to_embeddings.pkl> \
    --labels <path_to_labels.json> \
    --output_dir <output_directory>
```

### Example - Regression

```bash
# Using wiki-join-search embeddings for containment regression
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings embeddings/wiki_containment_starmie.pkl \
    --labels datasets/wiki_containment/labels.json \
    --output_dir results/wiki_containment_regression \
    --task_type regression \
    --embedding_type column \
    --combination_method concat
```

### Example - Classification

```bash
# Using binary classification with threshold
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings embeddings/wiki_containment_starmie.pkl \
    --labels datasets/wiki_containment/labels.json \
    --output_dir results/wiki_containment_classification \
    --task_type classification \
    --num_labels 2 \
    --embedding_type column
```

### Advanced Options

```bash
bash downstream_tasks/join_search/containment/run_regression.sh \
    --embeddings embeddings/wiki_containment_starmie.pkl \
    --labels datasets/wiki_containment/labels.json \
    --output_dir results/wiki_containment_custom \
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

### Required
- `--embeddings`: Path to embeddings pickle file (in TabSketchFM format)
- `--labels`: Path to labels JSON file (with train/valid/test splits)
- `--output_dir`: Directory to save results

### Optional
- `--task_name`: Task name for logging (default: `wiki_containment`)
- `--task_type`: Task type: `regression`, `classification` (default: `regression`)
- `--embedding_type`: Which embedding to use: `cls`, `table`, `column_mean`, `column` (default: `column`)
  - **Recommended**: Use `column` for containment tasks, as it uses specific column embeddings from the labels
- `--combination_method`: How to combine table pairs: `concat`, `add`, `multiply`, `diff` (default: `concat`)
- `--hidden_dim`: Hidden layer dimension (default: `256`)
- `--num_labels`: Number of outputs (default: `1` for regression, `2` for classification)
- `--batch_size`: Training batch size (default: `32`)
- `--max_epochs`: Maximum training epochs (default: `50`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--dropout_prob`: Dropout probability (default: `0.1`)
- `--random_seed`: Random seed for reproducibility (default: `0`)
- `--accelerator`: Hardware accelerator: `gpu`, `cpu` (default: `gpu`)
- `--devices`: Number of devices to use (default: `1`)

## Embedding Format

The script expects embeddings in TabSketchFM format (list of dicts):

```python
[
    {
        'table': 'path/to/table.csv',
        'column_embedding': {0: [...], 1: [...], ...},  # Required for column-level embedding_type
        'cls_embedding': [...],      # optional
        'table_embedding': [...]     # optional
    },
    ...
]
```

**Note**: For containment tasks, column embeddings are essential since the task operates at the column level. The script will use the specific columns specified in `join_col_table1` and `join_col_table2` from the labels.

See `utils/embedding_conversion/` for conversion scripts from other formats (e.g., Starmie).

## Labels Format

The labels file should be a JSON with train/valid/test splits. For regression, labels are continuous scores (0.0 to 1.0). For classification, labels are binary (0 or 1).

### Regression Format

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

### Classification Format

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

**Important**: The `join_col_table1` and `join_col_table2` fields specify which columns to use for the containment comparison. These are used when `--embedding_type column` is specified.

## Generating Labels

To generate labels from containment ground truth:

```bash
python utils/data_utils/generate_containment_labels.py \
    --input datasets/wiki-join-search/labels/join_search_containment_min_gt.jsonl \
    --tables_dir datasets/wiki-join-search/tables \
    --output datasets/wiki_containment/labels.json \
    --task_type regression \
    --negative_ratio 1.0
```

For classification:

```bash
python utils/data_utils/generate_containment_labels.py \
    --input datasets/wiki-join-search/labels/join_search_containment_min_gt.jsonl \
    --tables_dir datasets/wiki-join-search/tables \
    --output datasets/wiki_containment/labels.json \
    --task_type classification \
    --threshold 0.05 \
    --negative_ratio 1.0
```

## Output

Results are saved to the specified output directory:
- `checkpoints/best.ckpt`: Best model checkpoint
- `results.json`: Test metrics and configuration
- `lightning_logs/`: Training logs

### Regression Metrics
- MSE (Mean Squared Error)
- R² (Coefficient of Determination)

### Classification Metrics
- Accuracy
- F1 Score

## Full Pipeline Example

For a complete end-to-end pipeline including label generation, embedding extraction, and training, see:

```bash
bash scripts/starmie/join_search_containment_wiki_containment.sh
```

This script handles:
1. Label generation from ground truth
2. Model training (Starmie)
3. Embedding extraction
4. Format conversion
5. Regression/classification training

## Tips

1. **Use column-level embeddings**: For containment tasks, always use `--embedding_type column` for best results, as containment operates at the column level.

2. **Fast experimentation**: Since embeddings are pre-extracted, you can quickly iterate on:
   - Different combination methods (concat, diff, add, multiply)
   - Various architectures (hidden dimensions)
   - Hyperparameters (learning rate, batch size)
   - Task types (regression vs classification)

3. **Comparing embeddings**: Use the same labels with different embedding files to compare embedding quality.

4. **Reproducibility**: Set `--random_seed` for deterministic results.

5. **Regression vs Classification**:
   - Use regression for continuous containment scores (more informative)
   - Use classification for binary containment decisions (simpler, but less nuanced)
