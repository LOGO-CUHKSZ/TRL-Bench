# Join Search Classification

This directory contains scripts for training classifiers on join search tasks using pre-extracted embeddings.

## Overview

The classification script assumes embeddings have already been generated and focuses solely on training the classification head. This allows for fast iteration on hyperparameters without re-extracting embeddings.

## Usage

### Basic Usage

```bash
bash downstream_tasks/join_search/classification/run_classification.sh \
    --embeddings <path_to_embeddings.pkl> \
    --labels <path_to_labels.json> \
    --output_dir <output_directory>
```

### Example

```bash
# Using wiki-join-search embeddings
bash downstream_tasks/join_search/classification/run_classification.sh \
    --embeddings embeddings/wiki_join_search_starmie.pkl \
    --labels data/wiki_join_search/labels.json \
    --output_dir results/wiki_join_search_cls
```

### Advanced Options

```bash
bash downstream_tasks/join_search/classification/run_classification.sh \
    --embeddings embeddings/wiki_join_search_starmie.pkl \
    --labels data/wiki_join_search/labels.json \
    --output_dir results/wiki_join_search_custom \
    --task_name wiki_join_search \
    --embedding_type table \
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
- `--task_name`: Task name for logging (default: `join_search`)
- `--embedding_type`: Which embedding to use: `cls`, `table`, `column_mean` (default: `cls`)
- `--combination_method`: How to combine table pairs: `concat`, `add`, `multiply`, `diff` (default: `concat`)
- `--hidden_dim`: Hidden layer dimension (default: `256`)
- `--num_labels`: Number of output classes (default: `2`)
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
        'column_embedding': {0: [...], 1: [...], ...},
        'cls_embedding': [...],      # optional
        'table_embedding': [...]     # optional
    },
    ...
]
```

See `utils/embedding_conversion/` for conversion scripts from other formats (e.g., Starmie).

## Labels Format

The labels file should be a JSON with train/valid/test splits:

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

Results are saved to the specified output directory:
- `checkpoints/best.ckpt`: Best model checkpoint
- `results.json`: Test metrics and configuration
- `lightning_logs/`: Training logs

## Tips

1. **Fast experimentation**: Since embeddings are pre-extracted, you can quickly iterate on:
   - Different combination methods (concat, diff, add, multiply)
   - Various architectures (hidden dimensions)
   - Hyperparameters (learning rate, batch size)

2. **Comparing embeddings**: Use the same labels with different embedding files to compare embedding quality

3. **Reproducibility**: Set `--random_seed` for deterministic results
