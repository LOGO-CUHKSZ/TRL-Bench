# TUTA (Tree-based Transformers for Tabular Data)

## Overview

TUTA generates row-level embeddings using a tree-structured transformer architecture that captures the hierarchical nature of tables.

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `row_embedding` | `(768,)` | One embedding per row |

**Output files:**
- `train_embeddings.npy` - Training set row embeddings
- `test_embeddings.npy` - Test set row embeddings
- `train_labels.npy` - Training labels (if provided)
- `test_labels.npy` - Test labels (if provided)
- `metadata.json` - Embedding metadata

## Model Type

| Property | Value |
|----------|-------|
| **Training Required** | No (pretrained) |
| **Embedding Dimension** | 768 |

TUTA uses pretrained weights and generates embeddings without additional training.

**Checkpoint required:** Download `tuta.bin` from the official repository.

## Input Data

**Expected directory structure:**
```
data_dir/
├── train.csv
└── test.csv
```

**CSV requirements:**
- First row: column headers
- Supports: categorical and continuous columns
- Optional: label column for downstream tasks

## Example Commands

### With Label Column

```bash
python models/tuta/generate_row_embeddings.py \
    --dataset_dir datasets/adult \
    --output_dir embeddings/row_prediction/tuta/adult \
    --model_path checkpoints/tuta/tuta.bin \
    --label_column income \
    --device_id 0
```

### Without Label Column

```bash
python models/tuta/generate_row_embeddings.py \
    --dataset_dir datasets/adult \
    --output_dir embeddings/row_prediction/tuta/adult \
    --model_path checkpoints/tuta/tuta.bin \
    --device_id 0
```

## Downstream Tasks

- Row classification
- Row regression
- Table understanding
