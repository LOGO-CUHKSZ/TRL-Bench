# DAE (Denoising Autoencoder)

## Overview

DAE generates row-level embeddings by learning to reconstruct corrupted tabular data. It produces dense vector representations for each row in the dataset.

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `row_embedding` | `(embedding_dim,)` | One embedding per row |

**Output files:**
- `train_embeddings.npy` - Training set row embeddings
- `test_embeddings.npy` - Test set row embeddings
- `train_labels.npy` - Training labels (if provided)
- `test_labels.npy` - Test labels (if provided)
- `metadata.json` - Embedding metadata

## Model Type

| Property | Value |
|----------|-------|
| **Training Required** | Yes |
| **Embedding Dimension** | Configurable (default: 256) |

DAE must be trained on your dataset before generating embeddings. The model learns dataset-specific patterns through self-supervised reconstruction.

## Input Data

### Option A: Pre-split Data (Recommended)

```
data_dir/
├── train.csv
└── test.csv
```

### Option B: Single CSV File

A single CSV file can be provided and will be automatically split:
```
input_file.csv  # Will be split based on --split_ratio
```

**CSV requirements:**
- First row: column headers
- Supports: categorical and continuous columns
- Optional: label column for downstream tasks

## Example Commands

### Step 1: Train the Model

```bash
python models/dae/train.py \
    --data_dir datasets/adult \
    --label_column income \
    --checkpoint_dir models/dae/checkpoints/adult \
    --max_epochs 100
```

### Step 2: Generate Embeddings

#### From Pre-split Data (Default)

```bash
python models/dae/generate_embeddings.py \
    --data_dir datasets/adult \
    --checkpoint_dir models/dae/checkpoints/adult \
    --embedding_dir embeddings/row_prediction/dae/adult
```

#### From Single CSV File

```bash
python models/dae/generate_embeddings.py \
    --input datasets/adult/data.csv \
    --checkpoint_dir models/dae/checkpoints/adult \
    --embedding_dir embeddings/row_prediction/dae/adult \
    --split_ratio 0.8 \
    --random_seed 42
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes* | - | Directory with train.csv and test.csv |
| `--input` | Yes* | - | Single CSV file (alternative to --data_dir) |
| `--checkpoint_dir` | Yes | - | Directory containing model checkpoints |
| `--embedding_dir` | Yes | - | Output directory for embeddings |
| `--batch_size` | No | 256 | Batch size for embedding generation |
| `--num_workers` | No | 4 | Number of data loading workers |
| `--checkpoint` | No | - | Path to specific checkpoint file |
| `--split_ratio` | No | 0.8 | Train/test split ratio (only for --input mode) |
| `--random_seed` | No | 42 | Random seed for splitting |
*Either `--data_dir` or `--input` must be specified (mutually exclusive).

## Output Format

Creates `metadata.json` with full provenance:

```json
{
  "version": "1.0",
  "format": "unified_row_embedding",
  "model_name": "DAE",
  "embedding_dim": 256,
  "embedding_level": "row",
  "train_samples": 32561,
  "test_samples": 16281,
  "has_labels": true,
  "label_column": "income",
  "feature_columns": ["age", "workclass", ...],
  "checkpoint_path": "checkpoints/dae/model.ckpt",
  "generation_config": {
    "batch_size": 256,
    "data_source": "datasets/adult",
    "split_ratio": null
  }
}
```

## Downstream Tasks

- Row classification
- Row regression
- Clustering
- Anomaly detection

## Next Steps

After generating embeddings, run the downstream task:

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/dae/adult
```
