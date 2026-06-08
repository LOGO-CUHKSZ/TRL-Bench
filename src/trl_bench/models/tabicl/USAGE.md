# TabICL (Tabular In-Context Learner)

## Overview

TabICL generates row-level embeddings using a transformer architecture pretrained for tabular in-context learning (ICML 2025). It extracts 512-dimensional representations from its internal Stage 2 (row_interactor), which captures feature interactions and data distribution without using label information.

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `row_embedding` | `(512,)` | One embedding per row (fixed dimension) |

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
| **Embedding Dimension** | 512 (fixed: 4 CLS tokens x 128) |
| **Extraction Point** | Stage 2 (row_interactor) |

TabICL comes with pretrained weights and generates embeddings without additional training. Embeddings are label-free — supervised and self-supervised modes produce identical embeddings because labels only enter at Stage 3 (predictor), which is not used for extraction.

**Operating modes:**
- `supervised` - Label column excluded from features, saved with embeddings
- `self-supervised` - Generates embeddings without label information

> **Note:** Since embeddings are extracted before the label-dependent stage, supervised and self-supervised modes produce identical embeddings for the same feature set.

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
- Numeric and categorical features supported (automatic encoding applied)
- Optional: label column

## Example Commands

### Self-Supervised Mode

#### From Pre-split Data (Default)

```bash
python models/TabICL/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/tabicl/adult \
    --mode self-supervised
```

#### From Single CSV File

```bash
python models/TabICL/generate_embeddings_train_test.py \
    --input datasets/adult/data.csv \
    --embedding_dir embeddings/row_prediction/tabicl/adult \
    --mode self-supervised \
    --split_ratio 0.8 \
    --random_seed 42
```

### Supervised Mode

```bash
python models/TabICL/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/tabicl/adult \
    --label_column income \
    --mode supervised
```

### Self-Supervised with Label Saved

```bash
python models/TabICL/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/tabicl/adult \
    --label_column income \
    --mode self-supervised
```

### Single File Mode (Legacy Pickle Output)

```bash
python models/TabICL/generate_embeddings_single_file.py \
    --csv_file datasets/adult/train.csv \
    --output_dir embeddings/row_prediction/tabicl/adult \
    --label_column income \
    --mode self-supervised
```

## CLI Arguments

### generate_embeddings_train_test.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes* | - | Directory with train.csv and test.csv |
| `--input` | Yes* | - | Single CSV file (alternative to --data_dir) |
| `--embedding_dir` | No | `embeddings/row_prediction/TabICL` | Output directory for embeddings |
| `--mode` | No | auto | Operating mode: `supervised` or `self-supervised` |
| `--label_column` | No | - | Label column name |
| `--n_estimators` | No | 1 | Number of estimators (see note below) |
| `--checkpoint_version` | No | `tabicl-classifier-v1.1-0506.ckpt` | Model checkpoint |
| `--device` | No | auto | Device: `auto`, `cuda`, `cpu` |
| `--split_ratio` | No | 0.8 | Train/test split ratio (only for --input mode) |
| `--random_seed` | No | 42 | Random seed for splitting |
*Either `--data_dir` or `--input` must be specified (mutually exclusive).

### generate_embeddings_single_file.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv_file` | Yes | - | Path to CSV file |
| `--output_dir` | No | `embeddings/row_prediction/TabICL` | Output directory |
| `--mode` | No | auto | Operating mode |
| `--label_column` | No | - | Label column name |
| `--n_estimators` | No | 1 | Number of estimators |
| `--checkpoint_version` | No | `tabicl-classifier-v1.1-0506.ckpt` | Model checkpoint |
| `--device` | No | auto | Device |

> **Note on `--n_estimators`:** Default is 1 (unlike TabPFN's default of 8). TabICL's ensemble uses feature column shuffles with RoPE positional encodings. Averaging embeddings across different column orderings is semantically incoherent. Use n_estimators=1 for embeddings.

## Output Format

Creates `metadata.json` with full provenance:

```json
{
  "version": "1.0",
  "format": "unified_row_embedding",
  "model_name": "TabICL",
  "embedding_dim": 512,
  "embedding_level": "row",
  "train_samples": 800,
  "test_samples": 200,
  "has_labels": true,
  "label_column": "income",
  "feature_columns": ["age", "workclass", "..."],
  "checkpoint_path": "tabicl-classifier-v1.1-0506.ckpt",
  "generation_config": {
    "mode": "self-supervised",
    "n_estimators": 1,
    "extraction_point": "Stage 2 (row_interactor)",
    "data_source": "datasets/adult"
  }
}
```

## Downstream Tasks

- Row classification
- Few-shot learning
- Transfer learning
- Clustering

## Next Steps

After generating embeddings, run the downstream task:

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/tabicl/adult
```
