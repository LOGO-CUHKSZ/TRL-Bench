# TabPFN (Tabular Prior-Fitted Network)

## Overview

TabPFN generates row-level embeddings using a transformer architecture pre-trained on synthetic tabular data. It performs in-context learning without task-specific training.

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
| **Training Required** | No (pretrained) |
| **Embedding Dimension** | Model-dependent |

TabPFN comes with pretrained weights and generates embeddings without additional training.

**Operating modes:**
- `supervised` - Uses label column as target for context-aware embeddings
- `self-supervised` - Generates embeddings without label information

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
- Numeric features preferred (automatic encoding applied)
- Optional: label column

**Usage notes:**
- Designed for datasets with <1000 training samples
- Up to ~100 features recommended

## Example Commands

### Self-Supervised Mode

#### From Pre-split Data (Default)

```bash
python models/TabPFN/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/tabpfn/adult \
    --mode self-supervised
```

#### From Single CSV File

```bash
python models/TabPFN/generate_embeddings_train_test.py \
    --input datasets/adult/data.csv \
    --embedding_dir embeddings/row_prediction/tabpfn/adult \
    --mode self-supervised \
    --split_ratio 0.8 \
    --random_seed 42
```

### Supervised Mode

```bash
python models/TabPFN/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/tabpfn/adult \
    --label_column income \
    --mode supervised
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes* | - | Directory with train.csv and test.csv |
| `--input` | Yes* | - | Single CSV file (alternative to --data_dir) |
| `--embedding_dir` | Yes | - | Output directory for embeddings |
| `--mode` | No | self-supervised | Operating mode: `supervised` or `self-supervised` |
| `--label_column` | No | - | Label column (required for supervised mode) |
| `--n_estimators` | No | 8 | Number of ensemble estimators |
| `--batch_size` | No | 256 | Batch size for embedding generation |
| `--split_ratio` | No | 0.8 | Train/test split ratio (only for --input mode) |
| `--random_seed` | No | 42 | Random seed for splitting |
*Either `--data_dir` or `--input` must be specified (mutually exclusive).

## Output Format

Creates `metadata.json` with full provenance:

```json
{
  "version": "1.0",
  "format": "unified_row_embedding",
  "model_name": "TabPFN",
  "embedding_dim": 512,
  "embedding_level": "row",
  "train_samples": 800,
  "test_samples": 200,
  "has_labels": true,
  "label_column": "income",
  "feature_columns": ["age", "workclass", ...],
  "checkpoint_path": null,
  "generation_config": {
    "mode": "supervised",
    "n_estimators": 8,
    "data_source": "datasets/adult",
    "split_ratio": null
  }
}
```

## Downstream Tasks

- Row classification (especially small datasets)
- Few-shot learning
- Transfer learning

## Next Steps

After generating embeddings, run the downstream task:

```bash
python downstream_tasks/row_prediction/train_downstream.py \
    --embedding_dir embeddings/row_prediction/tabpfn/adult
```

## Notes

- **Version pin (no token needed).** The `[tabpfn]` extra pins
  `tabpfn==6.4.1` (+ `tabpfn-extensions==0.2.2`), the paper-time version.
  6.4.1 downloads its weights from a public Google Cloud bucket
  (`storage.googleapis.com/tabpfn-v2-model-files/...`) with **no credentials**.
  Do *not* upgrade to `tabpfn>=8`: that line added a Prior Labs license gate
  (`browser_auth.ensure_license_accepted` → `TabPFNLicenseError`) that demands
  `export TABPFN_TOKEN=...` (from <https://ux.priorlabs.ai/account>) for any
  non-interactive run. The pin keeps reproduction token-free.
- For best results, use datasets with <1000 samples and <100 features
- Supervised mode typically produces better embeddings for classification tasks
