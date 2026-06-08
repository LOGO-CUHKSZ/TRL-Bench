# TabICL Embedding Generator

This script generates embeddings using TabICL (Tabular In-Context Learner, ICML 2025) for both supervised and self-supervised learning scenarios.

## Features

- **Label-Free Embeddings**: Extracts representations from Stage 2, before label information enters the model
- **Fixed 512-dim Output**: 4 CLS tokens x 128 dimensions, regardless of input size
- **GPU Support**: Optimized for GPU with chunked test processing for large datasets
- **Compatible Output**: Generates the same output format as TabPFN and other row-level models

## Architecture

TabICL uses a three-stage architecture:

```
Input Features (N rows x D cols)
        │
        ▼
┌─────────────────┐
│  Stage 1:       │  Column embedding with SetTransformer
│  col_embedder   │  Maps raw features → column representations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2:       │  Row interaction with attention
│  row_interactor │  Captures row-row relationships → (N, 512)
└────────┬────────┘  ← EMBEDDINGS EXTRACTED HERE
         │
         ▼
┌─────────────────┐
│  Stage 3:       │  Label-dependent prediction
│  predictor      │  Uses labels for classification (NOT used)
└─────────────────┘
```

We extract embeddings after Stage 2 because:
1. **Label-free**: Labels only enter at Stage 3, so Stage 2 outputs are purely unsupervised
2. **Rich representations**: Captures both feature interactions (Stage 1) and data distribution (Stage 2)
3. **Fixed dimension**: Always 512-dim (4 CLS tokens x 128), independent of input features

## Installation

Make sure `tabicl` is installed in your environment:

```bash
pip install tabicl
```

## Preprocessing

TabICL includes its own preprocessing pipeline (accessed via `clf.X_encoder_`):
- Numerical features: standardized
- Categorical features: ordinal encoded
- Missing values: handled internally

The preprocessing is initialized during `fit()` and applied via `transform()`.

## Usage

### Supervised Mode

```bash
python models/TabICL/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --label_column income \
    --embedding_dir embeddings/row_prediction/TabICL \
    --mode supervised
```

### Self-Supervised Mode

```bash
python models/TabICL/generate_embeddings_train_test.py \
    --data_dir datasets/adult \
    --embedding_dir embeddings/row_prediction/TabICL \
    --mode self-supervised
```

## Arguments

- `--data_dir`: Directory containing `train.csv` and `test.csv`
- `--input`: Single CSV file (alternative to `--data_dir`)
- `--embedding_dir`: Output directory for embeddings (default: `embeddings/row_prediction/TabICL`)
- `--label_column`: Name of the label column (if None, uses all columns as features)
- `--mode`: Mode selection: `auto`, `supervised`, or `self-supervised` (default: `auto`)
- `--n_estimators`: Number of estimators (default: 1, see note below)
- `--checkpoint_version`: Model checkpoint (default: `tabicl-classifier-v1.1-0506.ckpt`)
- `--device`: Device to use: `auto`, `cuda`, or `cpu` (default: `auto`)

## Technical Details

- **Embedding Dimension**: 512 (fixed — 4 CLS tokens x 128 hidden dim)
- **Dummy Labels**: `fit()` requires >=2 classes; we create balanced dummy labels that don't affect Stage 2 output
- **n_estimators=1**: TabICL's ensemble shuffles feature columns with RoPE positional encodings. Averaging embeddings from different column orderings produces semantically incoherent results. Always use n_estimators=1 for embedding extraction.
- **Large Dataset Handling**: Test data is processed in chunks (5000 rows), each prepended with the full train set for context. Train set is always processed in full (col_embedder needs complete distribution).
- **GPU Memory**: Chunks are processed sequentially with explicit cache clearing between chunks

## Example Output

```
Summary:
  Mode: self-supervised
  Embedding dimension: 512 (4 CLS tokens x 128)
  Train embeddings: (32561, 512)
  Test embeddings: (16281, 512)
  Label column: None
```
