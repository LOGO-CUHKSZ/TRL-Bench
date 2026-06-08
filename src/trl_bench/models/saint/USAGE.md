# SAINT: Self-Attention and INtersample attention Transformer

Row-level embedding model based on [arXiv:2106.01342](https://arxiv.org/abs/2106.01342).

## Overview

SAINT produces row embeddings via a `[CLS]` token using two types of attention:
- **Self-attention** across features (columns) within each row
- **Intersample attention** across rows in a batch (the "transpose trick")

Self-supervised pre-training combines:
- **CutMix + Mixup** augmentation in embedding space
- **Contrastive loss** (NTXent) between original and augmented views
- **Denoising loss** (feature reconstruction) from the augmented view

## Variants

| Variant | Self-Attention | Intersample Attention | Batch-Dependent |
|---------|:-:|:-:|:-:|
| `saint` (default) | Yes | Yes | Yes |
| `saint_s` | Yes | No | No |
| `saint_i` | No | Yes | Yes |

## Batch-Dependency Contract

**IMPORTANT**: The full `saint` and `saint_i` variants use intersample attention,
which makes row embeddings **batch-dependent**: a row's embedding depends on which
other rows share its batch.

- **Directory mode**: All rows processed in deterministic sequential chunks via
  `SequentialSampler`. Embeddings are **reproducible** for the same `batch_size`
  and row ordering, but **will differ** if `batch_size` changes or rows are reordered.
- **Data mode**: Same per-split — all rows processed sequentially with `SequentialSampler`.
- **`saint_s` variant**: No intersample attention — embeddings are fully
  batch-independent and deterministic regardless of batch size.

## Directory Mode (Batch Processing)

Processes a directory of CSV files, training one model per table:

```bash
python models/saint/generate_embeddings_directory.py \
    --input_dir /path/to/csvs \
    --output_path /path/to/output/saint.pkl \
    --checkpoint_base_dir /path/to/checkpoints \
    --phase1_epochs 20 \
    --batch_size 128 \
    --emb_dim 32 \
    --encoder_depth 6 \
    --n_head 8 \
    --saint_variant saint
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--emb_dim` | 32 | Per-feature embedding dimension |
| `--encoder_depth` | 6 | Number of SAINTBlocks |
| `--n_head` | 8 | Attention heads |
| `--ffn_factor` | 4.0 | FFN hidden multiplier |
| `--saint_variant` | saint | saint / saint_s / saint_i |
| `--cutmix_probability` | 0.3 | Feature swap probability |
| `--mixup_alpha` | 0.2 | Mixup Beta distribution alpha |
| `--tau` | 0.7 | NTXent temperature |
| `--lambda_denoise` | 10.0 | Denoising loss weight |
| `--pretraining_head_dim` | 256 | Projection head dimension |
| `--dropout_rate` | 0.0 | Dropout rate |

## Data Mode (Single Dataset)

### Training

```bash
python models/saint/train_saint.py \
    --data_dir datasets/openml/openml_3 \
    --checkpoint_dir /path/to/checkpoints \
    --label_policy manifest \
    --phase1_epochs 20
```

### Embedding Generation

```bash
python models/saint/generate_embeddings.py \
    --data_dir datasets/openml/openml_3 \
    --checkpoint_dir /path/to/checkpoints \
    --embedding_dir /path/to/embeddings
```

## Architecture

```
Input tensor (B, n_features)
        |
  FeatureTokenizer           (B, N+1, emb_dim)  [position 0 = CLS]
        |
  [Mixup in embedding space] [only during SSL training]
        |
  SAINTEncoder                L x SAINTBlock
   |-- SAINTBlock:
   |    |-- Self-Attention    nn.TransformerEncoderLayer on (B, N+1, d)
   |    |-- Intersample Attn  nn.TransformerEncoderLayer on (N+1, B, d)
        |
  CLS token x[:, 0]          (B, d) = row embedding
```

## Output

- **Directory mode**: Aggregate pickle with per-table entries
- **Data mode**: v2.0 format (metadata.json + per-split .npy files)
