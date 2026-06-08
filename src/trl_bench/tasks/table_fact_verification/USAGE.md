# Table Fact Verification

## Overview

Binary classification task to determine whether a natural language statement is entailed or refuted by a given table. Also known as the TabFact benchmark task.

Reference: "TabFact: A Large-scale Dataset for Table-based Fact Verification" (Chen et al., ICLR 2020)

## Embeddings Consumed

> **Embedding Level:** Table + Statement
> **Primary Embedding:** `table_embeddings`, `statement_embeddings` (optional)
> **Pair Input:** Yes (table-statement pairs)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `table_embeddings` | **Yes** | `{example_id: (dim,)}` | Table embedding per example | TAPAS, TaBERT |
| `statement_embeddings` | Optional | `{example_id: (dim,)}` | Statement embedding per example | BERT |
| `labels` | **Yes** | `{example_id: int}` | 0=refuted, 1=entailed | - |

**Note:** Doduo does NOT produce table_embedding (sets to None). Use TAPAS or TaBERT which jointly encode table+statement.

**Input format:** Pickle file (`.pkl`) containing:

### Single-embedding mode (e.g., TAPAS joint encoding)
```python
{
    'table_embeddings': {example_id: np.array(768,), ...},
    'labels': {example_id: int, ...}
}
```

### Two-embedding mode (separate table and statement encoders)
```python
{
    'table_embeddings': {example_id: np.array(768,), ...},
    'statement_embeddings': {example_id: np.array(768,), ...},
    'labels': {example_id: int, ...}
}
```

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Binary Classification |
| **Embedding Level** | Table + Statement |
| **Pair Input** | Yes |
| **Labels** | 0=refuted, 1=entailed |

## Evaluation Metrics

| Metric | Primary | Description |
|--------|---------|-------------|
| Accuracy | Yes | Classification accuracy |
| F1 (Macro) | No | Macro-averaged F1 score |

## Input Data

**Embeddings:**
- `embeddings/tabfact/<model>/train.pkl`
- `embeddings/tabfact/<model>/validation.pkl`

**TabFact dataset:**
```
datasets/tabfact/
├── train/
├── validation/
└── test/
```

To download TabFact:
```bash
python downstream_tasks/table_fact_verification/download_tabfact.py
```

## Example Commands

### Single-Embedding Mode (TAPAS)

```bash
python downstream_tasks/table_fact_verification/train.py \
    --train_embeddings embeddings/tabfact/tapas/train.pkl \
    --val_embeddings embeddings/tabfact/tapas/validation.pkl \
    --output_dir checkpoints/tabfact/tapas \
    --device cuda
```

### Two-Embedding Mode with Concatenation (Doduo+BERT)

```bash
python downstream_tasks/table_fact_verification/train.py \
    --train_embeddings embeddings/tabfact/doduo/train.pkl \
    --val_embeddings embeddings/tabfact/doduo/validation.pkl \
    --output_dir checkpoints/tabfact/doduo_concat \
    --combine_method concat \
    --device cuda
```

### Two-Embedding Mode with Addition

```bash
python downstream_tasks/table_fact_verification/train.py \
    --train_embeddings embeddings/tabfact/doduo/train.pkl \
    --val_embeddings embeddings/tabfact/doduo/validation.pkl \
    --output_dir checkpoints/tabfact/doduo_add \
    --combine_method add \
    --device cuda
```

### Evaluation Only

```bash
python downstream_tasks/table_fact_verification/evaluate.py \
    --test_embeddings embeddings/tabfact/tapas/validation.pkl \
    --model_checkpoint checkpoints/tabfact/tapas/best_model.pt \
    --device cuda
```

## Arguments

### train.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--train_embeddings` | Yes | - | Path to training embeddings |
| `--val_embeddings` | No | - | Path to validation embeddings |
| `--output_dir` | No | checkpoints/tabfact | Output directory |
| `--model_type` | No | mlp | Model: linear or mlp |
| `--hidden_dim` | No | 256 | Hidden dimension for MLP |
| `--dropout` | No | 0.1 | Dropout rate |
| `--epochs` | No | 50 | Training epochs (overrides YAML when passed) |
| `--batch_size` | No | 32 | Batch size |
| `--lr` | No | 1e-3 | Learning rate |
| `--weight_decay` | No | 0 | Weight decay (overrides YAML when passed) |
| `--device` | No | cuda/cpu | Device to use |
| `--seed` | No | 42 | Random seed |
| `--combine_method` | No | None | concat or add (for two-embedding mode) |

## Embedding Modes

### Single-Embedding Mode
Used when the model jointly encodes table and statement (e.g., TAPAS).

- Input dimension: 768 (base) or 1024 (large)
- Only `table_embeddings` required
- `combine_method` is ignored

### Two-Embedding Mode
Used when table and statement are encoded separately (e.g., Doduo for table + BERT for statement).

- Requires both `table_embeddings` and `statement_embeddings`
- `combine_method=concat`: Input dimension = 2 x 768 = 1536
- `combine_method=add`: Input dimension = 768

## Output

Results saved to `{output_dir}/`:
- `best_model.pt`: Best model checkpoint
- `final_model.pt`: Final model checkpoint
- `config.json`: Training configuration
- `history.json`: Training history

### Example config.json
```json
{
    "train_embeddings": "embeddings/tabfact/tapas/train.pkl",
    "val_embeddings": "embeddings/tabfact/tapas/validation.pkl",
    "model_type": "mlp",
    "input_dim": 768,
    "hidden_dim": 256,
    "dropout": 0.1,
    "epochs": 50,
    "combine_method": "concat",
    "best_epoch": 12,
    "best_val_loss": 0.5432
}
```

## Generating Embeddings

### For TAPAS (joint encoding)
```bash
python downstream_tasks/table_fact_verification/generate_embeddings.py \
    --model tapas \
    --data_dir datasets/tabfact \
    --output_dir embeddings/tabfact/tapas \
    --split train
```

### For Doduo+BERT (separate encoding)
```bash
python downstream_tasks/table_fact_verification/generate_embeddings.py \
    --model doduo_bert \
    --data_dir datasets/tabfact \
    --output_dir embeddings/tabfact/doduo \
    --split train
```

---

## Troubleshooting

### Missing statement_embeddings
```
KeyError: 'statement_embeddings'
```
**Solution:** For single-embedding mode, ensure the model jointly encodes table+statement. For two-embedding mode, generate statement embeddings separately.

### Low accuracy
- Try `--combine_method concat` for two-embedding mode
- Increase `--hidden_dim` (e.g., 512)
- Increase `--epochs` (e.g., 20)
- Try MLP model instead of linear

### Class imbalance
TabFact has roughly balanced classes (~50% entailed, ~50% refuted). If your embeddings show significant imbalance, check embedding generation pipeline.

## Related

- Embedding generation: `models/tapas/USAGE.md`, `models/doduo/USAGE.md`
- Similar tasks: `downstream_tasks/semantic_parsing/`
