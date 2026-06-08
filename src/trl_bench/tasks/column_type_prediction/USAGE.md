# Column Type Prediction

## Overview

Single-label classification task to predict semantic types for table columns. Uses frozen column embeddings to classify each column into one semantic type.

Reference: "Doduo: Annotating Tables with Types and Relationships" (Suhara et al., VLDB 2022)

## Input Data

> **Embedding Level:** Column
> **Pair Input:** No

Two inputs are required:

1. **Unified column embeddings** (`.pkl`): List of dicts with `table_id` and `column_embeddings`
2. **Labels CSV** (`train.csv` / `test.csv`): Rows of `(table_id, column_id, class)`

**Embeddings format** (unified v2.0):
```python
[
    {'table_id': 'table_0', 'column_embeddings': {0: [...], 1: [...], ...}},
    {'table_id': 'table_1', 'column_embeddings': {0: [...], 1: [...], ...}},
    ...
]
```

**Labels CSV format:**
```csv
table_id,column_id,class
0,0,address
0,1,boolean
1,0,city
...
```

**Directory layout:**
```
embeddings/column/<model>/<dataset>.pkl   # Unified embeddings
datasets/<dataset>/
├── train.csv                                    # Training labels
└── test.csv                                     # Test labels
```

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Single-label Classification |
| **Embedding Level** | Column |
| **Pair Input** | No |
| **Num Types** | Dataset-dependent (e.g., 78 for SATO, 255 for WikiCT) |

## Evaluation Metrics

| Metric | Primary | Description |
|--------|---------|-------------|
| MAP | Yes | Mean Average Precision |
| Micro F1 | Yes | Micro-averaged F1 score |
| Macro F1 | Yes | Macro-averaged F1 score |

## Example Commands

### Basic Training

```bash
python downstream_tasks/column_type_prediction/train_ct_mode4.py \
    --embeddings embeddings/column/bert/sato.pkl \
    --dataset datasets/sato \
    --output_dir checkpoints/column_type/bert_sato \
    --num_epochs 10 \
    --learning_rate 5e-4
```

### With Custom Hyperparameters

```bash
python downstream_tasks/column_type_prediction/train_ct_mode4.py \
    --embeddings embeddings/column/bert/sato.pkl \
    --dataset datasets/sato \
    --output_dir checkpoints/column_type/bert_sato_custom \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_epochs 5 \
    --dropout 0.2 \
    --device cuda
```

### Evaluation Only

```bash
python downstream_tasks/column_type_prediction/evaluate_ct_mode4.py \
    --classifier_path checkpoints/column_type/bert_sato/best_model.pt \
    --embeddings embeddings/column/bert/sato.pkl \
    --test_csv datasets/sato/test.csv \
    --device cuda
```

## Arguments

### train_ct_mode4.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--embeddings` | **Yes** | - | Path to unified column embeddings .pkl file |
| `--dataset` | **Yes** | - | Dataset root (must contain `train.csv` and `test.csv`) |
| `--output_dir` | No | column_type_classifier | Output directory |
| `--batch_size` | No | 20 | Training batch size |
| `--learning_rate` | No | From YAML config | Learning rate |
| `--num_epochs` | No | 2 | Training epochs |
| `--warmup_steps` | No | 100 | LR scheduler warmup steps |
| `--dropout` | No | 0.1 | Dropout rate |
| `--device` | No | cuda/cpu | Device to use |
| `--seed` | No | 42 | Random seed |
| `--wandb_project` | No | column-type-prediction | W&B project name |
| `--wandb_run_name` | No | None | W&B run name |

### evaluate_ct_mode4.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--classifier_path` | **Yes** | - | Path to trained classifier checkpoint |
| `--embeddings` | **Yes** | - | Path to unified column embeddings .pkl file |
| `--test_csv` | **Yes** | - | Path to test labels CSV file |
| `--batch_size` | No | 20 | Evaluation batch size |
| `--device` | No | cuda/cpu | Device to use |

## Model Architecture

Unified MLP head on top of frozen embeddings:

```python
MLPHead (unified architecture):
    dropout(0.1)                        # dropout_first
    linear(hidden_size -> hidden_dim)   # 312 -> 256
    relu()
    dropout(0.1)
    linear(hidden_dim -> num_types)     # 256 -> 255
    BCEWithLogitsLoss
```

## Output

Results saved to `{output_dir}/`:
- `best_model.pt`: Best model checkpoint
- `checkpoint_epoch_N.pt`: Epoch checkpoints
- `logs/`: Training logs

### Checkpoint format
```python
{
    'epoch': 5,
    'model_state_dict': ...,
    'best_map': 0.8234,
    'hidden_size': 312,
    'num_types': 255,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.1,
    'class_to_idx': {'address': 0, 'boolean': 1, ...}  # Required for evaluation
}
```

## Mode 4 Embeddings

Mode 4 (cell content-based) embeddings are generated using only cell values, without headers or metadata. This evaluates the model's ability to infer types purely from cell content.

**Generating Mode 4 embeddings:**
```bash
python models/doduo/generate_column_embeddings_dataset.py \
    --dataset wikict \
    --mode 4 \
    --output_dir embeddings/column_type/doduo
```

---

## Troubleshooting

### Embedding shape mismatch
```
RuntimeError: size mismatch
```
**Solution:** Check that `hidden_size` in model matches embedding dimension.

### Low MAP score
- Increase `--num_epochs`
- Try different `--learning_rate`
- Verify embeddings are Mode 4 (cell content only)

### Missing embeddings file
```
FileNotFoundError: train_embeddings.pkl
```
**Solution:** Generate embeddings first using the Doduo pipeline.

## Related

- Embedding generation: `models/doduo/USAGE.md`
- Similar tasks: `downstream_tasks/column_relation_prediction/`
