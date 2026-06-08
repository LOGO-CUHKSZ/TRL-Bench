# Column Relation Prediction

## Overview

Multi-label classification task to predict relationships between column pairs within a table. Given column embeddings, predicts which Wikidata relations hold between columns (e.g., "country" - "capital" relation).

Reference: "Doduo: Annotating Tables with Types and Relationships" (Suhara et al., VLDB 2022)

## Embeddings Consumed

> **Embedding Level:** Column pairs
> **Primary Embedding:** Concatenated column embeddings
> **Pair Input:** Yes (column pairs within same table)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `embeddings` | **Yes** | `(num_cols, dim)` per table | Column embeddings matrix | Doduo, TURL (requires specific mode) |
| `table_ids` | **Yes** | List | Table identifiers | - |

**Input format (unified, preferred):** Single pickle file (`.pkl`) containing a list of dicts:
```python
[
    {
        'table_id': 'table_000001_27282378-2',
        'column_embeddings': {0: array(...), 1: array(...), ...}
    },
    ...
]
```
The `table_NNNNNN_` prefix is automatically stripped to match metadata IDs. Train/test split is determined by the dataset metadata files.

**Input format (legacy):** Separate train/test pickle files containing:
```python
{
    'embeddings': [array1, array2, ...],  # List of (num_cols, hidden_dim) arrays
    'table_ids': [id1, id2, ...]          # List of table identifiers
}
```

**Metadata format:** JSON file with relation annotations:
```json
[
    {
        "table_id": "table_0",
        "relation_annotations": [
            {
                "column_id": 0,
                "relation_ids": [0, 0, 1, 0, ...]  # Multi-hot vector (121 relations)
            },
            ...
        ]
    },
    ...
]
```

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Multi-label Classification |
| **Embedding Level** | Column pairs |
| **Pair Input** | Yes |
| **Num Relations** | 121 (WikiCT dataset) |

## Evaluation Metrics

| Metric | Primary | Description |
|--------|---------|-------------|
| Micro F1 | Yes | Micro-averaged F1 score |
| Macro F1 | No | Macro-averaged F1 score |

## Input Data

**Unified embedding file (preferred):**
```
embeddings/column/<model>/WikiCT_relation.pkl
```

**Legacy split embeddings:**
```
embeddings/column/<model>/WikiCT_relation_split/
├── train_embeddings.pkl
└── test_embeddings.pkl
```

**Dataset directory:**
```
datasets/WikiCT_relation/
├── train/
│   └── train_metadata.json
└── test/
    └── test_metadata.json
```

## Example Commands

### Unified Embedding File (Preferred)

```bash
python downstream_tasks/column_relation_prediction/csv_relation_pipeline.py \
    --embeddings_file embeddings/column/starmie/WikiCT_relation.pkl \
    --dataset_dir datasets/WikiCT_relation \
    --output_dir checkpoints/relation/starmie \
    --epochs 20
```

### Legacy Split Embeddings

```bash
python downstream_tasks/column_relation_prediction/csv_relation_pipeline.py \
    --embeddings_dir embeddings/column/starmie/WikiCT_relation_split \
    --dataset_dir datasets/WikiCT_relation \
    --output_dir checkpoints/relation/starmie \
    --epochs 20
```

### All Pairwise Combinations

```bash
python downstream_tasks/column_relation_prediction/csv_relation_pipeline.py \
    --embeddings_file embeddings/column/turl/WikiCT_relation.pkl \
    --dataset_dir datasets/WikiCT_relation \
    --output_dir checkpoints/relation/turl_allpairs \
    --epochs 20 \
    --use_all_pairs
```

### With Custom Hyperparameters

```bash
python downstream_tasks/column_relation_prediction/csv_relation_pipeline.py \
    --embeddings_file embeddings/column/doduo/WikiCT_relation.pkl \
    --dataset_dir datasets/WikiCT_relation \
    --output_dir checkpoints/relation/doduo_custom \
    --epochs 30 \
    --batch_size 64 \
    --lr 5e-4 \
    --hidden_dim 256 \
    --dropout 0.2
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--embeddings_file` | One of these | - | Single unified `.pkl` embedding file (preferred) |
| `--embeddings_dir` | required | - | Directory containing split embedding files (legacy) |
| `--dataset_dir` | Yes | - | Directory containing dataset metadata |
| `--output_dir` | No | output | Output directory for models |
| `--batch_size` | No | 32 | Training batch size |
| `--epochs` | No | 20 | Training epochs |
| `--lr` | No | 1e-3 | Learning rate |
| `--hidden_dim` | No | 256 | Hidden layer dimension |
| `--dropout` | No | 0.1 | Dropout rate |
| `--max_tables` | No | None | Limit tables for debugging |
| `--use_all_pairs` | No | False | Use all pairwise combinations |
| `--device` | No | cuda/cpu | Device to use |

## Pairing Strategies

### First-Column Pairs (Default - Doduo style)
Pairs the first column with all other columns in each table:
```
Table with columns [A, B, C, D]:
  Pairs: (A,B), (A,C), (A,D)
```
This assumes the first column is often a key/entity column.

### All Pairwise Combinations (`--use_all_pairs`)
Creates all unique column pairs:
```
Table with columns [A, B, C, D]:
  Pairs: (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)
```
More pairs but may include less meaningful combinations.

## Model Architecture

```python
MLPHead (unified architecture):
    linear(input_dim -> hidden_dim)  # 1536 -> 256
    relu()
    dropout(0.1)
    linear(hidden_dim -> num_relations)  # 256 -> 121
```

Input is concatenation of two column embeddings: `[col_i; col_j]`

## Output

Model saved to `{output_dir}/best_model.pt`:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'epoch': 15,
    'best_f1': 0.6234,
    'args': {...},
    'head_config': {
        'input_dim': 1536, 'output_dim': 121,
        'hidden_dim': 256, 'num_layers': 2,
        'activation': 'relu', 'dropout': 0.1
    }
}
```

### Training output
```
============================================================
Epoch 20/20
============================================================
Training: 100%|███████| 9421/9421 [02:34<00:00, loss=0.0234]
  train_loss=0.0234  train_micro_f1=0.6523
  -> New best model (epoch 20, train_micro_f1=0.6523)
Loaded best model from output/best_model.pt
Final test:  test_loss=0.0312  test_macro_f1=0.4123  test_micro_f1=0.6234
```

---

## Troubleshooting

### Column ID out of bounds
```
Warning: table_id col_id X >= Y embeddings
```
**Solution:** Ensure embeddings cover all annotated columns.

### Table not found in embeddings
```
Skipped X tables (not in embeddings)
```
**Solution:** Verify table IDs match between embeddings and metadata.

### Low F1 score
- Try `--use_all_pairs` for more training pairs
- Increase `--hidden_dim`
- Increase `--epochs`

### Missing files
```
Error: Missing required files
```
**Solution:** Ensure either `--embeddings_file` points to a valid `.pkl` file, or `--embeddings_dir` contains `train_embeddings.pkl` and `test_embeddings.pkl`. The dataset directory must contain `train/train_metadata.json` and `test/test_metadata.json`.

## Related

- Embedding generation: `models/doduo/USAGE.md`, `models/turl/USAGE.md`
- Similar tasks: `downstream_tasks/column_type_prediction/`
