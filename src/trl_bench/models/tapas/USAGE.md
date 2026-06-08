# TAPAS (Table Parser)

## Overview

TAPAS generates column-level embeddings using a BERT-based model designed for table understanding. It can encode tables with or without question context.

Reference: "TAPAS: Weakly Supervised Table Parsing via Pre-training" (Herzig et al., ACL 2020)

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `column_embeddings` | `{col_idx: (768,)}` | One embedding per column |
| `table_embedding` | `dict` | Table-level embeddings (v2.0 format) |

**Table embedding structure (v2.0):**
```python
'table_embedding': {
    'cls_embedding': array(768,),  # [CLS] token represents table+context
    'table_embedding': None,       # No native table embedding
    'column_mean': array(768,),    # Mean-pooled column embeddings
    'token_mean': array(768,),     # Mean of all non-padding token hidden states
}
```

**Output format:** Pickle file (`.pkl`) containing list of dicts:
```python
[
    {
        'table_id': 'table_name',           # Canonical identifier (filename without extension)
        'table': 'path/to/table.csv',       # Full path to source file
        'table_name': 'table_name',         # Table name (filename without extension)
        'column_embeddings': {0: array, 1: array, ...},
        'table_embedding': {
            'cls_embedding': array,
            'table_embedding': None,
            'column_mean': array,
            'token_mean': array,
        },
        'column_names': ['col1', 'col2', ...],
        'model_name': 'tapas-base',
        'embedding_dim': 768
    },
    ...
]
```

## Model Type

| Property | Value |
|----------|-------|
| **Training Required** | No (pretrained) |
| **Embedding Dimension** | 768 (base) / 1024 (large) |

TAPAS uses HuggingFace pretrained models. Available variants:
- `google/tapas-base` (default)
- `google/tapas-large`
- Fine-tuned: `tapas-base-finetuned-wtq`, `tapas-base-finetuned-sqa`

## Input Data

**Expected:** Single CSV file or directory of CSV files:
```
input_dir/
├── table1.csv
├── table2.csv
└── ...
```

**CSV requirements:**
- First row: column headers
- Supports text and numeric columns
- Tables should fit within ~512 tokens

## Example Commands

### Single File

```bash
python models/tapas/generate_column_embeddings.py \
    --input /path/to/table.csv \
    --output embeddings.pkl
```

### Batch Mode (Directory)

```bash
python models/tapas/generate_column_embeddings.py \
    --input datasets/santos/datalake \
    --output embeddings/join_search/tapas/santos_datalake.pkl
```

### With Question Context

```bash
python models/tapas/generate_column_embeddings.py \
    --input /path/to/table.csv \
    --question "What is the total revenue?" \
    --output embeddings.pkl
```

### Use Large Model

```bash
python models/tapas/generate_column_embeddings.py \
    --input datasets/santos/datalake \
    --model google/tapas-large \
    --output embeddings/join_search/tapas/santos_large.pkl
```

## Downstream Tasks

- Join search (column matching)
- Table question answering
- Table fact verification
- Schema matching
