# TaBERT (Table BERT)

## Overview

TaBERT generates column-level embeddings using a BERT-based model that jointly encodes natural language context and structured table data. It uses content snapshots for efficient encoding.

Reference: "TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data" (Yin et al., ACL 2020)

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `column_embeddings` | `{col_idx: (768,)}` | One embedding per column |
| `table_embedding` | `dict` | Table-level embeddings (v2.0 format) |

**Table embedding structure (v2.0):**
```python
'table_embedding': {
    'cls_embedding': None,        # TaBERT doesn't have dedicated CLS
    'table_embedding': None,      # No native table embedding
    'column_mean': array(768,),   # Mean-pooled column embeddings
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
            'cls_embedding': None,
            'table_embedding': None,
            'column_mean': array,
        },
        'column_names': ['col1', 'col2', ...],
        'model_name': 'tabert_base_k3',
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

Available model variants:
- `tabert_base_k1` - Base model, 1 content snapshot
- `tabert_base_k3` - Base model, 3 content snapshots (recommended)
- `tabert_large_k1` - Large model, 1 content snapshot
- `tabert_large_k3` - Large model, 3 content snapshots

**Checkpoint required:** Download from TaBERT repository.

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
- Content snapshot sampling works best with diverse cell values

## Example Commands

### Single File

```bash
python models/tabert/generate_column_embeddings.py \
    --input /path/to/table.csv \
    --checkpoint checkpoints/tabert/tabert_base_k3/model.bin \
    --output embeddings.pkl
```

### Batch Mode (Directory)

```bash
python models/tabert/generate_column_embeddings.py \
    --input datasets/santos/datalake \
    --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
    --output embeddings/join_search/tabert/santos_datalake.pkl
```

### With Context (Question)

```bash
python models/tabert/generate_column_embeddings.py \
    --input /path/to/table.csv \
    --checkpoint checkpoints/tabert/tabert_base_k3/model.bin \
    --context "What is the population of Tokyo?" \
    --context_mode context \
    --output embeddings.pkl
```

### Header-Aware Mode

```bash
python models/tabert/generate_column_embeddings.py \
    --input /path/to/table.csv \
    --checkpoint checkpoints/tabert/tabert_base_k3/model.bin \
    --context_mode header \
    --output embeddings.pkl
```

## Downstream Tasks

- Join search (column matching)
- Semantic parsing (text-to-SQL)
- Table question answering
- Schema matching
