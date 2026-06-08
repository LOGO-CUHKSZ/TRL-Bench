# TabSketchFM

## Overview

TabSketchFM generates column-level embeddings using a BERT-based model trained on table similarity tasks. It produces embeddings for columns, tables, and CLS representations.

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `column_embeddings` | `{col_idx: (768,)}` | One embedding per column |
| `table_embedding` | `dict` | Table-level embeddings (v2.0 format) |

**Table embedding structure (v2.0):**
```python
'table_embedding': {
    'cls_embedding': array(768,),  # [CLS] token embedding
    'table_embedding': None,       # No native table embedding
    'column_mean': array(768,),    # Mean-pooled column embeddings
    'token_mean': array(768,),     # Mean of non-special content tokens
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
        'column_names': ['col1', 'col2', ...]
    },
    ...
]
```

## Model Type

| Property | Value |
|----------|-------|
| **Training Required** | No (pretrained) |
| **Embedding Dimension** | 768 |

TabSketchFM uses pretrained checkpoints. Fine-tuned classifier checkpoints are NOT supported for embedding extraction.

**Checkpoint required:** Download `.ckpt` file from TabSketchFM repository.

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
- Tables are serialized column-by-column

## Example Commands

### Single File

```bash
python models/tabsketchfm/generate_column_embeddings.py \
    --input /path/to/table.csv \
    --checkpoint checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
    --output embeddings.pkl
```

### Batch Mode (Directory)

```bash
python models/tabsketchfm/generate_column_embeddings.py \
    --input datasets/santos/datalake \
    --checkpoint checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
    --output embeddings/join_search/tabsketchfm/santos_datalake.pkl
```

### Optimized Batch Extraction

For large-scale processing with GPU parallelization:

```bash
python models/tabsketchfm/scripts/embedding_extraction/extract_embeddings_unified_optimized.py \
    --checkpoint checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
    --data_dir datasets/santos/datalake \
    --output_file embeddings/join_search/tabsketchfm/santos_datalake.pkl \
    --batch_size 32 \
    --num_workers 4
```

## Downstream Tasks

- Join search (column matching)
- Union search
- Table similarity
- Schema matching
