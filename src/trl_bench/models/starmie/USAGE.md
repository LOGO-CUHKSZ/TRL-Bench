# Starmie

## Overview

Starmie generates column-level embeddings using contrastive learning on table columns. It's designed for data discovery tasks like join search and union search.

Reference: "Starmie: Semantics-aware Dataset Discovery from Data Lakes with Contextualized Column Embeddings" (Fan et al., VLDB 2023)

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `column_embeddings` | `{col_idx: (dim,)}` | One embedding per column |
| `table_embedding` | `dict` | Table-level embeddings (v2.0 format) |

**Table embedding structure (v2.0):**
```python
'table_embedding': {
    'cls_embedding': None,        # Not supported by Starmie
    'table_embedding': None,      # No native table embedding
    'column_mean': array(dim,),   # Mean-pooled column embeddings
}
```

## Output Formats

### Unified v2.0 Format (Recommended)

Use `generate_column_embeddings.py` for compatibility with all downstream tasks:

```python
[
    {
        'table': 'path/to/table.csv',
        'table_id': 'table_name',
        'column_embeddings': {0: array, 1: array, ...},
        'column_names': ['col1', 'col2', ...],
        'table_embedding': {
            'cls_embedding': None,
            'table_embedding': None,
            'column_mean': array,
        }
    },
    ...
]
```

### Legacy Formats

For backward compatibility with existing pipelines:

#### extractVectors.py (Union Search Format)

```python
[
    ('table1.csv', np.array([[0.1, 0.2, ...],    # col1 vector
                             [0.3, 0.4, ...]])), # col2 vector
    ...
]
```
Format: `[(table_name, column_vectors_matrix), ...]` where matrix is shape `(num_cols, dim)`

#### extractColumnVectors.py (Join Search Format)

```python
[
    ('table1.csv', 'column_name', array([0.1, 0.2, ...])),
    ('table1.csv', 'another_col', array([0.3, 0.4, ...])),
    ...
]
```
Format: `[(table_name, column_name, embedding), ...]`

## Model Type

| Property | Value |
|----------|-------|
| **Training Required** | Yes |
| **Embedding Dimension** | Configurable |

Starmie must be trained on a data lake before generating embeddings.

## Input Data

**Expected directory structure:**
```
input_dir/
├── table1.csv
├── table2.csv
└── ...
```

**CSV requirements:**
- First row: column headers
- Supports text and numeric columns

## Example Commands

### Step 1: Train the Model

```bash
python models/starmie/run_pretrain.py \
    --data_path datasets/santos/datalake \
    --checkpoint_dir checkpoints/starmie/santos \
    --save_model
```

### Step 2: Generate Embeddings (Unified v2.0 Format)

```bash
python models/starmie/generate_column_embeddings.py \
    --model_path checkpoints/starmie/santos/model.pt \
    --input_dir datasets/santos/datalake \
    --output_path embeddings/starmie/santos_unified.pkl
```

### Step 3: Convert to Legacy Format (If Needed)

```bash
# Convert to union search format
python utils/convert_unified_to_starmie.py \
    --input embeddings/starmie/santos_unified.pkl \
    --output embeddings/union_search/starmie/santos.pkl \
    --format union_search

# Convert to join search format
python utils/convert_unified_to_starmie.py \
    --input embeddings/starmie/santos_unified.pkl \
    --output embeddings/join_search/starmie/santos.pkl \
    --format join_search

# Convert to both formats at once
python utils/convert_unified_to_starmie.py \
    --input embeddings/starmie/santos_unified.pkl \
    --output_union embeddings/union_search/starmie/santos.pkl \
    --output_join embeddings/join_search/starmie/santos.pkl \
    --format both
```

### Legacy Commands (Direct Legacy Format Output)

```bash
# Union search format (direct)
python models/starmie/extractVectors.py \
    --model_path checkpoints/starmie/santos/model.pt \
    --input_dir datasets/santos/datalake \
    --output_path embeddings/union_search/starmie/santos.pkl

# Join search format (direct)
python models/starmie/extractColumnVectors.py \
    --model_path checkpoints/starmie/santos/model.pt \
    --input_dir datasets/santos/datalake \
    --output_path embeddings/join_search/starmie/santos.pkl
```

## Downstream Tasks

- Join search (column matching)
- Union search (table unionability)
- Data discovery
- Schema matching

## Related

- Conversion script: `utils/convert_unified_to_starmie.py`
- Union search task: `downstream_tasks/union_search/USAGE.md`
- Join search task: `downstream_tasks/join_search/USAGE.md`
