# TURL (Table Understanding through Representation Learning)

## Overview

In this repo, TURL is used as a **frozen column-embedding model** through
`models/turl/generate_column_embeddings_dataset.py`.

Important repo-specific detail: this is **not** the full original TURL
metadata+entity-linking pipeline. The benchmark extraction path uses a
**cell-content-only, mode-4-style configuration**:

- every CSV cell is treated as entity-text input
- the token stream is disabled
- pretrained KB entity-ID inputs are disabled
- row/column structure is preserved through the entity attention mask

Reference: Xiang Deng et al., *TURL: Table Understanding through Representation
Learning* (PVLDB 2021)

## Embeddings Generated

| Embedding | Shape | Description |
|-----------|-------|-------------|
| `column_embeddings` | `{col_idx: (312,)}` | One 312-d embedding per column |
| `table_embedding` | `dict` | Derived table-level variants |

**Table embedding structure (v2.0):**
```python
'table_embedding': {
    'cls_embedding': None,         # Not produced by this TURL path
    'table_embedding': None,       # No native table embedding exposed
    'column_mean': array(312,),    # Mean of column embeddings
}
```

**Output format:** pickle file (`.pkl`) containing a list of dicts:
```python
[
    {
        'table_id': 'table_name',
        'table': 'path/to/table.csv',
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

## Model / Checkpoint

| Property | Value |
|----------|-------|
| **Training Required** | No |
| **Checkpoint Type** | Local pretrained checkpoint directory |
| **Embedding Dimension** | 312 |
| **Active Checkpoint Config** | 4 layers, hidden size 312, 12 heads |

Expected checkpoint layout:
```text
checkpoints/turl/pretrained/
├── config.json
└── pytorch_model.bin
```

The checked-in checkpoint in this repo is **312-dimensional**, not 768-dimensional.

## Input Data

**Expected directory structure:**
```text
input_dir/
├── table1.csv
├── table2.csv
└── ...
```

**CSV requirements:**
- first row must be the header row
- remaining rows are treated as table content
- text and numeric cells are both accepted

## Script Defaults

The script defaults are:

- `--mode table_directory`
- `--max_rows 100`
- `--batch_size 16`
- `--max_cell_length 10`
- `--max_cell_chars 512`
- `--num_workers 0`
- `--checkpoint_interval 100`

In the Slurm pipeline, some of these defaults are overridden in
`slurm/config/models.yaml`:

- `num_workers` is typically set to `4`
- wide/large datasets may be forced to `batch_size = 1`
- `max_entities` is set to `12000`

## Example Commands

### Generate Column Embeddings

```bash
python models/turl/generate_column_embeddings_dataset.py \
    --mode table_directory \
    --input_dir datasets/santos/datalake \
    --output_file embeddings/column/turl/santos.pkl
```

### With Explicit Checkpoint

```bash
python models/turl/generate_column_embeddings_dataset.py \
    --mode table_directory \
    --input_dir datasets/santos/datalake \
    --output_file embeddings/column/turl/santos.pkl \
    --checkpoint checkpoints/turl/pretrained
```

### With Custom Processing Limits

```bash
python models/turl/generate_column_embeddings_dataset.py \
    --mode table_directory \
    --input_dir datasets/santos/datalake \
    --output_file embeddings/column/turl/santos.pkl \
    --max_rows 100 \
    --batch_size 8 \
    --max_entities 12000 \
    --max_cell_chars 512
```

### Sharded Run Support

```bash
python models/turl/generate_column_embeddings_dataset.py \
    --mode table_directory \
    --input_dir datasets/opendata/tables \
    --output_file embeddings/column/turl/opendata_shard0.pkl \
    --table_list slurm/scripts/generated/table_lists/opendata_shard0of5.txt
```

## Table-Level Use

TURL is **not** a native table-embedding generator in this benchmark.
Standalone table embedding pkls are produced later from the column pkls by:

```bash
python scripts/generate_table_embeddings.py --models turl
```

For TURL, the only supported table variant is:

- `column_mean`

## Current Benchmark Use

TURL is currently used for:

- column-level tasks such as join search, join containment, column clustering,
  column type prediction, column relation prediction, union search, schema
  matching, semantic parsing, and DLTE column stages
- table-level tasks only through the derived `column_mean` variant

TURL is **not** used as a row-embedding model in this repo.
