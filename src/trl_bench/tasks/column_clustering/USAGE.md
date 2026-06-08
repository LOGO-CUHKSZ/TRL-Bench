# Column Clustering

## Overview

Evaluates column embeddings by clustering columns based on embedding similarity and measuring cluster purity against ground truth semantic types. Implements the Starmie paper's clustering evaluation methodology.

Reference: "Starmie: Data Discovery with Column Annotations" (Fan et al., VLDB 2023)

## Embeddings Consumed

> **Embedding Level:** Column
> **Primary Embedding:** `(filename, embeddings_matrix)` tuples
> **Pair Input:** No

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| Column embeddings | **Yes** | `(N_cols, dim)` per table | Matrix of column embeddings | Starmie (extractVectors.py) |

**Input format:** Pickle file (`.pkl`) containing:
```python
[
    (filename, embeddings_array),  # ('table_123.csv', array of shape (num_cols, dim))
    ...
]
```

Example:
```python
[
    ('table_0.csv', np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]])),  # 2 columns
    ('table_1.csv', np.array([[0.5, 0.6, ...]])),  # 1 column
    ...
]
```

**Note:** This tuple format is produced by Starmie's `extractVectors.py`. Other models (TaBERT, TAPAS, TabSketchFM, Doduo, TURL) produce unified v2.0 dict format and require conversion:
```bash
python utils/convert_unified_to_starmie.py \
    --input embeddings/column/tapas/embeddings.pkl \
    --output embeddings/column/tapas/embeddings_starmie.pkl \
    --format union_search
```
The `union_search` format produces `[(filename, embeddings_array), ...]` tuples compatible with this task.

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Clustering Evaluation |
| **Embedding Level** | Column |
| **Pair Input** | No |
| **Algorithm** | Connected Components with size limit |

## Evaluation Metrics

| Metric | Primary | Description |
|--------|---------|-------------|
| Purity | No | Micro-average purity (weighted by cluster size) |
| NMI | Yes | Normalized Mutual Information (penalizes over/under-fragmentation) |
| ARI | No | Adjusted Rand Index (chance-corrected, can be negative) |
| Num Clusters | No | Number of clusters formed |
| Avg Cluster Size | No | Average size of clusters |

**Purity Calculation:**
```
Purity = (sum of dominant label counts) / (total columns)
```
This is the micro-average definition from the Starmie paper.

## Input Data

**Embeddings:** `embeddings/column/<model>/embeddings.pkl`

**Labels:** `{dataset}/all.csv`
Format: `table_id,column_id,class`

```csv
table_id,column_id,class
0,0,date
0,1,location
1,0,person_name
...
```

## Example Commands

### Basic Usage

```bash
python downstream_tasks/column_clustering/evaluate_clustering.py \
    --embeddings embeddings/column/starmie/embeddings.pkl \
    --dataset sato \
    --k 20 \
    --target_avg_size 50
```

### With Detailed Cluster Analysis

```bash
python downstream_tasks/column_clustering/evaluate_clustering.py \
    --embeddings embeddings/column/starmie/embeddings.pkl \
    --dataset sato \
    --k 20 \
    --target_avg_size 50 \
    --analyze
```

### Different K Value

```bash
python downstream_tasks/column_clustering/evaluate_clustering.py \
    --embeddings embeddings.pkl \
    --dataset sato \
    --k 50 \
    --target_avg_size 100
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--embeddings` | Yes | - | Path to embeddings pickle file |
| `--dataset` | Yes | - | Dataset name (currently only `sato` is supported) |
| `--k` | No | 20 | Number of nearest neighbors for similarity graph |
| `--target_avg_size` | No | 50 | Target average cluster size (tuned via binary search) |
| `--batch_size` | No | 4096 | Batch size for similarity computation |
| `--analyze` | No | False | Print detailed cluster analysis |

## Algorithm

The clustering algorithm from the Starmie paper:

1. **L2 Normalize:** Normalize column embeddings for cosine similarity
2. **Top-K Similarity:** Find k nearest neighbors for each column
3. **Connected Components:** Build graph and find connected components with max cluster size
4. **Tune Cluster Size:** Binary search to achieve target avg_cluster_size (~50)
5. **Compute Metrics:** Micro-average purity, NMI, and ARI

## Output

```
============================================================
RESULTS
============================================================
  Number of clusters:    234
  Avg cluster size:      50.23
  Purity:                0.8234 (82.34%)
  NMI:                   0.7521
  ARI:                   0.6843
  Total columns:         11,754
  Unique semantic types: 78
  k (nearest neighbors): 20
  Tuned max_cluster_size:523
============================================================
```

### With --analyze flag

```
============================================================
Top 10 largest clusters:
============================================================

Cluster 1 (size=98):
  date: 45 columns (45.9%)
  datetime: 32 columns (32.7%)
  time: 21 columns (21.4%)

Cluster 2 (size=87):
  location: 67 columns (77.0%)
  address: 20 columns (23.0%)
...
```

## Supported Datasets

| Dataset | Labels Path | Semantic Types |
|---------|-------------|----------------|
| sato | `sato/all.csv` | 78 types |

---

## Troubleshooting

### Missing table embeddings
```
⚠ Missing tables: 15
```
**Solution:** Ensure all tables referenced in all.csv have embeddings. Check table naming conventions (e.g., `table_123.csv`).

### Column mismatch
```
⚠ Column mismatches: 5
```
**Solution:** Ensure column IDs in all.csv match column indices in embedding arrays.

### Low purity
- Try increasing `--k` (more neighbors)
- Try different `--target_avg_size`
- Check embedding quality with different models

### Memory issues
```
MemoryError: Unable to allocate...
```
**Solution:** Reduce `--batch_size` for similarity computation.

## Related

- Embedding generation: `models/starmie/USAGE.md`
- Similar tasks: `downstream_tasks/column_type_prediction/`
