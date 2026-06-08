# Union Search

## Overview

Finds unionable tables in a data lake - tables that share the same schema and can be combined via UNION operation. Uses column embeddings for bipartite matching to compute union similarity scores.

Reference: "Starmie: Data Discovery with Column Annotations" (Fan et al., VLDB 2023)

## Embeddings Consumed

> **Embedding Level:** Column (table-level for classification/regression subtasks)
> **Primary Embedding:** Column tuples for search, table embeddings for subtasks
> **Pair Input:** No (search) / Yes (classification/regression)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| Column embeddings | **Yes** | `(table_name, col_embeddings)` | Per-table column matrix | Starmie (extractVectors.py) |
| `table_embedding` | Subtasks | `dict (v2.0)` | Table-level embeddings | TabSketchFM, TaBERT, TAPAS, TURL, Doduo |

**Note:** Search task uses Starmie's tuple format. Subtasks (classification/regression) use dict format from TabSketchFM/TaBERT/TAPAS/Doduo/TURL.

**Format Conversion:** To use embeddings from TaBERT/TAPAS/TabSketchFM/Doduo/TURL (unified v2.0 format), convert them first:
```bash
python utils/convert_unified_to_starmie.py \
    --input embeddings/union_search/tapas/query_embeddings.pkl \
    --output embeddings/union_search/tapas/query_starmie.pkl \
    --format union_search
```

**Search input format:** Pickle file (`.pkl`) containing:
```python
[
    (table_name, embeddings_array),  # embeddings_array: (num_cols, dim)
    ...
]
```

**Subtask input format (v2.0):**
```python
[
    {
        'table': 'table1.csv',
        'column_embeddings': {0: [...], 1: [...]},
        'table_embedding': {
            'cls_embedding': array or None,
            'table_embedding': None,
            'column_mean': array,
        }
    },
    ...
]
```

**Helper:** Use `get_table_level_embedding(item, variant='column_mean')` to extract the desired variant.

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Ranking (search) / Classification / Regression |
| **Embedding Level** | Column (search) / Table (subtasks) |
| **Pair Input** | No (search) / Yes (subtasks) |

## Evaluation Metrics

### Search Task
| Metric | Primary | Description |
|--------|---------|-------------|
| MAP@K | Yes | Mean Average Precision at K |
| P@K | No | Precision at K |
| R@K | No | Recall at K |

### Classification Subtask
| Metric | Primary | Description |
|--------|---------|-------------|
| Accuracy | Yes | Classification accuracy |
| F1 | No | F1 score |

### Regression Subtask
| Metric | Primary | Description |
|--------|---------|-------------|
| R² | Yes | Coefficient of determination |
| MSE | No | Mean squared error |

## Input Data

**Query embeddings:** `embeddings/union_search/<model>/query.pkl`

**Datalake embeddings:** `embeddings/union_search/<model>/datalake.pkl`

**Ground truth:** `datasets/santos/santosUnionBenchmark.pickle`

## Example Commands

### Linear Search (Exact)

```bash
python downstream_tasks/union_search/run_search.py \
    --query_embeddings embeddings/union_search/starmie/query.pkl \
    --datalake_embeddings embeddings/union_search/starmie/datalake.pkl \
    --groundtruth datasets/santos/santosUnionBenchmark.pickle \
    --method linear \
    --K 10
```

### HNSW Search (Approximate, Fast)

```bash
python downstream_tasks/union_search/run_search.py \
    --query_embeddings embeddings/union_search/starmie/query.pkl \
    --datalake_embeddings embeddings/union_search/starmie/datalake.pkl \
    --groundtruth datasets/santos/santosUnionBenchmark.pickle \
    --method hnsw \
    --ef 100 \
    --N 100 \
    --K 10
```

### Without Ground Truth (Search Only)

```bash
python downstream_tasks/union_search/run_search.py \
    --query_embeddings query.pkl \
    --datalake_embeddings datalake.pkl \
    --method hnsw \
    --K 20
```

## Arguments

### run_search.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--query_embeddings` | Yes | - | Path to query embeddings |
| `--datalake_embeddings` | Yes | - | Path to datalake embeddings |
| `--groundtruth` | No | None | Path to ground truth (for evaluation) |
| `--method` | No | linear | Search method: linear or hnsw |
| `--K` | No | 10 | Number of top results |
| `--threshold` | No | 0.7 | Column similarity threshold |
| `--ef` | No | 100 | HNSW search quality parameter |
| `--N` | No | 100 | HNSW columns per query column |

## Search Algorithm

Uses bipartite matching to compute union similarity:

1. **Build Index:** HNSW index on all datalake columns (or linear scan)
2. **Candidate Retrieval:** For each query column, find N nearest datalake columns
3. **Table Candidates:** Identify tables containing matched columns
4. **Bipartite Matching:** Hungarian algorithm to find optimal column alignment
5. **Score & Rank:** Sum matching scores, return top-K tables

## Output

### Search Results
```
Results:
  MAP@10  = 0.7234
  P@10    = 0.6500
  R@10    = 0.5800

Detailed Metrics:
  P@ 1 = 0.8200, R@ 1 = 0.1200
  P@ 5 = 0.7400, R@ 5 = 0.4500
  P@10 = 0.6500, R@10 = 0.5800
```

---

## Subtasks

### Subtask: Classification

Binary classification for union prediction.

**Location:** `downstream_tasks/union_search/classification/`

**Task:** Predict if two tables are unionable (same schema)

**Dataset:** wiki_union (301K train, 37K test pairs)

```bash
bash downstream_tasks/union_search/classification/run_classification.sh \
    --embeddings embeddings/union_search/wiki_union_embeddings.pkl \
    --labels datasets/wiki_union/labels.json \
    --output_dir results/union_classification
```

See `downstream_tasks/union_search/classification/README.md` for details.

### Subtask: Regression

Regression for union similarity prediction.

**Location:** `downstream_tasks/union_search/regression/`

**Task:** Predict continuous similarity/difference between table pairs

**Primary Dataset:** ECB-Union (dimension difference, labels 1-12)

```bash
bash downstream_tasks/union_search/regression/run_regression.sh \
    --embeddings embeddings/union_search/ecb_union_embeddings.pkl \
    --labels datasets/ecb_union/labels.json \
    --output_dir results/union_regression
```

See `downstream_tasks/union_search/regression/README.md` for details.

---

## Troubleshooting

### Low MAP score
- Try HNSW with higher `--N` and `--ef`
- Check embedding quality
- Try different `--threshold` values

### Slow linear search
- Use `--method hnsw` for large datalakes
- HNSW is approximate but much faster

### Missing ground truth
```
No ground truth provided. Skipping evaluation.
```
For evaluation, provide `--groundtruth` with pickle file mapping query names to lists of relevant table names.

### Memory issues
- Use `--method hnsw` instead of linear
- Reduce datalake size or batch processing

## Related

- Embedding generation: `models/starmie/USAGE.md`
- Similar tasks: `downstream_tasks/join_search/`
- Classification subtask: `downstream_tasks/union_search/classification/README.md`
- Regression subtask: `downstream_tasks/union_search/regression/README.md`
