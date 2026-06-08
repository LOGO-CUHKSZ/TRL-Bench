# Join Search

## Overview

Finds joinable columns in a data lake by identifying columns that share similar values. Uses column embeddings with exact cosine similarity search to retrieve top-K joinable (candidate_table, candidate_column) pairs for each query column.

This is a **column-level embedding task**: it evaluates how well column embeddings can identify joinable column pairs across tables.

## Embeddings Consumed

> **Embedding Level:** Column
> **Primary Embedding:** `column_embeddings`
> **Pair Input:** No (similarity search)

| Embedding Type | Required | Shape | Description | Compatible Models |
|----------------|----------|-------|-------------|-------------------|
| `column_embeddings` | **Yes** | `{col_idx: (dim,)}` | Per-column embedding | TaBERT, TAPAS, TabSketchFM, Doduo, TURL, Starmie, BERT, GTE |

**Input format:** Pickle file (`.pkl`) containing either:

**Unified dict format (preferred):**
```python
[
    {
        'table': 'path/to/table.csv',
        'column_names': ['col_a', 'col_b', ...],
        'column_embeddings': {0: array, 1: array, ...}
    },
    ...
]
```

**Legacy tuple format (Starmie):**
```python
[(table_name, column_name, embedding), ...]
```

Both formats are automatically detected and handled. Table names are normalized to basenames. Column keys are coerced to strings for consistent matching with ground truth.

## Task Configuration

| Property | Value |
|----------|-------|
| **Task Type** | Ranking / Retrieval |
| **Embedding Level** | Column |
| **Evaluation Level** | Column-pair |
| **Search Method** | Exact cosine similarity (L2-normalized dot product) |
| **Self-Match Policy** | Entire query table excluded from candidates |
| **Pair Input** | No |

## Evaluation Metrics

| Metric | Primary | K Values | Description |
|--------|---------|----------|-------------|
| COL F1@K | Yes | 10, 20, 50 | Mean per-query F1 at top-K |
| COL Precision@K | No | 10, 20, 50 | Precision = hits / min(K, returned) |
| COL Recall@K | No | 10, 20, 50 | Recall = hits / GT positives |
| COL MAP | No | K-independent | Mean Average Precision over full result list |

**Precision denominator:** `min(K, len(results))` — matches TabSketchFM convention, avoids penalizing queries with fewer valid candidates after self-match filtering.

**Averaging:** Macro (mean of per-query F1s, not F1 of mean P/R).

**GT handling:** Table names normalized to basenames. Self-table pairs filtered from GT. All identifiers read as strings (`dtype=str`) to prevent pandas type coercion.

## Input Data

**Embeddings:** Both query and datalake use the same pickle (query tables are a subset of the datalake):
```
embeddings/column/<model>/opendata.pkl
```

**Query list:**
```
datasets/opendata/queries/opendata_join/opendata_join_query.csv
```
Format: `query_table,query_column`

**Ground truth:**
```
datasets/opendata/gt/opendata_join_ground_truth.csv
```
Format: `query_table,candidate_table,query_column,candidate_column`

## Example Commands

### Full Pipeline (Search + Evaluation)

```bash
python downstream_tasks/join_search/run_search_and_evaluate.py \
    --query_emb embeddings/column/turl/opendata.pkl \
    --datalake_emb embeddings/column/turl/opendata.pkl \
    --query_list datasets/opendata/queries/opendata_join/opendata_join_query.csv \
    --ground_truth datasets/opendata/gt/opendata_join_ground_truth.csv \
    --output results/evaluation/join_search/turl/turl_opendata_results.csv \
    --k 50 --k_values 10 20 50
```

### Evaluate Existing Results

```bash
python downstream_tasks/join_search/run_search_and_evaluate.py \
    --eval_only \
    --results results/evaluation/join_search/turl/turl_opendata_results.csv \
    --ground_truth datasets/opendata/gt/opendata_join_ground_truth.csv \
    --k_values 10 20 50
```

### Search Only (Skip Evaluation)

```bash
python downstream_tasks/join_search/run_search_and_evaluate.py \
    --search_only \
    --query_emb embeddings/column/turl/opendata.pkl \
    --datalake_emb embeddings/column/turl/opendata.pkl \
    --output results.csv \
    --k 50
```

### LakeBench-Compatible Table-Level Metrics

For comparability with LakeBench (VLDB 2024), which reports table-level P/R/F1, use the separate post-hoc script:

```bash
python downstream_tasks/join_search/lakebench_compat.py \
    --results results/evaluation/join_search/turl/turl_opendata_results.csv \
    --ground_truth datasets/opendata/gt/opendata_join_ground_truth.csv \
    --k_values 1 5 10
```

This aggregates column-level results to table-level via max-similarity pooling. It is **not** a table-embedding evaluation — it is a table-level readout of column-embedding search results.

### Standalone Evaluation Script

```bash
python downstream_tasks/join_search/run_evaluation.py \
    --results results.csv \
    --ground_truth datasets/opendata/gt/opendata_join_ground_truth.csv \
    --k_values 10 20 50
```

## Arguments

### run_search_and_evaluate.py (canonical pipeline)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--query_emb` | Yes | - | Path to query embeddings pickle |
| `--datalake_emb` | Yes | - | Path to datalake embeddings pickle |
| `--query_list` | No | opendata_join_query.csv | Query list CSV |
| `--ground_truth` | No | opendata_join_ground_truth.csv | Ground truth CSV |
| `--output` | No | results.csv | Output results path |
| `--k` | No | 50 | Top-K results per query |
| `--k_values` | No | 10 20 50 | K values for evaluation |
| `--threshold` | No | 0 | Minimum similarity threshold |
| `--aggregate_to_table` | No | False | Aggregate to table level for primary eval |
| `--aggregation` | No | tabsketchfm | Method: tabsketchfm, max, mean, sum |
| `--search_only` | No | False | Skip evaluation |
| `--eval_only` | No | False | Skip search, evaluate existing results |
| `--results` | No | - | Path to existing results (for --eval_only) |

### lakebench_compat.py (post-hoc table-level)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--results` | Yes | - | Path to column-level results CSV |
| `--ground_truth` | No | opendata_join_ground_truth.csv | Ground truth CSV |
| `--k_values` | No | 1 5 10 | K values for table-level evaluation |
| `--aggregation` | No | max | Column-to-table aggregation method |

## Output

**Search results CSV:** `query_table,query_column,candidate_table,candidate_column,similarity`

**Evaluation output (stdout):**
```
SEARCH COMPLETE
   Total results: 154250
   Similarity range: [0.5379, 1.0000] (mean: 0.9536)
   Avg results/query (incl. skipped): 48.6

EVALUATION RESULTS
  COL Precision@10: 0.2863 (28.63%)
  COL Recall@10:    0.2770 (27.70%)
  COL F1@10:        0.2484 (24.84%)

  COL MAP: 0.1234 (12.34%)

Statistics:
  GT queries total:        3282
  GT queries with results: 3017
  GT queries w/o results:  265 (contribute zeros to averages)
  GT coverage:             3017/3282 (91.9%)
```

## Fail-Fast Guards

The pipeline enforces several invariants for benchmark correctness:

| Guard | Behavior |
|-------|----------|
| Duplicate `(table, col)` keys in datalake | Fatal error |
| Duplicate query keys in embeddings | Fatal error |
| Duplicate query rows in query list | Fatal error |
| `--k < max(--k_values)` | Fatal error |
| `--eval_only --aggregate_to_table` | Fatal error (unsupported) |
| Zero-norm datalake vectors | Excluded from results (logged) |
| Zero-norm query vectors | Skipped with warning |
| Duplicate column names within a table | Disambiguated with `_<index>` suffix |

## Scripts

| Script | Purpose | Used by SLURM |
|--------|---------|---------------|
| `run_search_and_evaluate.py` | Canonical pipeline: exact search + eval | **Yes** |
| `run_evaluation.py` | Standalone evaluation of saved results | No |
| `run_search.py` | **Deprecated.** Legacy HNSW search wrapper (use `run_search_and_evaluate.py`) | No |
| `lakebench_compat.py` | Post-hoc LakeBench table-level metrics | No |
| `run_tabsketchfm_search.py` | Legacy TabSketchFM reproduction | No |

## Subtasks

### Subtask: Classification

Binary classification for joinability prediction. Uses table-level embeddings (`cls_embedding` or `column_mean`) to predict if two tables are joinable.

**Location:** `downstream_tasks/join_search/classification/`

See `downstream_tasks/join_search/classification/README.md` for details.

### Subtask: Containment

Estimates containment (overlap) between column pairs. Uses column-level embeddings for regression/classification on containment scores.

**Location:** `downstream_tasks/join_search/containment/`

See `downstream_tasks/join_search/containment/USAGE.md` for details.

## Related

- Embedding generation: `models/*/USAGE.md`
- Similar tasks: `downstream_tasks/union_search/`
- SLURM config: `slurm/config/downstream/tasks.yaml` (task: `join_search`)
- SLURM template: `slurm/scripts/templates/downstream/join_search.sbatch.template`
