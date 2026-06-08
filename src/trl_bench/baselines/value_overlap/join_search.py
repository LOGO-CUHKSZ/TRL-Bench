#!/usr/bin/env python3
"""Value-overlap baseline for join search.

Computes column-level containment scores using an inverted index over raw
cell values, then evaluates with the same metrics as the embedding pipeline.

Usage:
    python -m utils.baselines.value_overlap.join_search --dataset opendata_main
    python -m utils.baselines.value_overlap.join_search --datasets opendata_main opendata_can opendata_usa opendata_uk_sg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from trl_bench.baselines.value_overlap.core import (
    containment,
    extract_column_value_sets,
    normalize_column_name,
    normalize_value,
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Inverted index construction (streaming — one table at a time)
# ---------------------------------------------------------------------------

def build_inverted_index(
    tables_dir: Path,
    max_posting: int = 10_000,
) -> tuple[dict[str, list[tuple[str, str]]], int, int]:
    """Build inverted index from cell values to (table, column) posting lists.

    Streams tables one at a time to limit memory.  Discards each table's
    value sets after inserting into the index.

    Args:
        tables_dir: Directory containing CSV table files.
        max_posting: Drop values whose posting list exceeds this length.

    Returns:
        inverted_index: {normalized_value: [(table_basename, col_name), ...]}
        n_tables: Number of tables indexed.
        n_columns: Total number of columns indexed.
    """
    inverted_index: dict[str, list[tuple[str, str]]] = {}
    n_tables = 0
    n_columns = 0

    csv_files = sorted(tables_dir.glob('*.csv'))
    for csv_path in tqdm(csv_files, desc="Indexing datalake", ncols=80):
        col_sets = extract_column_value_sets(csv_path)
        if not col_sets:
            continue
        table_name = csv_path.name
        n_tables += 1
        for col_name, value_set in col_sets.items():
            n_columns += 1
            for val in value_set:
                posting = inverted_index.get(val)
                if posting is None:
                    inverted_index[val] = [(table_name, col_name)]
                else:
                    posting.append((table_name, col_name))

    # Prune ubiquitous values
    pruned = 0
    for val in list(inverted_index.keys()):
        if len(inverted_index[val]) > max_posting:
            del inverted_index[val]
            pruned += 1
    if pruned:
        print(f"  Pruned {pruned} ubiquitous values (posting list > {max_posting})")

    return inverted_index, n_tables, n_columns


# ---------------------------------------------------------------------------
# Query scoring
# ---------------------------------------------------------------------------

def resolve_query_table(
    query_table: str,
    dataset_dir: Path,
) -> Path | None:
    """Resolve query table file with fallback paths.

    Checks queries/opendata_join/tables/ first (exists for opendata only),
    then falls back to tables/.
    """
    query_tables_dir = dataset_dir / 'queries' / 'opendata_join' / 'tables' / query_table
    if query_tables_dir.exists():
        return query_tables_dir
    datalake_path = dataset_dir / 'tables' / query_table
    if datalake_path.exists():
        return datalake_path
    return None


def score_queries(
    query_list_path: Path,
    dataset_dir: Path,
    inverted_index: dict[str, list[tuple[str, str]]],
    k: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Score all queries using the inverted index.

    Returns:
        results_df: DataFrame with columns matching run_evaluation() expectation.
        coverage: Audit dict with coverage statistics.
    """
    query_df = pd.read_csv(
        query_list_path, dtype={'query_table': str, 'query_column': str},
        keep_default_na=False,
    )
    query_df['query_table'] = query_df['query_table'].apply(os.path.basename)

    results = []
    unresolved_query_cols = 0
    _COLUMNS = ['query_table', 'query_column', 'candidate_table', 'candidate_column', 'similarity']

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Scoring queries", ncols=80):
        query_table = row['query_table']
        query_column = row['query_column']

        # Resolve query table file
        table_path = resolve_query_table(query_table, dataset_dir)
        if table_path is None:
            unresolved_query_cols += 1
            continue

        # Extract value set for just this query column
        all_col_sets = extract_column_value_sets(table_path)
        if query_column not in all_col_sets:
            # Try normalized match
            found = False
            for col_name in all_col_sets:
                if normalize_column_name(col_name) == query_column:
                    query_set = all_col_sets[col_name]
                    found = True
                    break
            if not found:
                unresolved_query_cols += 1
                continue
        else:
            query_set = all_col_sets[query_column]

        if not query_set:
            continue

        # Look up each value in inverted index, accumulate hits
        hits: Counter = Counter()
        for val in query_set:
            posting = inverted_index.get(val)
            if posting is not None:
                for entry in posting:
                    hits[entry] += 1

        # Compute containment and rank
        query_set_size = len(query_set)
        scored = []
        for (cand_table, cand_col), hit_count in hits.items():
            # Exclude self-table matches
            if cand_table == query_table:
                continue
            score = hit_count / query_set_size
            scored.append((cand_table, cand_col, score))

        # Top-K
        scored.sort(key=lambda x: x[2], reverse=True)
        for cand_table, cand_col, score in scored[:k]:
            results.append({
                'query_table': query_table,
                'query_column': query_column,
                'candidate_table': cand_table,
                'candidate_column': cand_col,
                'similarity': score,
            })

    results_df = pd.DataFrame(results, columns=_COLUMNS) if results else pd.DataFrame(columns=_COLUMNS)

    coverage = {
        'unresolved_query_cols': unresolved_query_cols,
        'total_queries': len(query_df),
        'resolved_queries': len(query_df) - unresolved_query_cols,
    }

    return results_df, coverage


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_join_search_baseline(
    dataset: str,
    project_root: Path,
    k: int = 50,
    k_values: list[int] | None = None,
    max_posting: int = 10_000,
) -> dict:
    """End-to-end: index datalake, score queries, evaluate, write results JSON."""
    if k_values is None:
        k_values = [10, 20, 50]

    dataset_dir = project_root / 'datasets' / dataset

    print("=" * 60)
    print(f"Value-Overlap Baseline: Join Search ({dataset})")
    print(f"  k={k}, k_values={k_values}, max_posting={max_posting}")
    print("=" * 60)

    # Build inverted index
    print("\n[1/3] Building inverted index...")
    tables_dir = dataset_dir / 'tables'
    start = time.time()
    inverted_index, n_tables, n_columns = build_inverted_index(tables_dir, max_posting)
    elapsed = time.time() - start
    print(f"  Indexed {n_tables} tables, {n_columns} columns in {elapsed:.1f}s")
    print(f"  Index size: {len(inverted_index)} distinct values")

    # Score queries
    print("\n[2/3] Scoring queries...")
    query_list_path = dataset_dir / 'queries' / 'opendata_join' / 'opendata_join_query.csv'
    start = time.time()
    results_df, coverage = score_queries(query_list_path, dataset_dir, inverted_index, k)
    elapsed = time.time() - start
    print(f"  Scored {coverage['resolved_queries']}/{coverage['total_queries']} queries in {elapsed:.1f}s")
    print(f"  Total results: {len(results_df)}")
    if coverage['unresolved_query_cols'] > 0:
        print(f"  WARNING: {coverage['unresolved_query_cols']} query columns could not be resolved")

    # Evaluate — reuse existing evaluation function
    print("\n[3/3] Evaluating...")
    from trl_bench.tasks.join_search.run_search_and_evaluate import run_evaluation

    gt_path = dataset_dir / 'gt' / 'opendata_join_ground_truth.csv'
    metrics = run_evaluation(results_df, str(gt_path), k_values)

    # Build result JSON (matching SLURM template keys)
    result_json: dict = {
        'model': 'value_overlap',
        'dataset': dataset,
        'task': 'join_search',
        'hyperparameters': {
            'method': 'value_containment',
            'k': k,
            'max_posting': max_posting,
        },
        'coverage': coverage,
        'status': 'completed',
    }

    # Extract metrics in col_ prefix format
    for kv in k_values:
        if kv in metrics:
            result_json[f'col_precision_at_{kv}'] = float(metrics[kv]['precision'])
            result_json[f'col_recall_at_{kv}'] = float(metrics[kv]['recall'])
            result_json[f'col_f1_at_{kv}'] = float(metrics[kv]['f1'])
    if 'map' in metrics:
        result_json['col_map'] = float(metrics['map'])

    # Write results
    output_dir = project_root / 'assets' / 'evaluation_results' / 'join_search' / 'value_overlap'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'value_overlap_{dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return result_json


def main():
    parser = argparse.ArgumentParser(
        description="Value-overlap baseline for join search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name (e.g., opendata_main)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Multiple dataset names for batch mode')
    parser.add_argument('--k', type=int, default=50,
                        help='Top-K results per query for search')
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 20, 50],
                        help='K values for evaluation metrics')
    parser.add_argument('--max_posting', type=int, default=10_000,
                        help='Max posting list length before pruning')
    args = parser.parse_args()

    if args.dataset is None and args.datasets is None:
        parser.error("Provide --dataset or --datasets")

    datasets = args.datasets if args.datasets else [args.dataset]
    project_root = get_project_root()

    for dataset in datasets:
        run_join_search_baseline(
            dataset=dataset,
            project_root=project_root,
            k=args.k,
            k_values=args.k_values,
            max_posting=args.max_posting,
        )
        print()


if __name__ == '__main__':
    main()
