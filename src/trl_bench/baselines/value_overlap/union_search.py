#!/usr/bin/env python3
"""Value-overlap baseline for union search.

Computes bipartite column matching between query and datalake tables using
value containment instead of cosine similarity on embeddings.  Uses the same
Hungarian matching algorithm and evaluation metrics as the embedding pipeline.

Usage:
    python -m utils.baselines.value_overlap.union_search
    python -m utils.baselines.value_overlap.union_search --K 10 --threshold 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from trl_bench.baselines.value_overlap.core import containment, extract_column_value_sets

# Hungarian algorithm — same library used by downstream_tasks/union_search/run_search.py
from munkres import Munkres, make_cost_matrix, DISALLOWED


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Bipartite matching (mirrors verify_table_match in run_search.py:73-105)
# ---------------------------------------------------------------------------

def verify_table_match_overlap(
    query_cols: dict[str, set[str]],
    cand_cols: dict[str, set[str]],
    threshold: float = 0.0,
) -> float:
    """Bipartite matching score using value containment.

    Mirrors ``verify_table_match()`` from ``run_search.py`` but replaces
    cosine similarity with containment scoring.
    """
    q_names = list(query_cols.keys())
    c_names = list(cand_cols.keys())
    nrow, ncol = len(q_names), len(c_names)

    if nrow == 0 or ncol == 0:
        return 0.0

    graph = np.zeros((nrow, ncol), dtype=float)
    for i, qn in enumerate(q_names):
        for j, cn in enumerate(c_names):
            score = containment(query_cols[qn], cand_cols[cn])
            if score > threshold:
                graph[i, j] = score

    if graph.max() == 0:
        return 0.0

    max_graph = make_cost_matrix(
        graph, lambda cost: (graph.max() - cost) if cost != DISALLOWED else DISALLOWED
    )
    m = Munkres()
    indexes = m.compute(max_graph)

    return sum(graph[row, col] for row, col in indexes)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_union(
    queries: list[tuple[str, dict[str, set[str]]]],
    datalake: list[tuple[str, dict[str, set[str]]]],
    K: int = 10,
    threshold: float = 0.0,
) -> dict[str, list[str]]:
    """Linear search over all query-datalake pairs."""
    results: dict[str, list[str]] = {}

    for query_name, query_cols in tqdm(queries, desc="Union search", ncols=80):
        scores = []
        for cand_name, cand_cols in datalake:
            score = verify_table_match_overlap(query_cols, cand_cols, threshold)
            scores.append((score, cand_name))
        scores.sort(reverse=True)
        results[query_name] = [name for _, name in scores[:K]]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Dataset configurations: dataset_name -> (gt_filename, query_subdir, datalake_subdir)
DATASET_CONFIGS = {
    'santos': ('santosUnionBenchmark.pickle', 'query', 'datalake'),
    'ugen_v1': ('benchmark.pkl', 'query', 'datalake'),
    'ugen_v2': ('benchmark.pkl', 'query', 'datalake'),
    'tus': ('benchmark.pkl', 'query', 'datalake'),
    'tus_hard': ('benchmark.pkl', 'query', 'datalake'),
}


def run_union_search_baseline(
    dataset: str,
    project_root: Path,
    K: int = 10,
    threshold: float = 0.0,
) -> dict:
    """End-to-end: load tables, search, evaluate, write results JSON."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    gt_filename, query_subdir, datalake_subdir = DATASET_CONFIGS[dataset]
    dataset_dir = project_root / 'datasets' / dataset

    print("=" * 60)
    print(f"Value-Overlap Baseline: Union Search ({dataset})")
    print(f"  K={K}, threshold={threshold}")
    print("=" * 60)

    # Load tables
    query_dir = dataset_dir / query_subdir
    datalake_dir = dataset_dir / datalake_subdir

    print("\n[1/4] Loading query tables...")
    queries = []
    for csv_file in sorted(query_dir.glob('*.csv')):
        col_sets = extract_column_value_sets(csv_file)
        if col_sets:
            queries.append((csv_file.name, col_sets))
    print(f"  Loaded {len(queries)} query tables")

    print("\n[2/4] Loading datalake tables...")
    datalake = []
    for csv_file in sorted(datalake_dir.glob('*.csv')):
        col_sets = extract_column_value_sets(csv_file)
        if col_sets:
            datalake.append((csv_file.name, col_sets))
    print(f"  Loaded {len(datalake)} datalake tables")

    # Search
    print(f"\n[3/4] Running union search ({len(queries)} x {len(datalake)} pairs)...")
    start = time.time()
    results = search_union(queries, datalake, K=K, threshold=threshold)
    elapsed = time.time() - start
    print(f"  Search completed in {elapsed:.1f}s")

    # Evaluate — reuse compute_metrics from the embedding pipeline
    print("\n[4/4] Evaluating...")
    gt_path = dataset_dir / gt_filename
    with open(gt_path, 'rb') as f:
        groundtruth = pickle.load(f)

    from trl_bench.tasks.union_search.run_search import compute_metrics

    map_score, precision_at_k, recall_at_k, gt_eval, gt_miss = compute_metrics(
        results, groundtruth, K
    )

    print(f"\n{'=' * 60}")
    print(f"  MAP@{K}:       {map_score:.4f}")
    print(f"  Precision@{K}: {precision_at_k[-1]:.4f}")
    print(f"  Recall@{K}:    {recall_at_k[-1]:.4f}")
    print(f"  GT queries evaluated: {gt_eval}, missing: {gt_miss}")
    print(f"{'=' * 60}")

    # Write results JSON (matching SLURM template format)
    result_json = {
        'model': 'value_overlap',
        'dataset': dataset,
        'task': 'union_search',
        'map_at_k': float(map_score),
        'precision_at_k': float(precision_at_k[-1]),
        'recall_at_k': float(recall_at_k[-1]),
        'hyperparameters': {
            'method': 'value_containment',
            'k': K,
            'threshold': threshold,
        },
        'gt_queries_evaluated': gt_eval,
        'gt_queries_missing': gt_miss,
        'status': 'completed',
    }

    output_dir = project_root / 'assets' / 'evaluation_results' / 'union_search' / 'value_overlap'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'value_overlap_{dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return result_json


def main():
    parser = argparse.ArgumentParser(
        description="Value-overlap baseline for union search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset (e.g., santos, ugen_v1, ugen_v2)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Multiple datasets for batch mode')
    parser.add_argument('--K', type=int, default=10, help='Top-K results per query')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Minimum containment threshold for column matching')
    args = parser.parse_args()

    if args.dataset is None and args.datasets is None:
        # Default: run all configured datasets
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = args.datasets if args.datasets else [args.dataset]

    project_root = get_project_root()
    for dataset in datasets:
        dataset_dir = project_root / 'datasets' / dataset
        if not dataset_dir.exists():
            print(f"Skipping {dataset}: directory not found at {dataset_dir}")
            continue
        run_union_search_baseline(dataset, project_root, K=args.K, threshold=args.threshold)
        print()


if __name__ == '__main__':
    main()
