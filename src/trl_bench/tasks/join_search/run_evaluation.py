#!/usr/bin/env python3
"""
Evaluation for join search results.

Supports two evaluation modes:
1. Column-level (default): Evaluates (candidate_table, candidate_column) pairs
2. Table-level (--table_level or auto-detected): Evaluates candidate_table only

Key features:
1. Handles column ordering difference between results and ground truth
2. Evaluates per (query_table, query_column) instead of per query_table
3. Properly sorts results by similarity before taking top-K
4. Filters self-table pairs from ground truth
5. Normalizes table names to basenames
"""
import os
import pandas as pd
import numpy as np
import argparse

# Resolve project root (two levels up from this script)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate join search results")
    parser.add_argument("--results", type=str,
                        default=os.path.join(_PROJECT_ROOT, "results/evaluation/join_search/results.csv"),
                        help="Path to results CSV file")
    parser.add_argument("--ground_truth", type=str,
                        default=os.path.join(_PROJECT_ROOT, "datasets/opendata/gt/opendata_join_ground_truth.csv"),
                        help="Path to ground truth CSV file")
    parser.add_argument("--k_values", type=int, nargs='+',
                        default=[10, 20, 50],
                        help="K values for evaluation (default: 10 20 50)")
    parser.add_argument("--table_level", action="store_true",
                        help="Evaluate at table level (ignore candidate_column)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load results first (needed for auto-detection before printing banner)
    print(f"\n[1/3] Loading data...")

    _id_str = {c: str for c in ['query_table', 'query_column', 'candidate_table', 'candidate_column']}
    results_df = pd.read_csv(args.results, dtype=_id_str, keep_default_na=False)
    # Normalize results table names to basenames
    results_df['query_table'] = results_df['query_table'].apply(os.path.basename)
    results_df['candidate_table'] = results_df['candidate_table'].apply(os.path.basename)
    print(f"  Results columns: {list(results_df.columns)}")
    print(f"  Results: {len(results_df)} pairs")

    _gt_str = {c: str for c in ['query_table', 'query_column', 'candidate_table', 'candidate_column']}
    gt_df = pd.read_csv(args.ground_truth, dtype=_gt_str, keep_default_na=False)
    print(f"  Ground truth columns: {list(gt_df.columns)}")
    print(f"  Ground truth: {len(gt_df)} pairs")

    # Consolidated table-level detection
    results_table_level = 'candidate_column' not in results_df.columns
    effective_table_level = args.table_level or results_table_level
    level_tag = "TBL" if effective_table_level else "COL"

    print("=" * 60)
    print(f"Evaluation ({'Table-level' if effective_table_level else 'Column-level'})")
    print("=" * 60)

    if results_table_level and not args.table_level:
        print(f"\n  Results have no candidate_column. Forced table-level evaluation.")

    if args.table_level and not results_table_level:
        print(f"\n  Results contain candidate_column but --table_level specified.")
        print(f"  Will ignore candidate_column for evaluation.")

    # Normalize table names to basenames
    gt_df['query_table'] = gt_df['query_table'].apply(os.path.basename)
    gt_df['candidate_table'] = gt_df['candidate_table'].apply(os.path.basename)

    # Filter self-table pairs from GT (after normalization)
    gt_self_mask = gt_df['query_table'] == gt_df['candidate_table']
    gt_self_count = gt_self_mask.sum()
    if gt_self_count > 0:
        print(f"  Filtered {gt_self_count} self-table pairs from ground truth")
        gt_df = gt_df[~gt_self_mask].reset_index(drop=True)

    # [2/3] Prepare data structures
    print(f"\n[2/3] Preparing data structures...")

    # Sort results by (query_table, query_column, similarity descending)
    results_df = results_df.sort_values(
        ['query_table', 'query_column', 'similarity'],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    # Create ground truth lookup based on evaluation mode
    gt_lookup = {}

    if effective_table_level:
        # Table-level: (query_table, query_column) -> set of candidate_tables
        for _, row in gt_df.iterrows():
            key = (row['query_table'], row['query_column'])
            if key not in gt_lookup:
                gt_lookup[key] = set()
            gt_lookup[key].add(row['candidate_table'])
        print(f"  Ground truth covers {len(gt_lookup)} unique query columns")
        if gt_lookup:
            print(f"  Average candidate tables per query: {np.mean([len(v) for v in gt_lookup.values()]):.1f}")
    else:
        # Column-level: (query_table, query_column) -> set of (candidate_table, candidate_column)
        for _, row in gt_df.iterrows():
            key = (row['query_table'], row['query_column'])
            if key not in gt_lookup:
                gt_lookup[key] = set()
            gt_lookup[key].add((row['candidate_table'], row['candidate_column']))
        print(f"  Ground truth covers {len(gt_lookup)} unique query columns")
        if gt_lookup:
            print(f"  Average candidate column-pairs per query: {np.mean([len(v) for v in gt_lookup.values()]):.1f}")

    # Group results by (query_table, query_column)
    results_grouped = dict(list(results_df.groupby(['query_table', 'query_column'])))
    print(f"  Results cover {len(results_grouped)} unique query columns")

    # [3/3] Calculate metrics
    print(f"\n[3/3] Calculating metrics...")

    K_values = args.k_values
    metrics = {k: {'precisions': [], 'recalls': [], 'f1s': []} for k in K_values}
    aps = []  # Average Precision per query (k-independent)

    gt_queries_with_results = 0
    gt_queries_without_results = 0
    results_not_in_gt = 0

    for query_key, gt_set in gt_lookup.items():
        group = results_grouped.get(query_key)
        if group is None:
            gt_queries_without_results += 1
            for k in K_values:
                metrics[k]['precisions'].append(0.0)
                metrics[k]['recalls'].append(0.0)
                metrics[k]['f1s'].append(0.0)
            aps.append(0.0)
            continue
        gt_queries_with_results += 1

        # Get result items for this query (already sorted by similarity)
        if effective_table_level:
            # Table-level: just candidate_table
            result_items = list(group['candidate_table'])
            # Remove duplicates while preserving order
            seen = set()
            result_items_unique = []
            for item in result_items:
                if item not in seen:
                    seen.add(item)
                    result_items_unique.append(item)
            result_items = result_items_unique
        else:
            # Column-level: (candidate_table, candidate_column) pairs
            result_items = list(zip(group['candidate_table'], group['candidate_column']))

        # Calculate metrics for different K values
        for k in K_values:
            # Take top-K results
            topk_items = set(result_items[:k])

            # Calculate intersection
            intersection = topk_items.intersection(gt_set)

            # Precision = hits / returned (not /k): matches TabSketchFM convention,
            # avoids penalizing queries with fewer valid candidates after self-match filtering.
            precision = len(intersection) / min(k, len(result_items)) if result_items else 0
            recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[k]['precisions'].append(precision)
            metrics[k]['recalls'].append(recall)
            metrics[k]['f1s'].append(f1)

        # Average Precision: AP = (1/|relevant|) * sum(P@i * rel(i))
        ap_score = 0.0
        num_hits = 0.0
        for i, item in enumerate(result_items):
            if item in gt_set:
                num_hits += 1.0
                ap_score += num_hits / (i + 1.0)
        aps.append(ap_score / len(gt_set) if len(gt_set) > 0 else 0.0)

    for query_key in results_grouped:
        if query_key not in gt_lookup:
            results_not_in_gt += 1

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for k in K_values:
        if len(metrics[k]['precisions']) == 0:
            print(f"\n{'─' * 30}")
            print(f"K = {k}")
            print(f"{'─' * 30}")
            print(f"  No data to evaluate!")
            continue

        avg_precision = np.mean(metrics[k]['precisions'])
        avg_recall = np.mean(metrics[k]['recalls'])
        avg_f1 = np.mean(metrics[k]['f1s'])

        print(f"\n{'─' * 30}")
        print(f"K = {k}")
        print(f"{'─' * 30}")
        print(f"  {level_tag} Precision@{k}: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
        print(f"  {level_tag} Recall@{k}:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
        print(f"  {level_tag} F1@{k}:        {avg_f1:.4f} ({avg_f1*100:.2f}%)")

    # MAP (Mean Average Precision) — k-independent
    map_value = np.mean(aps) if aps else 0.0
    print(f"\n{'─' * 30}")
    print(f"  {level_tag} MAP: {map_value:.4f} ({map_value*100:.2f}%)")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Evaluation mode: {'Table-level' if effective_table_level else 'Column-level'}")
    print(f"  GT queries total:        {len(gt_lookup)}")
    print(f"  GT queries with results: {gt_queries_with_results}")
    print(f"  GT queries w/o results:  {gt_queries_without_results} (contribute zeros to averages)")
    if len(gt_lookup) > 0:
        print(f"  GT coverage:             {gt_queries_with_results}/{len(gt_lookup)} ({100 * gt_queries_with_results / len(gt_lookup):.1f}%)")
    else:
        print(f"  GT coverage:             0/0 (no ground truth queries)")
    print(f"  Result queries not in GT:{results_not_in_gt} (excluded from metrics)")
