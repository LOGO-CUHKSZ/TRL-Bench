#!/usr/bin/env python3
"""
LakeBench-compatible table-level evaluation for join search.

Post-hoc script that reads column-level join search results (from
run_search_and_evaluate.py) and produces table-level P/R/F1 metrics
for comparability with LakeBench (VLDB 2024).

This aggregates candidate-column similarities to candidate-table scores
via max pooling, then evaluates at table level. This is NOT a table-
embedding evaluation — it is a table-level readout of column-embedding
search results.

Usage:
    python lakebench_compat.py --results <column_results.csv> --ground_truth <gt.csv>
    python lakebench_compat.py --results <column_results.csv> --ground_truth <gt.csv> --k_values 1 5 10
"""
import os
import sys
import argparse

import pandas as pd
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="LakeBench-compatible table-level join search evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to column-level results CSV (from run_search_and_evaluate.py)")
    parser.add_argument("--ground_truth", type=str,
                        default=os.path.join(_PROJECT_ROOT, "datasets/opendata/gt/opendata_join_ground_truth.csv"),
                        help="Path to ground truth CSV file")
    parser.add_argument("--k_values", type=int, nargs='+',
                        default=[1, 5, 10],
                        help="K values for table-level evaluation")
    parser.add_argument("--aggregation", type=str, default="max",
                        choices=["max", "mean", "sum"],
                        help="Column-to-table aggregation method")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LakeBench-Compatible Table-Level Evaluation")
    print("=" * 60)
    print(f"\nNote: This is a table-level readout of column-embedding search")
    print(f"results, not a table-embedding evaluation.")

    # Load column-level results
    print(f"\n[1/4] Loading column-level results...")
    _id_str = {c: str for c in ['query_table', 'query_column', 'candidate_table', 'candidate_column']}
    results_df = pd.read_csv(args.results, dtype=_id_str, keep_default_na=False)
    results_df['query_table'] = results_df['query_table'].apply(os.path.basename)
    results_df['candidate_table'] = results_df['candidate_table'].apply(os.path.basename)
    print(f"  Loaded {len(results_df)} column-level results")

    if 'candidate_column' not in results_df.columns:
        print("Error: Results appear to already be table-level (no candidate_column column).")
        print("  This script expects column-level results from run_search_and_evaluate.py.")
        sys.exit(1)

    # Aggregate to table level
    print(f"\n[2/4] Aggregating to table level (method: {args.aggregation})...")
    grouped = results_df.groupby(['query_table', 'query_column', 'candidate_table'])

    table_results = []
    for (qt, qc, ct), group in grouped:
        sims = group['similarity'].values
        if args.aggregation == 'max':
            score = float(np.max(sims))
        elif args.aggregation == 'mean':
            score = float(np.mean(sims))
        elif args.aggregation == 'sum':
            score = float(np.sum(sims))
        else:
            score = float(np.max(sims))
        table_results.append({
            'query_table': qt,
            'query_column': qc,
            'candidate_table': ct,
            'similarity': score
        })

    table_df = pd.DataFrame(table_results)
    if len(table_df) == 0:
        print("  No results after aggregation (all GT queries will score zero).")
        table_df = pd.DataFrame(columns=['query_table', 'query_column', 'candidate_table', 'similarity'])

    # Sort by similarity descending per query
    table_df = table_df.sort_values(
        ['query_table', 'query_column', 'similarity'],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    print(f"  Aggregated to {len(table_df)} table-level results")

    # Load and prepare ground truth
    print(f"\n[3/4] Loading ground truth...")
    _gt_str = {c: str for c in ['query_table', 'query_column', 'candidate_table', 'candidate_column']}
    gt_df = pd.read_csv(args.ground_truth, dtype=_gt_str, keep_default_na=False)
    gt_df['query_table'] = gt_df['query_table'].apply(os.path.basename)
    gt_df['candidate_table'] = gt_df['candidate_table'].apply(os.path.basename)

    # Filter self-table pairs
    gt_self_mask = gt_df['query_table'] == gt_df['candidate_table']
    gt_self_count = gt_self_mask.sum()
    if gt_self_count > 0:
        print(f"  Filtered {gt_self_count} self-table pairs from ground truth")
        gt_df = gt_df[~gt_self_mask].reset_index(drop=True)

    # Build GT lookup: (query_table, query_column) -> set of candidate_tables
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        key = (row['query_table'], row['query_column'])
        if key not in gt_lookup:
            gt_lookup[key] = set()
        gt_lookup[key].add(row['candidate_table'])

    print(f"  Ground truth covers {len(gt_lookup)} unique query columns")
    if gt_lookup:
        print(f"  Average candidate tables per query: {np.mean([len(v) for v in gt_lookup.values()]):.1f}")

    # Group results
    results_grouped = dict(list(table_df.groupby(['query_table', 'query_column'])))

    # Evaluate
    print(f"\n[4/4] Calculating table-level metrics...")

    metrics = {k: {'precisions': [], 'recalls': [], 'f1s': []} for k in args.k_values}
    gt_with = 0
    gt_without = 0

    for query_key, gt_set in gt_lookup.items():
        group = results_grouped.get(query_key)

        if group is None:
            gt_without += 1
            for k in args.k_values:
                metrics[k]['precisions'].append(0.0)
                metrics[k]['recalls'].append(0.0)
                metrics[k]['f1s'].append(0.0)
            continue

        gt_with += 1
        result_tables = list(group['candidate_table'])
        # Deduplicate preserving order
        seen = set()
        unique_tables = []
        for t in result_tables:
            if t not in seen:
                seen.add(t)
                unique_tables.append(t)

        for k in args.k_values:
            topk = set(unique_tables[:k])
            hits = topk.intersection(gt_set)
            precision = len(hits) / min(k, len(unique_tables)) if unique_tables else 0
            recall = len(hits) / len(gt_set) if gt_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics[k]['precisions'].append(precision)
            metrics[k]['recalls'].append(recall)
            metrics[k]['f1s'].append(f1)

    # Print results
    print("\n" + "=" * 60)
    print("LAKEBENCH-COMPATIBLE TABLE-LEVEL RESULTS")
    print("=" * 60)

    if not gt_lookup:
        print("\n  No ground truth queries after filtering. Nothing to evaluate.")
        return

    for k in args.k_values:
        p = np.mean(metrics[k]['precisions'])
        r = np.mean(metrics[k]['recalls'])
        f = np.mean(metrics[k]['f1s'])
        print(f"\n  TBL Precision@{k}: {p:.4f} ({p*100:.2f}%)")
        print(f"  TBL Recall@{k}:    {r:.4f} ({r*100:.2f}%)")
        print(f"  TBL F1@{k}:        {f:.4f} ({f*100:.2f}%)")

    print(f"\nStatistics:")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  GT queries total:        {len(gt_lookup)}")
    print(f"  GT queries with results: {gt_with}")
    print(f"  GT queries w/o results:  {gt_without}")
    if gt_lookup:
        print(f"  GT coverage:             {gt_with}/{len(gt_lookup)} ({100*gt_with/len(gt_lookup):.1f}%)")


if __name__ == "__main__":
    main()
