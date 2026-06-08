#!/usr/bin/env python3
"""
Run Valentine classical baselines (JaccardDistance, DistributionBased) on the
schema matching task.

These are training-free baselines that compute column similarity from raw data
(column names and/or values) rather than learned embeddings. They provide a
reference point for whether embeddings capture anything beyond simple
lexical/statistical column features.

Usage:
    # Run JaccardDistance baseline (fast, header+value Jaccard)
    python utils/baselines/valentine_baselines.py \
        --matcher jaccard \
        --pairs datasets/valentine/pairs.json \
        --ground_truth datasets/valentine/ground_truth.csv \
        --tables_dir datasets/valentine/tables \
        --output_dir results/evaluation/schema_matching/jaccard_baseline/valentine

    # Run DistributionBased baseline (slower, value distribution comparison)
    python utils/baselines/valentine_baselines.py \
        --matcher distribution \
        --pairs datasets/valentine/pairs.json \
        --ground_truth datasets/valentine/ground_truth.csv \
        --tables_dir datasets/valentine/tables \
        --output_dir results/evaluation/schema_matching/distribution_baseline/valentine

    # Run both
    python utils/baselines/valentine_baselines.py \
        --matcher jaccard distribution \
        --pairs datasets/valentine/pairs.json \
        --ground_truth datasets/valentine/ground_truth.csv \
        --tables_dir datasets/valentine/tables \
        --output_dir results/evaluation/schema_matching
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Valentine matchers
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher, DistributionBased
from valentine.algorithms.jaccard_distance import StringDistanceFunction


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth & pairs loading (mirrors run_schema_matching.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth CSV into dict: pair_id -> set of (col_a, col_b)."""
    gt = defaultdict(set)
    with open(gt_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['pair_id']].add((row['column_a'], row['column_b']))
    return dict(gt)


def load_pairs(pairs_path: str) -> list:
    """Load pairs JSON."""
    with open(pairs_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Recall@GT computation (mirrors run_schema_matching.py:compute_recall_at_gt)
# ─────────────────────────────────────────────────────────────────────────────

def compute_recall_at_gt_from_valentine_matches(matches, gt_pairs: set) -> float:
    """Compute Recall@GT from valentine MatcherResults.

    Ranks all scored column pairs descending, takes top-k where k=|GT|,
    counts how many are correct. This mirrors the embedding-based evaluation.

    Args:
        matches: valentine MatcherResults dict {((t1, col_a), (t2, col_b)): score}
        gt_pairs: set of (col_a, col_b) ground truth correspondences

    Returns:
        Recall@GT value (0.0 to 1.0)
    """
    k = len(gt_pairs)
    if k == 0:
        return 0.0

    # Extract (col_a, col_b, score) from valentine results
    scored = []
    for ((_t1, col_a), (_t2, col_b)), score in matches.items():
        scored.append((col_a, col_b, score))

    # Sort descending by score
    scored.sort(key=lambda x: x[2], reverse=True)

    # Take top-k and count correct
    tp = sum(1 for ca, cb, _ in scored[:k] if (ca, cb) in gt_pairs)
    return tp / k


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_valentine_baseline(
    matcher_name: str,
    pairs: list,
    gt: dict,
    tables_dir: str,
    verbose: bool = False,
    max_rows: int = 500,
) -> dict:
    """Run a Valentine matcher on all pairs and compute Recall@GT.

    Args:
        matcher_name: 'jaccard' or 'distribution'
        pairs: list of pair dicts from pairs.json
        gt: dict of pair_id -> set of (col_a, col_b) ground truth
        tables_dir: directory containing table CSV files
        verbose: print per-pair results
        max_rows: maximum rows per table (sample if larger, for performance)

    Returns:
        dict with aggregate metrics and per-pair results
    """
    # Initialize matcher
    if matcher_name == 'jaccard':
        # Use Exact string equality for value comparison — fast and interpretable.
        # Tests "do columns share the same cell values?" without fuzzy matching.
        matcher = JaccardDistanceMatcher(distance_fun=StringDistanceFunction.Exact)
        model_name = 'jaccard_baseline'
    elif matcher_name == 'jaccard_levenshtein':
        # Use Levenshtein for fuzzy matching — slower but handles typos/formatting.
        matcher = JaccardDistanceMatcher(distance_fun=StringDistanceFunction.Levenshtein)
        model_name = 'jaccard_lev_baseline'
    elif matcher_name == 'distribution':
        matcher = DistributionBased()
        model_name = 'distribution_baseline'
    else:
        raise ValueError(f"Unknown matcher: {matcher_name}")

    pair_results = []
    skipped = 0
    start_time = time.time()

    for i, pair in enumerate(pairs):
        pair_id = pair['pair_id']
        table_a_path = os.path.join(tables_dir, pair['table_a'])
        table_b_path = os.path.join(tables_dir, pair['table_b'])

        # Skip if files missing
        if not os.path.exists(table_a_path) or not os.path.exists(table_b_path):
            skipped += 1
            continue

        # Skip if no ground truth for this pair
        gt_pairs = gt.get(pair_id, set())
        if not gt_pairs:
            skipped += 1
            continue

        try:
            df_a = pd.read_csv(table_a_path, low_memory=False)
            df_b = pd.read_csv(table_b_path, low_memory=False)
            # Sample rows for performance — column-level similarity saturates
            # well before all rows are compared
            if max_rows and len(df_a) > max_rows:
                df_a = df_a.sample(n=max_rows, random_state=42)
            if max_rows and len(df_b) > max_rows:
                df_b = df_b.sample(n=max_rows, random_state=42)
        except Exception as e:
            if verbose:
                print(f"  Skipping {pair_id}: failed to read CSV: {e}")
            skipped += 1
            continue

        # Run valentine matcher
        try:
            matches = valentine_match(df_a, df_b, matcher,
                                      pair['table_a'], pair['table_b'])
        except Exception as e:
            if verbose:
                print(f"  Skipping {pair_id}: matcher failed: {e}")
            skipped += 1
            continue

        # Compute Recall@GT
        recall_at_gt = compute_recall_at_gt_from_valentine_matches(matches, gt_pairs)

        pair_results.append({
            'pair_id': pair_id,
            'source': pair.get('source', ''),
            'recall_at_gt': recall_at_gt,
            'n_gt': len(gt_pairs),
            'n_matches': len(matches),
            'n_cols_a': len(df_a.columns),
            'n_cols_b': len(df_b.columns),
        })

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s "
                  f"avg_recall@gt={np.mean([r['recall_at_gt'] for r in pair_results]):.4f}")

    elapsed = time.time() - start_time

    # Aggregate: macro-average Recall@GT across pairs
    if pair_results:
        macro_recall_at_gt = np.mean([r['recall_at_gt'] for r in pair_results])
    else:
        macro_recall_at_gt = 0.0

    # Per-source breakdown
    source_recalls = defaultdict(list)
    for r in pair_results:
        source_recalls[r['source']].append(r['recall_at_gt'])
    per_source = {src: float(np.mean(vals)) for src, vals in source_recalls.items()}

    results = {
        'task': 'schema_matching',
        'model': model_name,
        'dataset': 'valentine',
        'matcher': matcher_name,
        'recall_at_gt': float(macro_recall_at_gt),
        'n_pairs_evaluated': len(pair_results),
        'n_pairs_skipped': skipped,
        'n_pairs_total': len(pairs),
        'per_source': per_source,
        'elapsed_seconds': round(elapsed, 1),
        'status': 'completed',
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Valentine classical baselines for schema matching"
    )
    parser.add_argument('--matcher', nargs='+', default=['jaccard'],
                        choices=['jaccard', 'jaccard_levenshtein', 'distribution'],
                        help='Valentine matcher(s) to run')
    parser.add_argument('--pairs', type=str, required=True,
                        help='Path to pairs.json')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground_truth.csv')
    parser.add_argument('--tables_dir', type=str, required=True,
                        help='Directory containing table CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--max_rows', type=int, default=500,
                        help='Max rows per table (sample if larger, 0=no limit)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-pair progress')
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    gt = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(pairs)} pairs, {len(gt)} with ground truth")

    for matcher_name in args.matcher:
        print(f"\n{'='*60}")
        print(f"Running {matcher_name} baseline on {len(pairs)} pairs...")
        print(f"{'='*60}")

        results = run_valentine_baseline(
            matcher_name=matcher_name,
            pairs=pairs,
            gt=gt,
            tables_dir=args.tables_dir,
            verbose=args.verbose,
            max_rows=args.max_rows or None,
        )

        print(f"\nResults ({matcher_name}):")
        print(f"  Recall@GT (macro): {results['recall_at_gt']:.4f} "
              f"({results['recall_at_gt']*100:.2f}%)")
        print(f"  Pairs evaluated: {results['n_pairs_evaluated']}/{results['n_pairs_total']}")
        print(f"  Elapsed: {results['elapsed_seconds']}s")
        if results.get('per_source'):
            print(f"  Per-source breakdown:")
            for src, val in sorted(results['per_source'].items()):
                print(f"    {src}: {val:.4f}")

        # Save results
        out_dir = os.path.join(args.output_dir, f"{matcher_name}_baseline", "valentine")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{matcher_name}_baseline_valentine.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
