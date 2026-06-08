#!/usr/bin/env python3
"""
Schema matching evaluation using column embedding cosine similarity + bipartite matching.

Evaluates how well column embeddings can identify correct column correspondences
between pairs of tables in the Valentine benchmark.

Usage:
    python downstream_tasks/schema_matching/run_schema_matching.py \
        --embeddings embeddings/column/doduo/valentine.pkl \
        --pairs datasets/valentine/pairs.json \
        --ground_truth datasets/valentine/ground_truth.csv \
        --tables_dir datasets/valentine/tables \
        --output_dir results/evaluation/schema_matching/doduo/valentine \
        --matching_strategy hungarian \
        --threshold 0.0
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Resolve project root (two levels up from this script)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path for imports
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trl_bench.utils.unified_embedding_format.io import load_embeddings


# =============================================================================
# Embedding Lookup
# =============================================================================

def build_embedding_lookup(data):
    """Build a lookup dict from embedding data, handling various table ID formats.

    Tries all key variants: 'table', 'source_path', 'table_id', 'table_name'.
    Registers both with-.csv and stem-only forms for flexible matching.

    Args:
        data: Raw embedding data (list of dicts) loaded with as_dataclass=False.

    Returns:
        dict mapping table filename -> {'column_embeddings': dict, 'column_names': list|None}
    """
    lookup = {}

    # Handle unified batch format (dict with 'results' key)
    if isinstance(data, dict):
        if 'results' in data and isinstance(data['results'], list):
            data = data['results']
        else:
            return lookup

    if not isinstance(data, list):
        return lookup

    for entry in data:
        if not isinstance(entry, dict):
            continue

        col_emb = entry.get('column_embeddings') or entry.get('column_embedding', {})
        if not col_emb:
            continue

        col_names = entry.get('column_names')

        record = {
            'column_embeddings': col_emb,
            'column_names': col_names,
        }

        # Register under all available name variants
        for key in ('table', 'source_path', 'table_id', 'table_name'):
            raw = entry.get(key)
            if not raw:
                continue

            basename = os.path.basename(str(raw))
            stem = Path(basename).stem

            # Register both with extension and stem-only
            if basename not in lookup:
                lookup[basename] = record
            if stem not in lookup:
                lookup[stem] = record

    return lookup


def resolve_column_names(col_emb, col_names_from_pkl, table_csv_path=None):
    """Resolve integer column indices to column name strings.

    Prefers column_names from pickle. Falls back to reading CSV header.

    Args:
        col_emb: dict mapping col_idx -> embedding vector
        col_names_from_pkl: list of column names from pickle (may be None)
        table_csv_path: path to the CSV file for fallback header reading

    Returns:
        dict mapping column_name (str) -> embedding vector (ndarray)
    """
    resolved = {}

    # Determine column names for index resolution
    col_names = col_names_from_pkl
    if col_names is None and table_csv_path and os.path.exists(table_csv_path):
        try:
            with open(table_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                col_names = next(reader, None)
        except Exception:
            col_names = None

    for col_idx, emb in col_emb.items():
        # Determine column name
        if isinstance(col_idx, str) and not col_idx.isdigit():
            # Already a string column name
            name = col_idx
        else:
            # Integer or digit-string index
            idx = int(col_idx)
            if col_names and idx < len(col_names):
                name = col_names[idx]
            else:
                name = str(idx)

        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            emb = emb.astype(np.float32)

        if name in resolved:
            print(f"  Warning: duplicate column name '{name}' in table, "
                  f"overwriting embedding (column index {col_idx})", file=sys.stderr)
        resolved[name] = emb

    return resolved


# =============================================================================
# Matching Algorithms
# =============================================================================

def compute_cosine_sim_matrix(emb_a, emb_b):
    """Compute cosine similarity matrix between two sets of embeddings.

    Args:
        emb_a: dict mapping col_name -> embedding (for table A)
        emb_b: dict mapping col_name -> embedding (for table B)

    Returns:
        sim_matrix: ndarray of shape (n_a, n_b)
        names_a: list of column names for table A
        names_b: list of column names for table B
    """
    names_a = sorted(emb_a.keys())
    names_b = sorted(emb_b.keys())

    vecs_a = np.array([emb_a[n] for n in names_a], dtype=np.float32)
    vecs_b = np.array([emb_b[n] for n in names_b], dtype=np.float32)

    # L2-normalize for cosine similarity
    norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    norms_a[norms_a == 0] = 1.0
    norms_b[norms_b == 0] = 1.0
    vecs_a_normed = vecs_a / norms_a
    vecs_b_normed = vecs_b / norms_b

    sim_matrix = vecs_a_normed @ vecs_b_normed.T  # (n_a, n_b)

    return sim_matrix, names_a, names_b


def compute_recall_at_gt(sim_matrix, names_a, names_b, gt_pairs):
    """Compute Recall@GT (Valentine-standard metric) for a single table pair.

    Ranks all m x n pairwise cosine similarities descending, takes the top-k
    entries where k = |ground_truth|, and counts how many are correct.

    This is the primary metric from the Valentine benchmark (Koutras et al.,
    ICDE 2021), also used by Magneto (VLDB 2025) and Unicorn (SIGMOD 2024).

    Args:
        sim_matrix: (m, n) cosine similarity matrix
        names_a: list of m column names for table A
        names_b: list of n column names for table B
        gt_pairs: set of (col_a, col_b) ground-truth correspondences

    Returns:
        float: Recall@GT value (0.0 to 1.0), or 0.0 if gt_pairs is empty
    """
    k = len(gt_pairs)
    if k == 0:
        return 0.0

    # Build flat list of (col_a, col_b, similarity)
    scored_pairs = []
    for i, na in enumerate(names_a):
        for j, nb in enumerate(names_b):
            scored_pairs.append((na, nb, float(sim_matrix[i, j])))

    # Sort descending by similarity (stable sort preserves alphabetical
    # column order on ties, since names_a/names_b are sorted alphabetically
    # by compute_cosine_sim_matrix)
    scored_pairs.sort(key=lambda x: x[2], reverse=True)

    # Take top-k and count correct
    tp = sum(1 for na, nb, _ in scored_pairs[:k] if (na, nb) in gt_pairs)

    return tp / k


def compute_gt_column_coverage(emb_a, emb_b, gt_pairs):
    """Compute what fraction of ground-truth correspondences have embeddings for both columns.

    Args:
        emb_a: dict mapping col_name -> embedding for table A (already resolved)
        emb_b: dict mapping col_name -> embedding for table B (already resolved)
        gt_pairs: set of (col_a, col_b) ground-truth correspondences

    Returns:
        dict with gt_covered, gt_total, gt_coverage_pct
    """
    if not gt_pairs:
        return {'gt_covered': 0, 'gt_total': 0, 'gt_coverage_pct': 1.0}

    covered = sum(1 for ca, cb in gt_pairs if ca in emb_a and cb in emb_b)
    total = len(gt_pairs)

    return {
        'gt_covered': covered,
        'gt_total': total,
        'gt_coverage_pct': covered / total if total > 0 else 0.0,
    }


def match_hungarian(sim_matrix, names_a, names_b, threshold=0.0):
    """Hungarian (optimal) bipartite matching with threshold applied INSIDE the solver.

    Args:
        sim_matrix: (n_a, n_b) cosine similarity matrix
        names_a: column names for rows
        names_b: column names for columns
        threshold: minimum similarity to consider a match

    Returns:
        set of (col_a, col_b) predicted matches
    """
    # Zero out similarities below threshold BEFORE running solver
    sim = sim_matrix.copy()
    sim[sim < threshold] = 0.0

    if sim.max() <= 0:
        return set()

    # Convert to cost matrix for minimization
    cost = sim.max() - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    # Return matched pairs where thresholded similarity > 0
    matches = set()
    for r, c in zip(row_ind, col_ind):
        if sim[r, c] > 0:
            matches.add((names_a[r], names_b[c]))

    return matches


def match_greedy(sim_matrix, names_a, names_b, threshold=0.0):
    """Greedy bipartite matching: assign highest-similarity pairs first.

    Args:
        sim_matrix: (n_a, n_b) cosine similarity matrix
        names_a: column names for rows
        names_b: column names for columns
        threshold: minimum similarity to consider a match

    Returns:
        set of (col_a, col_b) predicted matches
    """
    # Zero out similarities below threshold BEFORE greedy assignment
    # (consistent with hungarian, which also thresholds then checks > 0)
    sim = sim_matrix.copy()
    sim[sim < threshold] = 0.0

    # Flatten and sort by similarity (descending)
    flat_indices = np.argsort(-sim.ravel())

    used_a = set()
    used_b = set()
    matches = set()

    for flat_idx in flat_indices:
        r = flat_idx // sim.shape[1]
        c = flat_idx % sim.shape[1]

        if sim[r, c] <= 0:
            break  # All remaining are zero or negative

        if r not in used_a and c not in used_b:
            matches.add((names_a[r], names_b[c]))
            used_a.add(r)
            used_b.add(c)

    return matches


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_prf1(predicted, ground_truth):
    """Compute precision, recall, F1 for two sets of (col_a, col_b) pairs.

    Returns:
        dict with precision, recall, f1, tp, n_predicted, n_gt
    """
    tp = len(predicted & ground_truth)
    n_predicted = len(predicted)
    n_gt = len(ground_truth)

    precision = tp / n_predicted if n_predicted > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'n_predicted': n_predicted,
        'n_gt': n_gt,
    }


def compute_micro_prf1(pair_results):
    """Compute micro-averaged P/R/F1 from pooled counts."""
    total_tp = sum(r['tp'] for r in pair_results)
    total_predicted = sum(r['n_predicted'] for r in pair_results)
    total_gt = sum(r['n_gt'] for r in pair_results)

    precision = total_tp / total_predicted if total_predicted > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_predicted': total_predicted,
        'total_gt': total_gt,
    }


def compute_macro_prf1(pair_results):
    """Compute macro-averaged P/R/F1 (equal weight per pair)."""
    if not pair_results:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return {
        'precision': sum(r['precision'] for r in pair_results) / len(pair_results),
        'recall': sum(r['recall'] for r in pair_results) / len(pair_results),
        'f1': sum(r['f1'] for r in pair_results) / len(pair_results),
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def load_ground_truth(gt_path):
    """Load ground truth CSV into a dict: pair_id -> set of (col_a, col_b)."""
    gt = defaultdict(set)
    gt_metadata = {}

    with open(gt_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row['pair_id']
            gt[pair_id].add((row['column_a'], row['column_b']))
            if pair_id not in gt_metadata:
                gt_metadata[pair_id] = {
                    'source': row.get('source', ''),
                    'noise_type': row.get('noise_type', ''),
                    'noise_param': row.get('noise_param', ''),
                }

    return dict(gt), gt_metadata


def load_pairs(pairs_path):
    """Load pairs manifest JSON."""
    with open(pairs_path, 'r') as f:
        return json.load(f)


def evaluate_schema_matching(args):
    """Run schema matching evaluation."""

    # Load data
    print(f"Loading embeddings from {args.embeddings} ...")
    raw_data = load_embeddings(args.embeddings, as_dataclass=False)
    lookup = build_embedding_lookup(raw_data)
    print(f"  Embedding lookup: {len(lookup)} table entries")

    print(f"Loading pairs from {args.pairs} ...")
    pairs = load_pairs(args.pairs)
    print(f"  Pairs: {len(pairs)}")

    print(f"Loading ground truth from {args.ground_truth} ...")
    gt_map, gt_metadata = load_ground_truth(args.ground_truth)
    print(f"  Ground truth: {len(gt_map)} pairs with mappings")

    # Select matching function
    if args.matching_strategy == 'hungarian':
        match_fn = match_hungarian
    elif args.matching_strategy == 'greedy':
        match_fn = match_greedy
    else:
        print(f"Error: unknown matching strategy: {args.matching_strategy}", file=sys.stderr)
        sys.exit(1)

    # Evaluate each pair
    all_pair_results = []
    n_skipped = 0

    for pair in pairs:
        pair_id = pair['pair_id']
        table_a = pair['table_a']
        table_b = pair['table_b']

        # Look up embeddings (try with extension first, then stem)
        emb_entry_a = lookup.get(table_a) or lookup.get(Path(table_a).stem)
        emb_entry_b = lookup.get(table_b) or lookup.get(Path(table_b).stem)

        if emb_entry_a is None or emb_entry_b is None:
            n_skipped += 1
            if emb_entry_a is None:
                print(f"  Warning: missing embeddings for {table_a}", file=sys.stderr)
            if emb_entry_b is None:
                print(f"  Warning: missing embeddings for {table_b}", file=sys.stderr)
            continue

        # Resolve column names
        table_a_csv = os.path.join(args.tables_dir, table_a) if args.tables_dir else None
        table_b_csv = os.path.join(args.tables_dir, table_b) if args.tables_dir else None

        emb_a = resolve_column_names(
            emb_entry_a['column_embeddings'],
            emb_entry_a['column_names'],
            table_a_csv
        )
        emb_b = resolve_column_names(
            emb_entry_b['column_embeddings'],
            emb_entry_b['column_names'],
            table_b_csv
        )

        if not emb_a or not emb_b:
            n_skipped += 1
            continue

        # Compute similarity matrix and match
        sim_matrix, names_a, names_b = compute_cosine_sim_matrix(emb_a, emb_b)
        predicted = match_fn(sim_matrix, names_a, names_b, threshold=args.threshold)

        # Get ground truth for this pair
        gt_pairs = gt_map.get(pair_id, set())

        # Compute Recall@GT (primary metric, Valentine-standard)
        recall_gt = compute_recall_at_gt(sim_matrix, names_a, names_b, gt_pairs)

        # Compute GT column coverage
        coverage = compute_gt_column_coverage(emb_a, emb_b, gt_pairs)

        # Compute P/R/F1 (secondary metrics)
        prf1 = compute_prf1(predicted, gt_pairs)
        prf1['pair_id'] = pair_id
        prf1['source'] = pair.get('source', '')
        prf1['noise_type'] = pair.get('noise_type', '')
        prf1['noise_param'] = pair.get('noise_param', '')
        prf1['recall_at_gt'] = recall_gt
        prf1['gt_covered'] = coverage['gt_covered']
        prf1['gt_total'] = coverage['gt_total']
        prf1['gt_coverage_pct'] = coverage['gt_coverage_pct']

        all_pair_results.append(prf1)

    # Aggregate results
    print(f"\nEvaluated {len(all_pair_results)} pairs, skipped {n_skipped}")

    overall_micro = compute_micro_prf1(all_pair_results)
    overall_macro = compute_macro_prf1(all_pair_results)

    # Recall@GT aggregation (macro = mean across pairs, the Valentine convention)
    if all_pair_results:
        recall_at_gt = sum(r['recall_at_gt'] for r in all_pair_results) / len(all_pair_results)
    else:
        recall_at_gt = 0.0

    # GT coverage aggregation
    total_gt_covered = sum(r['gt_covered'] for r in all_pair_results)
    total_gt_total = sum(r['gt_total'] for r in all_pair_results)
    gt_coverage = total_gt_covered / total_gt_total if total_gt_total > 0 else 0.0

    # Per-source breakdown
    per_source = {}
    source_groups = defaultdict(list)
    for r in all_pair_results:
        source_groups[r['source']].append(r)
    for source, results in sorted(source_groups.items()):
        src_recall_gt = sum(r['recall_at_gt'] for r in results) / len(results) if results else 0.0
        per_source[source] = {
            'recall_at_gt': src_recall_gt,
            'micro': compute_micro_prf1(results),
            'macro': compute_macro_prf1(results),
            'n_pairs': len(results),
        }

    # Per-noise_type breakdown
    per_noise = {}
    noise_groups = defaultdict(list)
    for r in all_pair_results:
        noise_groups[r['noise_type']].append(r)
    for noise_type, results in sorted(noise_groups.items()):
        noise_recall_gt = sum(r['recall_at_gt'] for r in results) / len(results) if results else 0.0
        per_noise[noise_type] = {
            'recall_at_gt': noise_recall_gt,
            'micro': compute_micro_prf1(results),
            'macro': compute_macro_prf1(results),
            'n_pairs': len(results),
        }

    # Extract model name from embeddings path
    emb_path = Path(args.embeddings)
    model_name = emb_path.parent.name

    # Build results dict
    results = {
        'model': model_name,
        'dataset': 'valentine',
        'task': 'schema_matching',
        'config': {
            'matching_strategy': args.matching_strategy,
            'threshold': args.threshold,
        },
        # Primary metric (flat key for aggregator compatibility)
        'recall_at_gt': recall_at_gt,
        'gt_coverage': gt_coverage,
        # Secondary metrics (flat keys for aggregator compatibility)
        'micro_precision': overall_micro['precision'],
        'micro_recall': overall_micro['recall'],
        'micro_f1': overall_micro['f1'],
        'macro_precision': overall_macro['precision'],
        'macro_recall': overall_macro['recall'],
        'macro_f1': overall_macro['f1'],
        # Detailed breakdowns (nested, for analysis)
        'overall_micro': overall_micro,
        'overall_macro': overall_macro,
        'n_pairs': len(all_pair_results),
        'n_skipped': n_skipped,
        'n_gt_covered': total_gt_covered,
        'n_gt_total': total_gt_total,
        'per_source': per_source,
        'per_noise_type': per_noise,
    }

    # Output to stdout (for SLURM template extract_metrics)
    print(f"\nRecall@GT: {recall_at_gt:.4f}")
    print(f"GT Coverage: {gt_coverage:.4f}")
    print(f"Micro Precision: {overall_micro['precision']:.4f}")
    print(f"Micro Recall: {overall_micro['recall']:.4f}")
    print(f"Micro F1: {overall_micro['f1']:.4f}")
    print(f"Macro Precision: {overall_macro['precision']:.4f}")
    print(f"Macro Recall: {overall_macro['recall']:.4f}")
    print(f"Macro F1: {overall_macro['f1']:.4f}")
    print(f"Pairs evaluated: {len(all_pair_results)}")
    print(f"Pairs skipped: {n_skipped}")

    # Write output files
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # results.json
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {results_path}")

        # per_pair_results.csv
        csv_path = output_dir / 'per_pair_results.csv'
        fieldnames = ['pair_id', 'source', 'noise_type', 'noise_param',
                      'recall_at_gt', 'gt_covered', 'gt_total', 'gt_coverage_pct',
                      'precision', 'recall', 'f1', 'tp', 'n_predicted', 'n_gt']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_pair_results)
        print(f"Per-pair results written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Schema matching evaluation using column embeddings")
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to column embeddings pickle')
    parser.add_argument('--pairs', type=str, required=True,
                        help='Path to pairs.json manifest')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground_truth.csv')
    parser.add_argument('--tables_dir', type=str, default=None,
                        help='Path to directory with table CSVs (for column name fallback)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--matching_strategy', type=str, default='hungarian',
                        choices=['hungarian', 'greedy'],
                        help='Matching strategy (default: hungarian)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Minimum cosine similarity threshold (default: 0.0)')
    args = parser.parse_args()

    evaluate_schema_matching(args)


if __name__ == '__main__':
    main()
