#!/usr/bin/env python3
"""
Combined join search and evaluation pipeline.

Runs exact cosine search followed by evaluation against ground truth.
Evaluates column-level retrieval: given a query column, find the top-K
joinable (candidate_table, candidate_column) pairs.

For LakeBench-compatible table-level metrics, use the separate
scripts/join_search_lakebench_compat.py post-hoc script.

Usage:
    python run_search_and_evaluate.py --query_emb <path> --datalake_emb <path>
"""
import os
import pickle
import sys
import argparse
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

# Resolve project root (two levels up from this script)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path for imports
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# =============================================================================
# Embedding Format Utilities
# =============================================================================

def convert_to_tuples(embeddings):
    """
    Convert embeddings to tuple format if needed.

    Handles:
    - Legacy tuple format: [(table, col, emb), ...] - normalized to basenames
    - Unified dict format: [{'table': ..., 'column_embeddings': {...}}, ...]

    Returns:
        List of tuples: [(table_name, col_idx, embedding), ...]
    """
    if not embeddings:
        return []

    first = embeddings[0]

    # Already in tuple format — normalize table names to basenames
    if isinstance(first, tuple) and len(first) == 3:
        return [(os.path.basename(table), str(col), emb) for table, col, emb in embeddings]

    # Dict format - convert to tuples
    if isinstance(first, dict):
        result = []
        column_name_cache = {}
        for item in embeddings:
            table_raw = item.get('table') or item.get('table_id') or item.get('table_name', '')
            table = os.path.basename(table_raw)

            col_emb = item.get('column_embeddings') or item.get('column_embedding', {})
            if not col_emb:
                continue

            # Always load column_names metadata when available (unified format).
            # Strip newlines from column names — OpenData CSVs sometimes have multi-line
            # headers that different parsers handle inconsistently.
            col_names = item.get('column_names')
            if col_names is not None:
                col_names = [c.replace('\n', '').replace('\r', '') for c in col_names]

            # Fall back to reading CSV header only for integer-keyed items without column_names.
            first_key = list(col_emb.keys())[0]
            if col_names is None and isinstance(first_key, (int, np.integer)):
                table_path = item.get('table', '')
                if table_path and table_path not in column_name_cache:
                    try:
                        import csv as csv_mod
                        with open(table_path, 'r', newline='', encoding='utf-8') as f:
                            column_name_cache[table_path] = [c.rstrip('\r\n') for c in next(csv_mod.reader(f))]
                    except Exception:
                        column_name_cache[table_path] = None
                col_names = column_name_cache.get(table_path)

            seen_cols = set()
            for col_idx, emb in col_emb.items():
                if isinstance(col_idx, str) and col_idx.isdigit():
                    col_idx_int = int(col_idx)
                    if col_names and col_idx_int < len(col_names):
                        col_key = col_names[col_idx_int]
                    else:
                        col_key = col_idx_int
                elif isinstance(col_idx, str):
                    col_key = col_idx
                else:
                    col_idx_int = int(col_idx)
                    if col_names and col_idx_int < len(col_names):
                        col_key = col_names[col_idx_int]
                    else:
                        col_key = col_idx_int

                # Disambiguate duplicate column names within the same table
                if col_key in seen_cols:
                    col_key = f"{col_key}_{col_idx}"
                seen_cols.add(col_key)

                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                elif isinstance(emb, np.ndarray):
                    emb = emb.astype(np.float32)

                # Ensure column key is always str (CSVs are read with dtype=str)
                result.append((table, str(col_key), emb))

        return result

    raise ValueError(f"Unknown embedding format: {type(first)}")


# =============================================================================
# Search Functions
# =============================================================================

def aggregate_to_table_level(results_df, aggregation_method, k):
    """
    Aggregate column-level results to table level.

    Aggregates across candidate_column only, preserving query_column.
    For each (query_table, query_column), returns top-k candidate tables.
    """
    grouped = results_df.groupby(['query_table', 'query_column', 'candidate_table'])

    table_results = []
    for (query_table, query_column, cand_table), group in grouped:
        num_matches = len(group)
        similarities = group['similarity'].values

        if aggregation_method == 'tabsketchfm':
            score = sum(similarities)
        elif aggregation_method == 'max':
            score = max(similarities)
        elif aggregation_method == 'mean':
            score = np.mean(similarities)
        elif aggregation_method == 'sum':
            score = sum(similarities)
        else:
            score = max(similarities)

        table_results.append({
            'query_table': query_table,
            'query_column': query_column,
            'candidate_table': cand_table,
            'similarity': score,
            'num_matches': num_matches
        })

    table_df = pd.DataFrame(table_results)

    if len(table_df) == 0:
        return pd.DataFrame(columns=['query_table', 'query_column', 'candidate_table', 'similarity'])

    if aggregation_method == 'tabsketchfm':
        table_df = table_df.sort_values(
            ['query_table', 'query_column', 'num_matches', 'similarity'],
            ascending=[True, True, False, False]
        )
    else:
        table_df = table_df.sort_values(
            ['query_table', 'query_column', 'similarity'],
            ascending=[True, True, False]
        )

    top_k_results = table_df.groupby(['query_table', 'query_column']).head(k).reset_index(drop=True)

    if 'num_matches' in top_k_results.columns:
        top_k_results = top_k_results.drop(columns=['num_matches'])

    return top_k_results


def run_search(query_emb_path, datalake_emb_path, query_list_path, k, threshold,
               aggregate_to_table=False, aggregation='tabsketchfm'):
    """
    Run exact cosine join search.

    Args:
        query_emb_path: Path to query embeddings pickle file
        datalake_emb_path: Path to data lake embeddings pickle file
        query_list_path: Path to query list CSV file
        k: Number of column-level results per query
        threshold: Minimum similarity threshold
        aggregate_to_table: Whether to aggregate primary results to table level
        aggregation: Aggregation method for --aggregate_to_table

    Returns:
        Tuple of (results_df, query_df)
    """
    print("=" * 60)
    print("Join Search (exact cosine)")
    print("=" * 60)

    # Load embeddings
    print(f"\n[1/5] Loading query embeddings...")
    with open(query_emb_path, "rb") as f:
        query_embeddings_raw = pickle.load(f)

    query_embeddings = convert_to_tuples(query_embeddings_raw)
    print(f"  Loaded {len(query_embeddings)} query column embeddings")

    print(f"\n[2/5] Loading data lake embeddings...")
    with open(datalake_emb_path, "rb") as f:
        datalake_embeddings_raw = pickle.load(f)

    datalake_embeddings = convert_to_tuples(datalake_embeddings_raw)
    print(f"  Loaded {len(datalake_embeddings)} data lake column embeddings")

    # Fail-fast: check for duplicate (table, col) keys after basename normalization.
    _dl_keys = [(table, col) for table, col, _ in datalake_embeddings]
    _dups = {k_: v for k_, v in Counter(_dl_keys).items() if v > 1}
    if _dups:
        print(f"FATAL: {len(_dups)} duplicate (table, col) keys after basename normalization:")
        for k_ in list(_dups.keys())[:5]:
            print(f"  {k_} (count={_dups[k_]})")
        sys.exit(1)

    # Create lookups
    print(f"\n[3/5] Preparing data structures...")
    query_lookup = {}
    _q_dup_count = 0
    for table, col, emb in query_embeddings:
        if (table, col) in query_lookup:
            _q_dup_count += 1
        query_lookup[(table, col)] = emb
    if _q_dup_count > 0:
        print(f"FATAL: {_q_dup_count} duplicate query (table, col) keys in embeddings")
        sys.exit(1)

    datalake_list = []
    for table, col, emb in datalake_embeddings:
        datalake_list.append((table, col, emb))

    print(f"  Query lookup: {len(query_lookup)} entries")
    print(f"  Datalake tables: {len(Counter(table for table, _, _ in datalake_list))}")

    # Build exact search index
    print(f"\n[4/5] Building exact search index...")
    if not datalake_list:
        print("ERROR: No datalake embeddings found after conversion.")
        sys.exit(1)
    N = len(datalake_list)
    dim = datalake_list[0][2].shape[0]
    print(f"  Dimension: {dim}")
    print(f"  Number of columns: {N}")

    # L2-normalize for cosine similarity via dot product
    embeddings_array = np.array([emb for _, _, emb in datalake_list], dtype=np.float32)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    zero_norm_mask = (norms.ravel() == 0)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_array /= norms
    embeddings_array[zero_norm_mask] = 0.0
    n_zero_norm = int(zero_norm_mask.sum())
    if n_zero_norm > 0:
        print(f"  Note: {n_zero_norm} zero-norm datalake vectors (excluded from results)")

    # Table name array for vectorized self-match masking
    datalake_tables = np.array([table for table, _, _ in datalake_list])

    print(f"  Exact search index ready ({embeddings_array.nbytes / 1e9:.2f} GB)")

    # Load query list and search
    print(f"\n[5/5] Running join search...")
    query_df = pd.read_csv(query_list_path, dtype={'query_table': str, 'query_column': str},
                           keep_default_na=False)
    query_df['query_table'] = query_df['query_table'].apply(os.path.basename)

    _q_df_dups = query_df.duplicated(subset=['query_table', 'query_column']).sum()
    if _q_df_dups > 0:
        print(f"FATAL: {_q_df_dups} duplicate (query_table, query_column) rows in query list")
        sys.exit(1)
    print(f"  Total queries: {len(query_df)}")

    results = []
    skipped_count = 0
    skipped_examples = []

    _RESULT_COLUMNS = ['query_table', 'query_column', 'candidate_table', 'candidate_column', 'similarity']

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Searching", ncols=80):
        query_table = row['query_table']
        query_column = row['query_column']

        key = (query_table, query_column)
        if key not in query_lookup:
            skipped_count += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(key)
            continue

        query_emb = query_lookup[key].astype(np.float32)
        qnorm = np.linalg.norm(query_emb)
        if qnorm == 0:
            skipped_count += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(key)
            continue
        query_emb = query_emb / qnorm

        # Cosine similarity = dot product of L2-normalized vectors
        similarities = embeddings_array @ query_emb

        # Mask self-table matches and zero-norm datalake vectors
        similarities[datalake_tables == query_table] = -np.inf
        if n_zero_norm > 0:
            similarities[zero_norm_mask] = -np.inf

        # Apply threshold
        if threshold > 0:
            similarities[similarities < threshold] = -np.inf

        # Top-k column results
        n_valid = int(np.sum(np.isfinite(similarities)))
        if n_valid > 0:
            actual_k = min(k, n_valid)
            if actual_k >= N:
                top_indices = np.argsort(-similarities)[:actual_k]
            else:
                top_indices = np.argpartition(-similarities, actual_k)[:actual_k]
                top_indices = top_indices[np.argsort(-similarities[top_indices])]

            for idx in top_indices:
                sim = similarities[idx]
                if not np.isfinite(sim):
                    break
                cand_table, cand_col, _ = datalake_list[idx]
                results.append({
                    'query_table': query_table,
                    'query_column': query_column,
                    'candidate_table': cand_table,
                    'candidate_column': cand_col,
                    'similarity': float(sim)
                })

    results_df = pd.DataFrame(results, columns=_RESULT_COLUMNS) if results else pd.DataFrame(columns=_RESULT_COLUMNS)

    if skipped_count:
        print(f"\n  WARNING: {skipped_count}/{len(query_df)} queries skipped (embedding not found or zero-norm)")
        for table, col in skipped_examples:
            print(f"    Missing: ({table}, {col})")
        if skipped_count > 5:
            print(f"    ... and {skipped_count - 5} more")

    # Apply table-level aggregation to primary results if requested
    if aggregate_to_table:
        print(f"\n  Aggregating primary results to table level (method: {aggregation})...")
        results_df = aggregate_to_table_level(results_df, aggregation, k)
        print(f"  Aggregated to {len(results_df)} table-level results")

    print("=" * 60)
    print("SEARCH COMPLETE")
    print(f"   Total results: {len(results_df)}")
    if len(results_df) > 0:
        sims = results_df['similarity']
        print(f"   Similarity range: [{sims.min():.4f}, {sims.max():.4f}] (mean: {sims.mean():.4f})")
    if not aggregate_to_table and len(query_df) > 0:
        print(f"   Avg results/query (incl. skipped): {len(results_df) / len(query_df):.1f}")
    print("=" * 60)

    return results_df, query_df


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_evaluation(results_df, ground_truth_path, k_values, table_level=False):
    """
    Evaluate join search results against ground truth.

    Args:
        results_df: DataFrame with search results
        ground_truth_path: Path to ground truth CSV file
        k_values: List of K values for evaluation
        table_level: Whether to evaluate at table level (ignore candidate_column)

    Returns:
        Dict of metrics: {k: {'precision': float, 'recall': float, 'f1': float}, 'map': float}
    """
    results_table_level = 'candidate_column' not in results_df.columns
    effective_table_level = table_level or results_table_level

    level_tag = "TBL" if effective_table_level else "COL"

    print("\n" + "=" * 60)
    print(f"Evaluation ({'Table-level' if effective_table_level else 'Column-level'})")
    print("=" * 60)

    if results_table_level and not table_level:
        print(f"\n  Results have no candidate_column. Forced table-level evaluation.")

    if table_level and not results_table_level:
        print(f"\n  Results contain candidate_column but --table_level specified.")
        print(f"  Will ignore candidate_column for evaluation.")

    # Load ground truth
    print(f"\n[1/3] Loading ground truth...")
    _gt_str_cols = {c: str for c in ['query_table', 'query_column', 'candidate_table', 'candidate_column']}
    gt_df = pd.read_csv(ground_truth_path, dtype=_gt_str_cols, keep_default_na=False)
    print(f"  Ground truth columns: {list(gt_df.columns)}")
    print(f"  Ground truth: {len(gt_df)} pairs")

    # Normalize table names to basenames
    gt_df['query_table'] = gt_df['query_table'].apply(os.path.basename)
    gt_df['candidate_table'] = gt_df['candidate_table'].apply(os.path.basename)

    # Normalize results table names (defensive: search produces basenames, but --eval_only CSVs may not)
    results_df = results_df.copy()
    results_df['query_table'] = results_df['query_table'].apply(os.path.basename)
    results_df['candidate_table'] = results_df['candidate_table'].apply(os.path.basename)

    # Filter self-table pairs from GT
    gt_self_mask = gt_df['query_table'] == gt_df['candidate_table']
    gt_self_count = gt_self_mask.sum()
    if gt_self_count > 0:
        print(f"  Filtered {gt_self_count} self-table pairs from ground truth")
        gt_df = gt_df[~gt_self_mask].reset_index(drop=True)

    # Prepare data structures
    print(f"\n[2/3] Preparing data structures...")

    results_df = results_df.sort_values(
        ['query_table', 'query_column', 'similarity'],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    gt_lookup = {}
    if effective_table_level:
        for _, row in gt_df.iterrows():
            key = (row['query_table'], row['query_column'])
            if key not in gt_lookup:
                gt_lookup[key] = set()
            gt_lookup[key].add(row['candidate_table'])
        print(f"  Ground truth covers {len(gt_lookup)} unique query columns")
        if gt_lookup:
            print(f"  Average candidate tables per query: {np.mean([len(v) for v in gt_lookup.values()]):.1f}")
    else:
        for _, row in gt_df.iterrows():
            key = (row['query_table'], row['query_column'])
            if key not in gt_lookup:
                gt_lookup[key] = set()
            gt_lookup[key].add((row['candidate_table'], row['candidate_column']))
        print(f"  Ground truth covers {len(gt_lookup)} unique query columns")
        if gt_lookup:
            print(f"  Average candidate column-pairs per query: {np.mean([len(v) for v in gt_lookup.values()]):.1f}")

    results_grouped = dict(list(results_df.groupby(['query_table', 'query_column'])))
    print(f"  Results cover {len(results_grouped)} unique query columns")

    # Calculate metrics
    print(f"\n[3/3] Calculating metrics...")

    metrics = {k: {'precisions': [], 'recalls': [], 'f1s': []} for k in k_values}
    aps = []  # Average Precision per query (k-independent)

    gt_queries_with_results = 0
    gt_queries_without_results = 0
    results_not_in_gt = 0

    for query_key, gt_set in gt_lookup.items():
        group = results_grouped.get(query_key)

        if group is None:
            gt_queries_without_results += 1
            for k in k_values:
                metrics[k]['precisions'].append(0.0)
                metrics[k]['recalls'].append(0.0)
                metrics[k]['f1s'].append(0.0)
            aps.append(0.0)
            continue

        gt_queries_with_results += 1

        if effective_table_level:
            result_items = list(group['candidate_table'])
            seen = set()
            result_items_unique = []
            for item in result_items:
                if item not in seen:
                    seen.add(item)
                    result_items_unique.append(item)
            result_items = result_items_unique
        else:
            result_items = list(zip(group['candidate_table'], group['candidate_column']))

        for k in k_values:
            topk_items = set(result_items[:k])
            intersection = topk_items.intersection(gt_set)

            precision = len(intersection) / min(k, len(result_items)) if result_items else 0
            recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[k]['precisions'].append(precision)
            metrics[k]['recalls'].append(recall)
            metrics[k]['f1s'].append(f1)

        # Average Precision: AP = (1/|relevant|) * sum(P@i * rel(i))
        # Computed over the full result list (not truncated to any K)
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
    print("EVALUATION RESULTS")
    print("=" * 60)

    final_metrics = {}

    for k in k_values:
        if len(metrics[k]['precisions']) == 0:
            print(f"\n{'─' * 30}")
            print(f"K = {k}")
            print(f"{'─' * 30}")
            print(f"  No data to evaluate!")
            final_metrics[k] = {'precision': 0, 'recall': 0, 'f1': 0}
            continue

        avg_precision = np.mean(metrics[k]['precisions'])
        avg_recall = np.mean(metrics[k]['recalls'])
        avg_f1 = np.mean(metrics[k]['f1s'])

        print(f"\n{'─' * 30}")
        print(f"K = {k}")
        print(f"{'─' * 30}")
        print(f"  {level_tag} Precision@{k}: {avg_precision:.4f} ({avg_precision * 100:.2f}%)")
        print(f"  {level_tag} Recall@{k}:    {avg_recall:.4f} ({avg_recall * 100:.2f}%)")
        print(f"  {level_tag} F1@{k}:        {avg_f1:.4f} ({avg_f1 * 100:.2f}%)")

        final_metrics[k] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }

    # MAP (Mean Average Precision) — k-independent
    map_value = np.mean(aps) if aps else 0.0
    print(f"\n{'─' * 30}")
    print(f"  {level_tag} MAP: {map_value:.4f} ({map_value * 100:.2f}%)")
    final_metrics['map'] = map_value

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
    print(f"  Result queries not in GT:{results_not_in_gt} (excluded from metrics)")

    return final_metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run join search and evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Search arguments
    parser.add_argument("--query_list", type=str,
                        default=os.path.join(_PROJECT_ROOT, "datasets/opendata/queries/opendata_join/opendata_join_query.csv"),
                        help="Path to query list CSV file")
    parser.add_argument("--query_emb", type=str, default=None,
                        help="Path to query embeddings pickle file (e.g., embeddings/column/<model>/<dataset>.pkl)")
    parser.add_argument("--datalake_emb", type=str, default=None,
                        help="Path to data lake embeddings pickle file (typically same as --query_emb)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_PROJECT_ROOT, "results/evaluation/join_search/results.csv"),
                        help="Path to output results CSV file")
    parser.add_argument("--k", type=int, default=50,
                        help="Return top-K results per query for search")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Minimum similarity threshold")
    parser.add_argument("--aggregate_to_table", action="store_true",
                        help="Aggregate primary results to table level (instead of column level)")
    parser.add_argument("--aggregation", type=str, default="tabsketchfm",
                        choices=["tabsketchfm", "max", "mean", "sum"],
                        help="Aggregation method for --aggregate_to_table")

    # Evaluation arguments
    parser.add_argument("--ground_truth", type=str,
                        default=os.path.join(_PROJECT_ROOT, "datasets/opendata/gt/opendata_join_ground_truth.csv"),
                        help="Path to ground truth CSV file")
    parser.add_argument("--k_values", type=int, nargs='+',
                        default=[10, 20, 50],
                        help="K values for evaluation metrics")

    # Pipeline control
    parser.add_argument("--search_only", action="store_true",
                        help="Run search only, skip evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only (requires --results)")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to existing results CSV (for --eval_only mode)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments
    if args.eval_only and args.search_only:
        print("Error: Cannot specify both --eval_only and --search_only")
        sys.exit(1)
    if args.eval_only and args.aggregate_to_table:
        print("Error: --eval_only --aggregate_to_table is not supported."
              " Run search with --aggregate_to_table to produce aggregated results.")
        sys.exit(1)

    # Embedding paths are required when running search
    if not args.eval_only:
        if not args.query_emb or not args.datalake_emb:
            print("Error: --query_emb and --datalake_emb are required (unless using --eval_only)")
            sys.exit(1)

    # Enforce search depth >= evaluation depth
    if not args.eval_only and not args.search_only:
        max_k_eval = max(args.k_values)
        if args.k < max_k_eval:
            print(f"Error: --k ({args.k}) must be >= max(--k_values) ({max_k_eval})")
            sys.exit(1)

    if args.eval_only:
        # Load existing results
        results_path = args.results or args.output
        print(f"Loading existing results from: {results_path}")
        _id_str = {c: str for c in ['query_table', 'query_column', 'candidate_table', 'candidate_column']}
        results_df = pd.read_csv(results_path, dtype=_id_str, keep_default_na=False)
        results_df['query_table'] = results_df['query_table'].apply(os.path.basename)
        results_df['candidate_table'] = results_df['candidate_table'].apply(os.path.basename)
    else:
        # Run search
        results_df, query_df = run_search(
            query_emb_path=args.query_emb,
            datalake_emb_path=args.datalake_emb,
            query_list_path=args.query_list,
            k=args.k,
            threshold=args.threshold,
            aggregate_to_table=args.aggregate_to_table,
            aggregation=args.aggregation
        )

        # Save results
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    if not args.search_only:
        # Evaluate
        metrics = run_evaluation(
            results_df=results_df,
            ground_truth_path=args.ground_truth,
            k_values=args.k_values,
            table_level=args.aggregate_to_table
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
