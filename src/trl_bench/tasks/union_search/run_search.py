#!/usr/bin/env python3
"""
Table Union Search Script

Evaluates column embeddings for table union search using bipartite column
matching (Munkres/Hungarian algorithm). Inherited from Starmie (PVLDB 2023).

Supports linear (exact) and HNSW (approximate) search methods.
Linear is the default and recommended method for benchmark publication.

Usage:
    # Linear search (exact, default for benchmark)
    python run_search.py \
        --query_embeddings embeddings.pkl \
        --datalake_embeddings embeddings.pkl \
        --groundtruth datasets/santos/santosUnionBenchmark.pickle \
        --method linear --K 10

    # HNSW search (approximate, faster for large datalakes)
    python run_search.py \
        --query_embeddings embeddings.pkl \
        --datalake_embeddings embeddings.pkl \
        --groundtruth datasets/santos/santosUnionBenchmark.pickle \
        --method hnsw --ef 100 --N 100 --K 10
"""

import argparse
import os
import pickle
import time
import numpy as np
from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm


def load_embeddings(path):
    """Load embeddings from pickle, supporting both legacy and unified v2 formats.

    Legacy format: list of (name, np.array) tuples
    Unified v2 format: list of dicts with 'table', 'column_embeddings', etc.

    Table names are normalized to basenames for consistent GT matching.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Detect unified v2 format (list of dicts with column_embeddings)
    if data and isinstance(data[0], dict) and 'column_embeddings' in data[0]:
        result = []
        for entry in data:
            # Use basename of table path (e.g. "data_mill_a.csv") to match
            # groundtruth keys which include the .csv extension.
            name = os.path.basename(entry['table'])
            col_embs = entry['column_embeddings']
            # Numeric-aware sorting: digit-string keys sorted numerically, others alphabetically
            sorted_keys = sorted(col_embs.keys(),
                                 key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
            embeddings = np.stack([col_embs[k] for k in sorted_keys])
            result.append((name, embeddings))
        return result
    # Legacy format: list of (name, array) tuples — normalize to basenames
    return [(os.path.basename(name), emb) for name, emb in data]


def cosine_sim(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    assert vec1.ndim == vec2.ndim
    n1, n2 = norm(vec1), norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (n1 * n2)


def verify_table_match(table1, table2, threshold):
    """Compute bipartite matching score between two tables.

    Uses the Munkres (Hungarian) algorithm to find optimal column alignment.
    Only column pairs with cosine similarity above threshold are considered.
    """
    nrow = len(table1)
    ncol = len(table2)

    # Guard: empty tables
    if nrow == 0 or ncol == 0:
        return 0.0

    graph = np.zeros(shape=(nrow, ncol), dtype=float)

    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i], table2[j])
            if sim > threshold:
                graph[i, j] = sim

    # If all similarities are below threshold, no matching needed
    if graph.max() == 0:
        return 0.0

    max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
    m = Munkres()
    indexes = m.compute(max_graph)

    score = 0.0
    for row, col in indexes:
        score += graph[row, col]
    return score


def search_linear(queries, datalake, K=10, threshold=0.7):
    """Linear search: compute bipartite matching scores for all query-datalake pairs.

    This is the exact (non-approximate) search method and matches Starmie's
    primary evaluation path (NaiveSearcher.topk).
    """
    results = {}
    query_times = []

    for qi, (query_name, query_embeddings) in enumerate(queries):
        start_time = time.time()
        scores = []

        for table_name, table_embeddings in datalake:
            score = verify_table_match(query_embeddings, table_embeddings, threshold)
            scores.append((score, table_name))

        # Sort by score and take top-K
        scores.sort(reverse=True)
        results[query_name] = [name for score, name in scores[:K]]
        elapsed = time.time() - start_time
        query_times.append(elapsed)

        if (qi + 1) % 10 == 0 or (qi + 1) == len(queries):
            print(f"  Query {qi+1}/{len(queries)} ({elapsed:.1f}s)")

    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    return results, avg_query_time


def search_hnsw(queries, datalake, K=10, threshold=0.7, ef=100, N=100):
    """
    HNSW search: build index on datalake columns and retrieve candidates.

    Uses HNSW to find candidate tables via column-level nearest neighbors,
    then runs bipartite matching on candidates only. Faster than linear
    but approximate — self-columns may consume neighbor budget.

    Args:
        queries: list of (name, embeddings) tuples for query tables
        datalake: list of (name, embeddings) tuples for datalake tables
        K: number of top results to return
        threshold: column similarity threshold for bipartite matching
        ef: HNSW search quality parameter
        N: number of columns to retrieve per query column
    """
    import hnswlib

    # Build HNSW index on all datalake columns
    print(f"Building HNSW index with ef={ef}, N={N}...")
    index_start_time = time.time()

    # Flatten all datalake columns
    all_columns = []
    col_table_ids = []
    for table_idx, (table_name, table_embeddings) in enumerate(datalake):
        for col_embedding in table_embeddings:
            all_columns.append(col_embedding)
            col_table_ids.append(table_idx)

    if not all_columns:
        print("ERROR: No datalake columns found.")
        return {}, 0

    vec_dim = len(all_columns[0])

    # Initialize HNSW index
    index = hnswlib.Index(space='cosine', dim=vec_dim)
    index.init_index(max_elements=len(all_columns), ef_construction=100, M=32)
    index.set_ef(ef)
    index.add_items(all_columns)

    print(f"Index built in {time.time() - index_start_time:.2f}s ({len(all_columns)} columns)")

    # Search for each query
    results = {}
    query_times = []

    # Clamp N to available columns
    actual_N = min(N, len(all_columns))

    for query_name, query_embeddings in queries:
        start_time = time.time()

        # Find candidate tables using HNSW index
        table_ids = set()
        labels, _ = index.knn_query(query_embeddings, k=actual_N)
        for result in labels:
            for idx in result:
                table_ids.add(col_table_ids[idx])

        # Get candidate tables
        candidates = [(datalake[tid][0], datalake[tid][1]) for tid in table_ids]

        # Compute bipartite matching scores for candidates
        scores = []
        for table_name, table_embeddings in candidates:
            score = verify_table_match(query_embeddings, table_embeddings, threshold)
            scores.append((score, table_name))

        # Sort by score and take top-K
        scores.sort(reverse=True)
        results[query_name] = [name for score, name in scores[:K]]
        query_times.append(time.time() - start_time)

    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    return results, avg_query_time


def compute_metrics(results, groundtruth, K=10):
    """
    Compute MAP, Precision, and Recall using Starmie's method.

    MAP = mean of P@1, P@2, ..., P@K (Starmie's definition, different from
    standard IR MAP which averages precision at relevant document positions).

    Precision and recall are macro-averaged across queries.
    Extension stripping uses os.path.splitext for safe comparison.
    """
    precision_at_k = []
    recall_at_k = []

    # Count GT coverage
    gt_queries_evaluated = 0
    gt_queries_missing = 0

    for k in range(1, K + 1):
        precisions = []
        recalls = []

        for query_name, retrieved in results.items():
            if query_name not in groundtruth:
                if k == 1:
                    gt_queries_missing += 1
                continue

            if k == 1:
                gt_queries_evaluated += 1

            # Compare using stem names (strip extension safely)
            relevant = set(os.path.splitext(x)[0] for x in groundtruth[query_name])
            retrieved_k = [os.path.splitext(x)[0] for x in retrieved[:k]]

            # Find intersection (true positives)
            tp = len(set(retrieved_k) & relevant)

            # Precision and recall at k
            precision = tp / k
            recall = tp / len(relevant) if len(relevant) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)

        # Average across all evaluated queries for this k
        precision_at_k.append(sum(precisions) / len(precisions) if precisions else 0.0)
        recall_at_k.append(sum(recalls) / len(recalls) if recalls else 0.0)

    # MAP is the mean of precisions at k=1,2,...,K (Starmie's definition)
    map_score = sum(precision_at_k) / K if precision_at_k else 0.0

    return map_score, precision_at_k, recall_at_k, gt_queries_evaluated, gt_queries_missing


def main():
    parser = argparse.ArgumentParser(description='Table Union Search')

    # Embedding files (required)
    parser.add_argument('--query_embeddings', type=str, required=True,
                       help='Path to query embeddings pickle file')
    parser.add_argument('--datalake_embeddings', type=str, required=True,
                       help='Path to datalake embeddings pickle file')

    # Ground truth (optional)
    parser.add_argument('--groundtruth', type=str, default=None,
                       help='Path to ground truth pickle file (optional, for evaluation)')

    # Search parameters
    parser.add_argument('--method', type=str, default='linear', choices=['linear', 'hnsw'],
                       help='Search method: linear (exact) or hnsw (approximate)')
    parser.add_argument('--K', type=int, default=10,
                       help='Number of top results to return')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Column similarity threshold')

    # HNSW-specific parameters
    parser.add_argument('--ef', type=int, default=100,
                       help='HNSW search quality parameter (default: 100)')
    parser.add_argument('--N', type=int, default=100,
                       help='HNSW: number of columns retrieved per query column (default: 100)')

    args = parser.parse_args()

    # =========================================================================
    # Load embeddings
    # =========================================================================
    print("=" * 60)
    print("Table Union Search")
    print("=" * 60)

    print(f"\nLoading embeddings...")
    print(f"  Query: {args.query_embeddings}")
    print(f"  Datalake: {args.datalake_embeddings}")

    queries = load_embeddings(args.query_embeddings)
    datalake = load_embeddings(args.datalake_embeddings)

    if not queries:
        print("ERROR: No query embeddings loaded.")
        return
    if not datalake:
        print("ERROR: No datalake embeddings loaded.")
        return

    print(f"  Loaded {len(queries)} query tables and {len(datalake)} datalake tables")

    # Check embedding dimension consistency
    dims = set()
    for _, emb in queries[:5]:
        if len(emb) > 0:
            dims.add(emb.shape[-1])
    for _, emb in datalake[:5]:
        if len(emb) > 0:
            dims.add(emb.shape[-1])
    if len(dims) > 1:
        print(f"  WARNING: Mixed embedding dimensions: {dims}")
    elif dims:
        print(f"  Embedding dimension: {dims.pop()}")

    # =========================================================================
    # Load ground truth and filter queries (if GT provided)
    # =========================================================================
    groundtruth = None
    if args.groundtruth:
        print(f"\nLoading ground truth from: {args.groundtruth}")
        with open(args.groundtruth, 'rb') as f:
            groundtruth = pickle.load(f)
        print(f"  GT entries: {len(groundtruth)} queries")

        # Filter queries to only those with GT entries
        # (matches Starmie's separate query pkl — avoids searching non-GT tables)
        query_names = set(groundtruth.keys())
        original_count = len(queries)
        queries = [(name, emb) for name, emb in queries if name in query_names]
        print(f"  Filtered queries to {len(queries)}/{original_count} with ground truth entries")

        if not queries:
            print("ERROR: No query tables found in ground truth — check name format.")
            print(f"  Sample query names: {[q[0] for q in load_embeddings(args.query_embeddings)[:3]]}")
            print(f"  Sample GT keys: {list(groundtruth.keys())[:3]}")
            return

        # Check GT target coverage in datalake
        datalake_names = set(name for name, _ in datalake)
        all_gt_targets = set()
        for targets in groundtruth.values():
            all_gt_targets.update(os.path.splitext(t)[0] for t in targets)
        datalake_stems = set(os.path.splitext(name)[0] for name in datalake_names)
        missing_targets = all_gt_targets - datalake_stems
        if missing_targets:
            print(f"  WARNING: {len(missing_targets)} GT target tables not found in datalake")
            for t in list(missing_targets)[:3]:
                print(f"    Missing: {t}")

    # =========================================================================
    # Search
    # =========================================================================
    print(f"\nSearch Configuration:")
    print(f"  Method: {args.method}")
    print(f"  K: {args.K}")
    print(f"  Threshold: {args.threshold}")
    if args.method == 'hnsw':
        print(f"  ef: {args.ef}")
        print(f"  N: {args.N} (columns per query column)")
    print()

    print(f"Performing {args.method} search...")
    start_time = time.time()

    if args.method == 'linear':
        results, avg_query_time = search_linear(queries, datalake, args.K, args.threshold)
    elif args.method == 'hnsw':
        results, avg_query_time = search_hnsw(queries, datalake, args.K, args.threshold, args.ef, args.N)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    total_time = time.time() - start_time

    print(f"\nSearch completed!")
    print(f"  Average query time: {avg_query_time:.4f}s")
    print(f"  Total time: {total_time:.2f}s")

    # =========================================================================
    # Evaluate
    # =========================================================================
    if groundtruth:
        print(f"\nEvaluating results...")
        map_score, precision_at_k, recall_at_k, gt_evaluated, gt_missing = \
            compute_metrics(results, groundtruth, args.K)

        print(f"\nResults:")
        print(f"  MAP@{args.K}  = {map_score:.4f}")
        print(f"  P@{args.K}    = {precision_at_k[-1]:.4f}")
        print(f"  R@{args.K}    = {recall_at_k[-1]:.4f}")

        # Detailed per-K metrics
        print(f"\nDetailed Metrics:")
        for k in range(1, min(args.K, 10) + 1):
            print(f"  P@{k:2d} = {precision_at_k[k-1]:.4f}, R@{k:2d} = {recall_at_k[k-1]:.4f}")

        print(f"\nStatistics:")
        print(f"  GT queries evaluated: {gt_evaluated}")
        print(f"  Result queries not in GT: {gt_missing}")
    else:
        print(f"\nNo ground truth provided. Skipping evaluation.")
        print(f"To evaluate results, add: --groundtruth /path/to/groundtruth.pickle")

    print("\nDone!")


if __name__ == '__main__':
    main()
