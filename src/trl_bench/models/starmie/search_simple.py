#!/usr/bin/env python3
"""
Simplified Table Union Search Script
This script only needs embedding files and search parameters - no training metadata!

Usage:
    # Linear search (exact, slow)
    python search_simple.py \
        --query_embeddings data/santos/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl \
        --datalake_embeddings data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl \
        --method linear

    # HNSW search (approximate, fast) with optimal settings
    python search_simple.py \
        --query_embeddings data/santos/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl \
        --datalake_embeddings data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl \
        --method hnsw --ef 100 --N 100

    # With ground truth for evaluation
    python search_simple.py \
        --query_embeddings data/santos/vectors/cl_query_drop_col_tfidf_entity_column_0.pkl \
        --datalake_embeddings data/santos/vectors/cl_datalake_drop_col_tfidf_entity_column_0.pkl \
        --groundtruth data/santos/santosUnionBenchmark.pickle \
        --method hnsw --ef 100 --N 100
"""

import argparse
import pickle
import time
import numpy as np
import hnswlib
from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm

def cosine_sim(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    assert vec1.ndim == vec2.ndim
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def verify_table_match(table1, table2, threshold):
    """Compute bipartite matching score between two tables"""
    score = 0.0
    nrow = len(table1)
    ncol = len(table2)
    graph = np.zeros(shape=(nrow, ncol), dtype=float)

    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i], table2[j])
            if sim > threshold:
                graph[i, j] = sim

    max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
    m = Munkres()
    indexes = m.compute(max_graph)
    for row, col in indexes:
        score += graph[row, col]
    return score

def search_linear(queries, datalake, K=10, threshold=0.7):
    """Linear search: compute scores for all query-datalake pairs"""
    results = {}
    query_times = []

    for query_name, query_embeddings in queries:
        start_time = time.time()
        scores = []

        for table_name, table_embeddings in datalake:
            score = verify_table_match(query_embeddings, table_embeddings, threshold)
            scores.append((score, table_name))

        # Sort by score and take top-K
        scores.sort(reverse=True)
        results[query_name] = [name for score, name in scores[:K]]
        query_times.append(time.time() - start_time)

    avg_query_time = sum(query_times) / len(query_times)
    return results, avg_query_time

def search_hnsw(queries, datalake, K=10, threshold=0.7, ef=100, N=100):
    """
    HNSW search: build index on datalake columns and retrieve candidates

    Args:
        queries: list of (name, embeddings) tuples for query tables
        datalake: list of (name, embeddings) tuples for datalake tables
        K: number of top results to return
        threshold: column similarity threshold for bipartite matching
        ef: HNSW search quality parameter
        N: number of columns to retrieve per query column (CRITICAL parameter!)
    """
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

    # Get vector dimension
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

    for query_name, query_embeddings in queries:
        start_time = time.time()

        # Find candidate tables using HNSW index
        table_ids = set()
        labels, _ = index.knn_query(query_embeddings, k=N)
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

    avg_query_time = sum(query_times) / len(query_times)
    return results, avg_query_time

def compute_metrics(results, groundtruth, K=10):
    """
    Compute MAP, Precision, and Recall using Starmie's method.

    Note: This matches the paper's calculation where MAP = mean of P@1, P@2, ..., P@K
    (different from standard IR MAP definition).
    """
    # Compute precision and recall at each k from 1 to K
    precision_at_k = []
    recall_at_k = []

    for k in range(1, K + 1):
        precisions = []
        recalls = []

        for query_name, retrieved in results.items():
            if query_name not in groundtruth:
                continue

            # Remove file extensions for comparison (match original code)
            relevant = set([x.split('.')[0] for x in groundtruth[query_name]])
            retrieved_k = [x.split('.')[0] for x in retrieved[:k]]

            # Find intersection (true positives)
            tp = len(set(retrieved_k) & relevant)

            # Precision at k
            if k > 0:
                precision = tp / k
            else:
                precision = 0.0

            # Recall at k
            if len(relevant) > 0:
                recall = tp / len(relevant)
            else:
                recall = 0.0

            precisions.append(precision)
            recalls.append(recall)

        # Average across all queries for this k
        precision_at_k.append(sum(precisions) / len(precisions) if precisions else 0.0)
        recall_at_k.append(sum(recalls) / len(recalls) if recalls else 0.0)

    # MAP is the mean of precisions at k=1,2,...,K (paper's definition)
    map_score = sum(precision_at_k) / K if precision_at_k else 0.0

    return map_score, precision_at_k, recall_at_k


def main():
    parser = argparse.ArgumentParser(description='Simplified Table Union Search')

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
                       help='HNSW: number of columns retrieved per query column (default: 100, CRITICAL!)')

    args = parser.parse_args()

    # Load embeddings
    query_path = args.query_embeddings
    datalake_path = args.datalake_embeddings

    print(f"Loading embedding files:")
    print(f"  Query: {query_path}")
    print(f"  Datalake: {datalake_path}")

    print(f"\nLoading embeddings...")
    with open(query_path, 'rb') as f:
        queries = pickle.load(f)
    with open(datalake_path, 'rb') as f:
        datalake = pickle.load(f)

    print(f"Loaded {len(queries)} query tables and {len(datalake)} datalake tables")

    # Search parameters
    print(f"\nSearch Configuration:")
    print(f"  Method: {args.method}")
    print(f"  K: {args.K}")
    print(f"  Threshold: {args.threshold}")
    if args.method == 'hnsw':
        print(f"  ef: {args.ef}")
        print(f"  N: {args.N} (columns per query column)")
    print()

    # Perform search
    print(f"Performing {args.method} search...")
    start_time = time.time()

    if args.method == 'linear':
        results, avg_query_time = search_linear(queries, datalake, args.K, args.threshold)
    elif args.method == 'hnsw':
        results, avg_query_time = search_hnsw(queries, datalake, args.K, args.threshold, args.ef, args.N)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    total_time = time.time() - start_time

    print(f"Search completed!")
    print(f"  Average query time: {avg_query_time:.4f}s")
    print(f"  Total time: {total_time:.2f}s")

    # Evaluate if ground truth provided
    if args.groundtruth:
        print(f"\nLoading ground truth from: {args.groundtruth}")
        with open(args.groundtruth, 'rb') as f:
            groundtruth = pickle.load(f)

        print(f"Evaluating results...")
        map_score, precision_at_k, recall_at_k = compute_metrics(results, groundtruth, args.K)

        print(f"\nResults:")
        print(f"  MAP@{args.K}  = {map_score:.4f}")
        print(f"  P@{args.K}    = {precision_at_k[-1]:.4f}")
        print(f"  R@{args.K}    = {recall_at_k[-1]:.4f}")

        # Detailed per-K metrics
        print(f"\nDetailed Metrics:")
        for k in range(1, min(args.K, 10) + 1):
            print(f"  P@{k:2d} = {precision_at_k[k-1]:.4f}, R@{k:2d} = {recall_at_k[k-1]:.4f}")
    else:
        print(f"\nNo ground truth provided. Skipping evaluation.")
        print(f"To evaluate results, add: --groundtruth /path/to/groundtruth.pickle")

    print("\nDone!")

if __name__ == '__main__':
    main()
