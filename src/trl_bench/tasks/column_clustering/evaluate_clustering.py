#!/usr/bin/env python3
"""
Standalone clustering evaluation for VizNet column embeddings.
Loads pre-computed embeddings and evaluates clustering quality.

This script implements the column clustering algorithm from the Starmie paper
(VLDB 2023) with correct purity computation (micro-average, weighted by cluster size).

Usage:
    python evaluate_clustering.py \
        --embeddings cl_datalake_drop_col_head_column_0.pkl \
        --dataset sato \
        --k 20 \
        --target_avg_size 50
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from collections import Counter, deque
from tqdm import tqdm


def blocked_matmul(mata, matb, threshold=None, k=None, batch_size=512,
                   exclude_self=False):
    """Find the most similar pairs of vectors from two matrices (top-k or threshold).

    Args:
        mata (np.ndarray): First matrix of shape (n, d)
        matb (np.ndarray): Second matrix of shape (m, d)
        threshold (float, optional): If set, return all pairs with similarity >= threshold
        k (int, optional): If set, return top-k most similar vectors for each row in matb
        batch_size (int, optional): Batch size for memory-efficient computation
        exclude_self (bool): If True, filter pairs where idx_a == idx_b (for self-similarity)

    Returns:
        list of tuples: Each tuple is (idx_a, idx_b, similarity)
    """
    mata = np.array(mata)
    matb = np.array(matb)
    results = []

    # Guard against k >= n (argpartition requires kth < n)
    if k is not None:
        max_k = mata.shape[0] - (1 if exclude_self else 0)
        if min(k, max_k) <= 0:
            return results

    for start in tqdm(range(0, len(matb), batch_size), desc="Computing similarities"):
        block = matb[start:start+batch_size]
        sim_mat = np.matmul(mata, block.transpose())

        if k is not None:
            n = sim_mat.shape[0]
            # Fetch k+1 candidates when excluding self, so we still get k true neighbors
            k_fetch = min(k + 1, n) if exclude_self else min(k, n)
            kth = min(k_fetch, n) - 1  # argpartition requires kth < n
            indices = np.argpartition(-sim_mat, kth, axis=0)
            # Track per-column (idx_b) counts to cap at exactly k non-self neighbors
            col_counts = {}
            for row in indices[:k_fetch]:
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    if exclude_self and idx_a == idx_b:
                        continue
                    cnt = col_counts.get(idx_b, 0)
                    if cnt >= k:
                        continue
                    col_counts[idx_b] = cnt + 1
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
        elif threshold is not None:
            indices = np.argwhere(sim_mat >= threshold)
            for idx_a, idx_b in indices:
                idx_b += start
                if exclude_self and idx_a == idx_b:
                    continue
                results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))

    return results


def connected_components(pairs, cluster_size=50):
    """Compute connected components with maximum cluster size limit.

    Args:
        pairs (list): List of (idx_a, idx_b, similarity) tuples
        cluster_size (int): Maximum size of each cluster

    Returns:
        list of lists: Each inner list contains column indices in a cluster
    """
    # Build adjacency list (sorted by similarity for deterministic results)
    edges = {}
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

    for left, right, _ in pairs_sorted:
        if left not in edges:
            edges[left] = []
        if right not in edges:
            edges[right] = []
        edges[left].append(right)
        edges[right].append(left)

    # Find connected components with size limit
    all_ccs = []
    used = set()

    for start in edges:
        if start in used:
            continue

        used.add(start)
        cc = [start]
        queue = deque([start])

        while len(queue) > 0:
            u = queue.popleft()
            for v in edges[u]:
                if v not in used:
                    cc.append(v)
                    used.add(v)
                    queue.append(v)
                    if len(cc) >= cluster_size:
                        break
            if len(cc) >= cluster_size:
                break

        all_ccs.append(cc)

    return all_ccs


def compute_nmi_ari(clusters, labels):
    """Compute NMI and ARI for clustering evaluation.

    Converts list-of-lists clusters into flat predicted/true label arrays
    (only for columns present in clusters) and delegates to sklearn.

    Args:
        clusters (list of lists): Cluster assignments (list of column indices per cluster)
        labels: Semantic type label for each column

    Returns:
        tuple: (nmi, ari) scores
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    pred_labels = []
    true_labels = []
    for cluster_id, cc in enumerate(clusters):
        for column_id in cc:
            pred_labels.append(cluster_id)
            true_labels.append(labels[column_id])

    if len(pred_labels) == 0:
        return 0.0, 0.0

    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ari = adjusted_rand_score(true_labels, pred_labels)
    return nmi, ari


def compute_purity(clusters, labels):
    """Compute clustering purity (micro-average, weighted by cluster size).

    Purity = sum of dominant label counts / total columns
    This is the CORRECT definition from the Starmie paper.

    Args:
        clusters (list of lists): Cluster assignments (list of column indices per cluster)
        labels: Semantic type label for each column

    Returns:
        float: Purity score (0 to 1, higher is better)
    """
    correct = 0
    total = 0

    for cc in clusters:
        cnt = Counter()
        for column_id in cc:
            label = labels[column_id]
            cnt[label] += 1
        correct += cnt.most_common(1)[0][1]
        total += len(cc)

    return correct / total if total > 0 else 0.0


def tune_cluster_size(pairs, labels, target=50):
    """Binary search to find cluster_size that achieves target avg_cluster_size.

    The Starmie paper tunes max_cluster_size to achieve avg_cluster_size ≈ 50
    for fair comparison across different embedding methods.

    Args:
        pairs (list): Similarity pairs from blocked_matmul
        labels: Semantic type labels
        target (float): Target average cluster size (default: 50)

    Returns:
        tuple: (clusters, purity, best_cluster_size)
    """
    left = 0
    right = 5000
    min_diff = float('inf')
    best_ccs = []
    best_size = 50

    print(f"  Tuning cluster_size to achieve avg_size ≈ {target}...")

    if not pairs:
        return [], 0.0, best_size

    while right - left > 10:
        mid = (left + right) // 2
        ccs = connected_components(pairs, cluster_size=mid)
        avg_size = np.mean([len(cc) for cc in ccs])

        if abs(avg_size - target) < min_diff:
            min_diff = abs(avg_size - target)
            best_ccs = ccs
            best_size = mid

        if avg_size > target:
            right = mid
        else:
            left = mid

    purity = compute_purity(best_ccs, labels)
    return best_ccs, purity, best_size


def load_embeddings(embeddings_path, dataset=None):
    """Load embeddings from supported formats and labels from dataset.

    Args:
        embeddings_path: Path to embeddings pickle file (.pkl)
                        Supported formats:
                          1) Starmie tuples: [(filename, embeddings_array), ...]
                          2) Unified v2.0 dicts: [{'table': ..., 'table_id': ..., 'column_embeddings': {...}}, ...]
        dataset: Dataset name (e.g., 'sato' or 'sotab') to load labels from {dataset}/all.csv

    Returns:
        embeddings (np.ndarray): Column embeddings (N, embedding_dim)
        labels (np.ndarray): Ground truth labels (N,)
    """
    import os
    import re

    def _sort_key(idx):
        try:
            return int(idx)
        except Exception:
            return str(idx)

    def _stack_column_embeddings(col_embeddings):
        if isinstance(col_embeddings, dict):
            sorted_indices = sorted(col_embeddings.keys(), key=_sort_key)
            vectors = [col_embeddings[idx] for idx in sorted_indices]
            return np.array(vectors)
        return np.array(col_embeddings)

    def _add_table_mapping(mapping, key, value):
        if key is None:
            return
        mapping[key] = value

    print(f"Loading embeddings from: {embeddings_path}")

    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected list of embeddings, got {type(data)}")

    first = data[0]
    is_unified = isinstance(first, dict) and (
        "column_embeddings" in first or "column_embedding" in first
    )
    is_starmie = isinstance(first, tuple) and len(first) == 2

    if not (is_unified or is_starmie):
        raise ValueError(
            "Unsupported embeddings format. Expected Starmie tuples or unified v2.0 dicts."
        )

    print(f"  Format: {'unified v2.0 dicts' if is_unified else 'starmie tuples'}")
    print(f"  Tables: {len(data)}")

    # Build table_id/table_name -> embeddings mapping
    table_embeddings_map = {}

    if is_starmie:
        for filename, table_emb in data:
            base_name = os.path.basename(filename)
            _add_table_mapping(table_embeddings_map, base_name, table_emb)
            match = re.match(r'table_(\d+)\.csv', base_name)
            if match:
                table_id = int(match.group(1))
                _add_table_mapping(table_embeddings_map, table_id, table_emb)
                _add_table_mapping(table_embeddings_map, str(table_id), table_emb)
    else:
        for item in data:
            col_embeddings = item.get('column_embeddings') or item.get('column_embedding')
            if col_embeddings is None:
                continue
            table_emb = _stack_column_embeddings(col_embeddings)

            table_id = item.get('table_id')
            table_path = item.get('table', '')
            table_name = os.path.basename(table_path) if table_path else None
            alt_table_name = item.get('table_name')

            _add_table_mapping(table_embeddings_map, table_id, table_emb)
            _add_table_mapping(table_embeddings_map, str(table_id) if table_id is not None else None, table_emb)
            _add_table_mapping(table_embeddings_map, table_name, table_emb)
            _add_table_mapping(table_embeddings_map, alt_table_name, table_emb)

            if isinstance(table_name, str):
                match = re.match(r'table_(\d+)\.csv', table_name)
                if match:
                    num_id = int(match.group(1))
                    _add_table_mapping(table_embeddings_map, num_id, table_emb)
                    _add_table_mapping(table_embeddings_map, str(num_id), table_emb)

    print(f"  Mapped {len(table_embeddings_map)} table keys")

    # Load labels from dataset
    if dataset is None:
        raise ValueError("--dataset argument required to load labels")

    labels_path = f'{dataset}/all.csv'
    print(f"\nLoading labels from: {labels_path}")

    test_df = pd.read_csv(labels_path)
    print(f"  Labels loaded: {len(test_df)}")
    print(f"  Unique classes: {test_df['class'].nunique()}")

    # Build embeddings and labels in the SAME ORDER as test.csv
    # test.csv has (table_id, column_id, class) - we need to match this order
    all_embeddings = []
    all_labels = []
    missing_tables = set()
    col_mismatch = 0

    for _, row in test_df.iterrows():
        table_id = row['table_id']
        column_id = row['column_id']
        label = row['class']

        # Resolve table embedding by trying multiple key variants
        key_candidates = [table_id]
        if isinstance(table_id, float) and table_id.is_integer():
            key_candidates.append(int(table_id))
        if not isinstance(table_id, str):
            key_candidates.append(str(table_id))
        if isinstance(table_id, str) and not table_id.endswith('.csv'):
            key_candidates.append(f"{table_id}.csv")
        if isinstance(table_id, (int, np.integer)):
            key_candidates.append(f"table_{int(table_id)}.csv")

        table_emb = None
        for key in key_candidates:
            if key in table_embeddings_map:
                table_emb = table_embeddings_map[key]
                break

        if table_emb is None:
            missing_tables.add(table_id)
            continue

        if column_id >= table_emb.shape[0]:
            col_mismatch += 1
            continue

        all_embeddings.append(table_emb[column_id])
        all_labels.append(label)

    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels)

    expected = len(test_df)
    matched = len(all_embeddings)
    coverage_pct = matched / expected if expected > 0 else 0.0

    coverage_info = {
        'expected_columns': expected,
        'matched_columns': matched,
        'coverage_pct': coverage_pct,
        'missing_tables_count': len(missing_tables),
        'col_mismatches': col_mismatch,
    }

    print(f"\n  Final embeddings shape: {embeddings.shape}")
    print(f"  Final labels count: {len(labels)}")
    print(f"  Coverage:              {matched}/{expected} ({coverage_pct:.6f})")
    print(f"  Missing tables:        {len(missing_tables)}")
    print(f"  Column mismatches:     {col_mismatch}")

    if matched == expected:
        print(f"  ✓ All rows matched successfully!")

    return embeddings, labels, coverage_info


def evaluate_clustering_from_embeddings(embeddings, labels, k=20, target_avg_size=50,
                                        batch_size=4096, coverage_info=None):
    """Evaluate column clustering on pre-computed embeddings.

    This implements the Starmie paper's clustering algorithm:
    1. L2 normalize vectors for cosine similarity
    2. Find top-k similar columns
    3. Tune cluster_size to achieve target avg_cluster_size
    4. Compute purity (micro-average, weighted by cluster size)

    Args:
        embeddings (np.ndarray): Column embeddings (N, dim)
        labels (np.ndarray): Ground truth labels (N,)
        k (int): Number of nearest neighbors to find
        target_avg_size (float): Target average cluster size (tuned via binary search)
        batch_size (int): Batch size for similarity computation
        coverage_info (dict, optional): Coverage metadata from load_embeddings

    Returns:
        Dict: Clustering metrics (num_clusters, avg_cluster_size, purity)
    """
    print("\n" + "="*60)
    print("STARMIE COLUMN CLUSTERING EVALUATION")
    print("="*60)

    # Step 1: L2 normalize vectors for cosine similarity
    print("\n[1/4] Normalizing vectors (L2)...")
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms
    print(f"  Shape: {embeddings.shape}")
    print(f"  Unique labels: {len(np.unique(labels))}")

    # Step 2: Find top-k similar columns
    print(f"\n[2/4] Finding top-{k} similar columns...")
    pairs = blocked_matmul(embeddings, embeddings, k=k, batch_size=batch_size,
                           exclude_self=True)
    print(f"  Total pairs: {len(pairs):,}")

    # Step 3: Tune cluster size and compute clusters
    print(f"\n[3/4] Running connected components clustering...")
    clusters, purity, best_cluster_size = tune_cluster_size(pairs, labels, target=target_avg_size)

    # Step 4: Compute final metrics
    print(f"\n[4/4] Computing metrics...")
    avg_cluster_size = np.mean([len(cc) for cc in clusters]) if clusters else 0.0

    nmi, ari = compute_nmi_ari(clusters, labels)

    results = {
        "num_clusters": len(clusters),
        "avg_cluster_size": avg_cluster_size,
        "purity": purity,
        "nmi": nmi,
        "ari": ari,
        "total_columns": len(labels),
        "unique_labels": len(set(labels)),
        "k": k,
        "tuned_max_cluster_size": best_cluster_size
    }

    if coverage_info is not None:
        results['coverage_pct'] = coverage_info['coverage_pct']
        results['expected_columns'] = coverage_info['expected_columns']
        results['matched_columns'] = coverage_info['matched_columns']
        results['missing_tables_count'] = coverage_info['missing_tables_count']
        results['col_mismatches'] = coverage_info['col_mismatches']

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Number of clusters:    {results['num_clusters']:,}")
    print(f"  Avg cluster size:      {results['avg_cluster_size']:.2f}")
    print(f"  Purity:                {results['purity']:.4f} ({results['purity']*100:.2f}%)")
    print(f"  NMI:                   {results['nmi']:.4f}")
    print(f"  ARI:                   {results['ari']:.4f}")
    print(f"  Total columns:         {results['total_columns']:,}")
    print(f"  Unique semantic types: {results['unique_labels']}")
    print(f"  k (nearest neighbors): {results['k']}")
    print(f"  Tuned max_cluster_size:{results['tuned_max_cluster_size']}")
    if coverage_info is not None:
        ci = coverage_info
        print(f"  Coverage:              {ci['matched_columns']}/{ci['expected_columns']} ({ci['coverage_pct']:.6f})")
        print(f"  Missing tables:        {ci['missing_tables_count']}")
        print(f"  Column mismatches:     {ci['col_mismatches']}")
    print("="*60)

    return results, clusters


def analyze_clusters(clusters, labels, top_n=5):
    """Print analysis of top clusters.

    Args:
        clusters (list): List of cluster assignments
        labels (np.ndarray): Ground truth labels
        top_n (int): Number of top clusters to analyze
    """
    print("\n" + "="*60)
    print(f"Top {top_n} largest clusters:")
    print("="*60)

    # Sort by cluster size
    sorted_clusters = sorted(clusters, key=len, reverse=True)

    for i, cluster in enumerate(sorted_clusters[:top_n]):
        cnt = Counter()
        for col_id in cluster:
            cnt[labels[col_id]] += 1

        print(f"\nCluster {i+1} (size={len(cluster)}):")
        for label, count in cnt.most_common():
            pct = count / len(cluster) * 100
            print(f"  {label}: {count} columns ({pct:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate clustering on pre-computed embeddings (Starmie paper algorithm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
    # Evaluate on sato dataset
    python evaluate_clustering.py \\
        --embeddings cl_datalake_drop_col_head_column_0.pkl \\
        --dataset sato \\
        --k 20 \\
        --target_avg_size 50

    # With detailed cluster analysis
    python evaluate_clustering.py \\
        --embeddings cl_datalake_drop_col_head_column_0.pkl \\
        --dataset sato \\
        --analyze
        """
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to embeddings pickle file (.pkl) from extractVectors.py'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g. sato, sotab) to load labels from {dataset}/all.csv'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='Number of nearest neighbors for similarity graph (paper default: 20)'
    )
    parser.add_argument(
        '--target_avg_size',
        type=float,
        default=50,
        help='Target average cluster size (tuned via binary search, paper default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4096,
        help='Batch size for similarity computation'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Print detailed cluster analysis'
    )
    parser.add_argument(
        '--min_coverage',
        type=float,
        default=0.95,
        help='Minimum coverage threshold (0-1). Exit with error if coverage is below this.'
    )

    args = parser.parse_args()

    # Load embeddings and labels
    embeddings, labels, coverage_info = load_embeddings(args.embeddings, args.dataset)

    # Coverage gate
    if coverage_info['coverage_pct'] < args.min_coverage:
        print(f"\nERROR: Coverage {coverage_info['coverage_pct']:.6f} is below "
              f"minimum threshold {args.min_coverage:.6f}")
        sys.exit(1)

    # Evaluate clustering
    results, clusters = evaluate_clustering_from_embeddings(
        embeddings,
        labels,
        k=args.k,
        target_avg_size=args.target_avg_size,
        batch_size=args.batch_size,
        coverage_info=coverage_info
    )

    # Optional: detailed analysis
    if args.analyze:
        analyze_clusters(clusters, labels, top_n=10)
