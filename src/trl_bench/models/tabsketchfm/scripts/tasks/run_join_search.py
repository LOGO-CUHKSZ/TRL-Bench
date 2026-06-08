"""
Task-agnostic join search.

Implements the 3-stage join search algorithm from the paper (Figure 6):
1. KNNSEARCH: Find k×3 nearest columns for each query column
2. COLUMNNEARESTTABLES: Group by table, track matching columns
3. RANK: Sort by (# matches DESC, distance sum ASC)

This script takes pre-extracted embeddings and a ground truth file,
performs join search, and evaluates the results.

Usage:
    python scripts/tasks/run_join_search.py \
        --embeddings embeddings/wiki_join_embeddings.pkl \
        --ground_truth wiki-join-search/labels/join_search_jaccard_gt.jsonl \
        --k 10 \
        --output_dir results/wiki_join_search

    # Or with legacy pickle format
    python scripts/tasks/run_join_search.py \
        --embeddings embeddings/wiki_join_embeddings.pkl \
        --ground_truth embeddings/wiki_join_ground_truth.pkl \
        --ground_truth_format pickle \
        --k 10 \
        --output_dir results/wiki_join_search
"""

import os
import sys
import pickle
import json
import numpy as np
import faiss
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def extract_table_id(table_path):
    """
    Extract table ID from table path.

    Examples:
        'wiki-join-search/tables/60EBA0LLPD95.csv' -> '60EBA0LLPD95'
        'path/to/table.csv' -> 'table'
        'tablename' -> 'tablename'
    """
    # Get basename and remove extension
    basename = os.path.basename(table_path)
    if basename.endswith('.csv'):
        return basename[:-4]
    return basename


def load_embeddings(embeddings_file):
    """
    Load embeddings from pickle file.

    Expected format:
        [{
            'table': 'table_name.csv' or path/to/table.csv,
            'cls_embedding': [768],
            'table_embedding': [768],
            'column_embedding': {0: [768], 1: [768], ...}
        }, ...]

    Returns:
        col_to_embedding: dict mapping 'table_id_col' to embedding vector
        col_order: list of column keys in order
        table_id_to_full_path: dict mapping table_id to original path
    """
    print(f"\n📂 Loading embeddings from: {embeddings_file}")
    with open(embeddings_file, 'rb') as f:
        table_embeddings = pickle.load(f)

    col_to_embedding = {}
    table_id_to_full_path = {}

    for item in table_embeddings:
        table_path = item['table']
        table_id = extract_table_id(table_path)
        table_id_to_full_path[table_id] = table_path

        # Extract all columns using format: table_id_col_index
        # Handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
        col_emb_data = item.get('column_embeddings') or item.get('column_embedding', {})
        for col_id, col_emb in col_emb_data.items():
            col_key = f"{table_id}_{col_id}"
            col_to_embedding[col_key] = col_emb

    col_order = list(col_to_embedding.keys())

    # Count unique tables
    unique_tables = len(set(extract_table_id(item['table']) for item in table_embeddings))

    print(f"   Loaded {len(table_embeddings)} entries ({unique_tables} unique tables)")
    print(f"   Total columns: {len(col_to_embedding)}")

    return col_to_embedding, col_order, table_id_to_full_path


def load_ground_truth_jsonl(jsonl_file, min_score=0.0):
    """
    Load ground truth from JSONL file.

    Format:
        {"query_column": "table1.csv:0", "target_column": "table2.csv:1", "jaccard_score": 0.75}

    Returns:
        dict mapping query_column to list of target_columns
    """
    print(f"\n📂 Loading ground truth from: {jsonl_file}")

    ground_truth = defaultdict(list)

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            # Handle wiki-join-search format: {"source": {...}, "joinable_list": [...]}
            if 'source' in data and 'joinable_list' in data:
                source_file = data['source']['filename']
                source_col = data['source']['col']
                query_col = f"{source_file}_{source_col}"

                for target in data['joinable_list']:
                    target_file = target['filename']
                    target_col_idx = target['col']
                    score = target.get('score', 1.0)

                    if score >= min_score:
                        target_col = f"{target_file}_{target_col_idx}"
                        ground_truth[query_col].append(target_col)

            # Handle old format: {"query_column": ..., "target_column": ...}
            elif 'query_column' in data and 'target_column' in data:
                query_col = data['query_column']
                target_col = data['target_column']
                score = data.get('jaccard_score', 1.0)

                if score >= min_score:
                    ground_truth[query_col].append(target_col)

            else:
                raise ValueError(f"Unknown ground truth format: {list(data.keys())}")

    # Convert to regular dict
    ground_truth = dict(ground_truth)

    num_queries = len(ground_truth)
    total_targets = sum(len(targets) for targets in ground_truth.values())
    avg_targets = total_targets / num_queries if num_queries > 0 else 0

    print(f"   Queries: {num_queries}")
    print(f"   Total target columns: {total_targets}")
    print(f"   Avg targets per query: {avg_targets:.2f}")

    return ground_truth


def load_ground_truth_pickle(pickle_file):
    """Load ground truth from pickle file."""
    print(f"\n📂 Loading ground truth from: {pickle_file}")

    with open(pickle_file, 'rb') as f:
        ground_truth = pickle.load(f)

    num_queries = len(ground_truth)
    total_targets = sum(len(targets) for targets in ground_truth.values())
    avg_targets = total_targets / num_queries if num_queries > 0 else 0

    print(f"   Queries: {num_queries}")
    print(f"   Total target columns: {total_targets}")
    print(f"   Avg targets per query: {avg_targets:.2f}")

    return ground_truth


def column_to_table(column_key):
    """
    Extract table ID from column key 'table_id_col'.

    Examples:
        '60EBA0LLPD95_0' -> '60EBA0LLPD95'
        '60EBA0LLPD95_12' -> '60EBA0LLPD95'
    """
    # Split by underscore and take everything except the last part (column index)
    parts = column_key.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return column_key


def build_faiss_index(embeddings_dict, col_order, use_gpu=False):
    """
    Build FAISS index for fast nearest neighbor search.

    Args:
        embeddings_dict: dict mapping column keys to embeddings
        col_order: list of column keys in order
        use_gpu: whether to use GPU acceleration

    Returns:
        FAISS index
    """
    print("\n🔧 Building FAISS index...")

    # Stack embeddings into matrix
    embeddings_matrix = np.array([embeddings_dict[col] for col in col_order])
    embeddings_matrix = embeddings_matrix.astype('float32')

    # Normalize for cosine similarity (using inner product on normalized vectors)
    faiss.normalize_L2(embeddings_matrix)

    # Build index
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity on normalized vectors)

    if use_gpu and faiss.get_num_gpus() > 0:
        num_gpus = faiss.get_num_gpus()
        print(f"   Using GPU acceleration ({num_gpus} GPUs)")

        # Use all available GPUs for better performance
        if num_gpus > 1:
            # Replicate index across all GPUs (faster search)
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            # Single GPU
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("   Using CPU")

    index.add(embeddings_matrix)

    print(f"   Indexed {index.ntotal} columns in {dim}-dimensional space")

    return index, embeddings_matrix


def join_search_single_query(query_col, col_to_embedding, col_order, index, embeddings_matrix, k=10):
    """
    Perform join search for a single query column using the paper's 3-stage algorithm.

    Stage 1 (KNNSEARCH): Find k×3 nearest columns
    Stage 2 (COLUMNNEARESTTABLES): Group by table, track matches
    Stage 3 (RANK): Sort by (# matches DESC, distance sum ASC)

    Args:
        query_col: query column key 'table:col'
        col_to_embedding: dict mapping column keys to embeddings
        col_order: list of column keys
        index: FAISS index
        embeddings_matrix: normalized embedding matrix
        k: number of top results to return

    Returns:
        List of (table_name, score) tuples, ranked
    """
    query_table = column_to_table(query_col)

    # Stage 1: KNNSEARCH - find k×3 nearest columns
    if query_col not in col_to_embedding:
        return []

    query_idx = col_order.index(query_col)
    query_emb = embeddings_matrix[query_idx:query_idx+1]

    n_neighbors = min(k * 3, index.ntotal)
    distances, indices = index.search(query_emb, n_neighbors)

    # Convert distances (inner product) to distances (1 - inner_product)
    distances = 1.0 - distances[0]
    indices = indices[0]

    # Stage 2: COLUMNNEARESTTABLES - group by table
    table_to_cols = defaultdict(list)

    for dist, idx in zip(distances, indices):
        col_key = col_order[idx]
        table_name = column_to_table(col_key)

        # Skip self-matches
        if table_name == query_table:
            continue

        col_id = col_key.rsplit('_', 1)[1] if '_' in col_key else '0'
        table_to_cols[table_name].append((col_id, dist))

    # For each table, compute: (num_matching_cols, sum_of_distances)
    table_scores = {}
    for table_name, col_dists in table_to_cols.items():
        # Sort by distance to ensure we keep the best match per column
        col_dists_sorted = sorted(col_dists, key=lambda x: x[1])

        # Ensure each column is counted only once
        seen_cols = set()
        unique_dists = []
        for col, dist in col_dists_sorted:
            if col not in seen_cols:
                seen_cols.add(col)
                unique_dists.append(dist)

        num_cols = len(unique_dists)
        sum_dist = sum(unique_dists)
        table_scores[table_name] = (num_cols, sum_dist)

    # Stage 3: RANK - sort by (# matches DESC, distance sum ASC)
    ranked_tables = sorted(
        table_scores.items(),
        key=lambda x: (-x[1][0], x[1][1])
    )

    return [(table, score) for table, score in ranked_tables[:k]]


def convert_to_table_level_ground_truth(column_level_gt):
    """
    Convert column-level ground truth to table-level.

    Input: {query_col: [target_col1, target_col2, ...]}
    Output: {query_col: [target_table1, target_table2, ...]}
    """
    table_level_gt = {}

    for query_col, target_cols in column_level_gt.items():
        target_tables = set()
        for target_col in target_cols:
            table_name = column_to_table(target_col)
            target_tables.add(table_name)
        table_level_gt[query_col] = list(target_tables)

    return table_level_gt


def evaluate_join_search(results, ground_truth, k_values=[1, 5, 10]):
    """
    Evaluate join search results at table level.

    Computes:
    - Mean F1 score
    - Precision@k
    - Recall@k

    Args:
        results: dict mapping query_col to list of (table, score) tuples
        ground_truth: dict mapping query_col to list of relevant tables
        k_values: list of k values to evaluate

    Returns:
        dict with metrics
    """
    metrics = {}

    for k in k_values:
        f1_scores = []
        precisions = []
        recalls = []

        for query_col in ground_truth:
            if query_col not in results:
                # No results for this query
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            # Get top-k predicted tables
            predicted = [table for table, score in results[query_col][:k]]
            relevant = set(ground_truth[query_col])

            if len(relevant) == 0:
                continue

            # Compute metrics
            true_positives = len(set(predicted) & relevant)

            precision = true_positives / len(predicted) if len(predicted) > 0 else 0.0
            recall = true_positives / len(relevant) if len(relevant) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        metrics[k] = {
            'mean_f1': np.mean(f1_scores),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'num_queries': len(f1_scores)
        }

    return metrics


def print_metrics(metrics, k_values=[1, 5, 10]):
    """Print metrics in a nice table format."""
    print()
    print("=" * 80)
    print("JOIN SEARCH EVALUATION METRICS (Table Level)")
    print("=" * 80)

    # Header
    print(f"{'Metric':<15}", end="")
    for k in k_values:
        print(f"{'@' + str(k):>12}", end="")
    print()
    print("-" * (15 + 12 * len(k_values)))

    # Rows
    for metric_name in ['mean_f1', 'precision', 'recall']:
        print(f"{metric_name.replace('_', ' ').title():<15}", end="")
        for k in k_values:
            value = metrics[k][metric_name]
            print(f"{value:>12.4f}", end="")
        print()

    # Num queries
    print()
    print(f"{'Queries':<15}", end="")
    for k in k_values:
        num = metrics[k]['num_queries']
        print(f"{num:>12}", end="")
    print()
    print("=" * 80)
    print()


def main():
    parser = ArgumentParser(description="Task-agnostic join search")
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Pickle file with table embeddings')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Ground truth file (JSONL or pickle)')
    parser.add_argument('--ground_truth_format', type=str, default='jsonl',
                        choices=['jsonl', 'pickle'],
                        help='Format of ground truth file')
    parser.add_argument('--min_score', type=float, default=0.5,
                        help='Minimum similarity score threshold for ground truth pairs (default: 0.5 for Jaccard, aligned with TabSketchFM paper)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top results to return')
    parser.add_argument('--k_values', type=str, default='1,5,10',
                        help='Comma-separated k values for evaluation')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for FAISS indexing')

    args = parser.parse_args()

    # Parse k_values
    k_values = [int(x) for x in args.k_values.split(',')]
    k_values = sorted(set([v for v in k_values if v <= args.k]))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    col_to_embedding, col_order, table_id_to_full_path = load_embeddings(args.embeddings)

    # Load ground truth
    if args.ground_truth_format == 'jsonl':
        column_gt = load_ground_truth_jsonl(args.ground_truth, args.min_score)
    else:
        column_gt = load_ground_truth_pickle(args.ground_truth)

    # Convert to table-level ground truth
    print("\n🔄 Converting to table-level ground truth...")
    table_gt = convert_to_table_level_ground_truth(column_gt)

    num_queries = len(table_gt)
    total_tables = sum(len(tables) for tables in table_gt.values())
    avg_tables = total_tables / num_queries if num_queries > 0 else 0
    print(f"   Queries: {num_queries}")
    print(f"   Total target tables: {total_tables}")
    print(f"   Avg target tables per query: {avg_tables:.2f}")

    # Build FAISS index
    index, embeddings_matrix = build_faiss_index(col_to_embedding, col_order, args.use_gpu)

    # Perform join search for all queries
    print(f"\n🔍 Performing join search (k={args.k})...")
    results = {}

    for query_col in tqdm(table_gt.keys(), desc="Searching"):
        ranked_tables = join_search_single_query(
            query_col, col_to_embedding, col_order, index, embeddings_matrix, k=args.k
        )
        results[query_col] = ranked_tables

    # Save results
    results_file = output_dir / 'search_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to: {results_file}")

    # Evaluate
    print("\n📊 Evaluating results...")
    metrics = evaluate_join_search(results, table_gt, k_values)

    # Print metrics
    print_metrics(metrics, k_values)

    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_file}")

    print(f"\n✅ Join search complete!")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    main()
