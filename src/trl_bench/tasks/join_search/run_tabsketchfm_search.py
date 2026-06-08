#!/usr/bin/env python3
"""
Join search using TabSketchFM methodology with integrated evaluation.

This script performs end-to-end join search:
1. Load embeddings in standard format: [(table.csv, column_name, embedding), ...]
2. Load ground truth from CSV with optional score filtering
3. Build FAISS/HNSW index for efficient similarity search
4. Execute TabSketchFM 3-stage search algorithm
5. Evaluate and report metrics (F1, Precision, Recall)

TabSketchFM Algorithm (arXiv:2407.01619):
- Stage 1 (KNNSEARCH): Find k×3 nearest columns using cosine similarity
- Stage 2 (COLUMNNEARESTTABLES): Group by table, track unique column matches
- Stage 3 (RANK): Sort by (# matches DESC, distance sum ASC)

Usage:
    python run_tabsketchfm_search.py \\
        --query_emb /path/to/query_embeddings.pkl \\
        --datalake_emb /path/to/datalake_embeddings.pkl \\
        --ground_truth /path/to/ground_truth.csv \\
        --min_score 0.5 \\
        --k 10 \\
        --use_gpu
"""

import os
import pickle
import json
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm

# Check available index libraries
FAISS_AVAILABLE = False
HNSW_AVAILABLE = False
GPU_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
    GPU_AVAILABLE = faiss.get_num_gpus() > 0
except ImportError:
    pass

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    pass

if not FAISS_AVAILABLE and not HNSW_AVAILABLE:
    raise ImportError("Either faiss or hnswlib is required. Install with: pip install faiss-cpu or pip install hnswlib")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Join search using TabSketchFM methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with GPU
  python run_tabsketchfm_search.py \\
      --query_emb embeddings/query.pkl \\
      --datalake_emb embeddings/datalake.pkl \\
      --ground_truth ground_truth.csv \\
      --use_gpu

  # With score filtering and custom k values
  python run_tabsketchfm_search.py \\
      --query_emb embeddings/query.pkl \\
      --datalake_emb embeddings/datalake.pkl \\
      --ground_truth ground_truth.csv \\
      --min_score 0.5 \\
      --k 10 \\
      --k_values 1,5,10
        """
    )

    # Required arguments
    parser.add_argument("--query_emb", type=str, required=True,
                        help="Path to query embeddings pickle file")
    parser.add_argument("--datalake_emb", type=str, required=True,
                        help="Path to datalake embeddings pickle file")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth CSV file")

    # Search parameters
    parser.add_argument("--k", type=int, default=10,
                        help="Number of top results to return (default: 10)")
    parser.add_argument("--k_values", type=str, default="1,5,10",
                        help="Comma-separated k values for evaluation (default: 1,5,10)")
    parser.add_argument("--min_score", type=float, default=0.0,
                        help="Minimum score for ground truth filtering (default: 0.0)")

    # Index options
    parser.add_argument("--index_type", type=str, default="faiss",
                        choices=["faiss", "hnsw"],
                        help="Index type: 'faiss' (exact, GPU support) or 'hnsw' (approximate, faster for large datasets)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for FAISS index (requires --index_type faiss)")

    # Output options
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for metrics JSON")
    parser.add_argument("--output_results", type=str, default=None,
                        help="Output file for search results pickle")

    return parser.parse_args()


def get_column_names_from_csv(csv_path):
    """
    Read column names from original CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        list: Column names in order
    """
    import csv

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            return [col.rstrip('\r\n') for col in header]
    except Exception as e:
        # Try with different encoding or return None
        try:
            with open(csv_path, 'r', newline='', encoding='latin-1') as f:
                reader = csv.reader(f)
                header = next(reader)
                return [col.rstrip('\r\n') for col in header]
        except:
            return None


def load_embeddings(embeddings_file):
    """
    Load embeddings from pickle file.

    Supports two formats:
    1. Tuple format: [(table.csv, column_name, embedding), ...]
    2. Dict format: [{table, cls_embedding, table_embedding, column_embedding: {0: [...]}}, ...]

    Returns:
        dict: {table_id_column_name: embedding}
    """
    print(f"Loading embeddings from: {embeddings_file}")
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    embeddings = {}

    # Detect format - handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
    if len(data) > 0 and isinstance(data[0], dict) and ('column_embeddings' in data[0] or 'column_embedding' in data[0]):
        # Dict format (supports both unified 'column_embeddings' and legacy 'column_embedding')
        print("   Detected dict format embeddings")
        column_name_cache = {}  # Cache column names per table

        for item in tqdm(data, desc="Processing"):
            table_path = item.get('table', '')
            # Handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
            col_embeddings = item.get('column_embeddings') or item.get('column_embedding', {})

            # Prefer 'table_id' (canonical) if available, otherwise extract from path
            table_id = item.get('table_id')
            if not table_id:
                table_basename = os.path.basename(table_path)
                table_id = table_basename[:-4] if table_basename.endswith('.csv') else table_basename

            # Get column names from original CSV
            if table_path not in column_name_cache:
                column_names = get_column_names_from_csv(table_path)
                column_name_cache[table_path] = column_names

            column_names = column_name_cache[table_path]

            # Add each column embedding
            for col_idx, emb in col_embeddings.items():
                col_idx_int = int(col_idx) if isinstance(col_idx, str) and col_idx.isdigit() else col_idx
                if column_names and isinstance(col_idx_int, (int, np.integer)) and col_idx_int < len(column_names):
                    col_name = column_names[col_idx_int]
                else:
                    # Fallback to index if column names not available
                    col_name = str(col_idx)

                key = f"{table_id}\t{col_name}"
                embeddings[key] = np.array(emb)
    else:
        # Old tuple format: [(table.csv, column_name, embedding), ...]
        print("   Detected tuple format embeddings")
        for table_name, col_name, emb in tqdm(data, desc="Processing"):
            # Remove .csv extension from table name
            table_id = table_name[:-4] if table_name.endswith('.csv') else table_name
            # Strip carriage return from column names (Windows line ending issue)
            col_name_clean = col_name.rstrip('\r\n')
            key = f"{table_id}\t{col_name_clean}"
            embeddings[key] = np.array(emb)

    print(f"   Loaded {len(embeddings)} column embeddings")
    return embeddings


def load_ground_truth_csv(csv_file, min_score=0.0):
    """
    Load ground truth from CSV file.

    Expected format: query_table,candidate_table,query_column,candidate_column,score

    Returns:
        dict: {query_key: [(target_key, score), ...]}
    """
    import csv

    print(f"Loading ground truth from: {csv_file}")
    print(f"   Filtering with min_score >= {min_score}")

    ground_truth = defaultdict(list)
    total_pairs = 0
    filtered_pairs = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_pairs += 1
            score = float(row['score']) if 'score' in row else 1.0

            if score >= min_score:
                # Build keys using table_id_column_name format
                query_table = row['query_table']
                query_table = query_table[:-4] if query_table.endswith('.csv') else query_table

                cand_table = row['candidate_table']
                cand_table = cand_table[:-4] if cand_table.endswith('.csv') else cand_table

                query_key = f"{query_table}\t{row['query_column']}"
                target_key = f"{cand_table}\t{row['candidate_column']}"

                ground_truth[query_key].append((target_key, score))
                filtered_pairs += 1

    # Sort targets by score descending
    for key in ground_truth:
        ground_truth[key].sort(key=lambda x: -x[1])

    num_queries = len(ground_truth)
    avg_targets = filtered_pairs / num_queries if num_queries > 0 else 0

    print(f"   Total pairs in file: {total_pairs:,}")
    print(f"   Pairs after filtering: {filtered_pairs:,}")
    print(f"   Queries with targets: {num_queries:,}")
    print(f"   Avg targets per query: {avg_targets:.2f}")

    return ground_truth


def column_to_table(column_key):
    """Extract table ID from column key 'table_id\\tcolumn_name'."""
    idx = column_key.find('\t')
    if idx > 0:
        return column_key[:idx]
    return column_key


def convert_to_table_level_ground_truth(column_level_gt):
    """Convert column-level ground truth to table-level."""
    table_level_gt = {}
    for query_col, targets in column_level_gt.items():
        target_tables = set()
        for target_col, score in targets:
            table_name = column_to_table(target_col)
            target_tables.add(table_name)
        table_level_gt[query_col] = list(target_tables)
    return table_level_gt


def build_index(col_to_embedding, col_order, index_type="faiss", use_gpu=False):
    """
    Build search index using FAISS or HNSW.

    Args:
        col_to_embedding: dict mapping column keys to embeddings
        col_order: list of column keys in order
        index_type: 'faiss' or 'hnsw'
        use_gpu: whether to use GPU for FAISS

    Returns:
        tuple: (index, embeddings_matrix, index_type)
    """
    print("Building search index...")

    embeddings_matrix = np.array([col_to_embedding[col] for col in col_order])
    embeddings_matrix = embeddings_matrix.astype('float32')

    dim = embeddings_matrix.shape[1]
    print(f"   Dimension: {dim}")
    print(f"   Number of columns: {len(col_order):,}")

    if index_type == "faiss":
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS requested but not installed. Install with: pip install faiss-cpu or faiss-gpu")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        index = faiss.IndexFlatIP(dim)

        if use_gpu and GPU_AVAILABLE:
            print(f"   Moving index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"   Using FAISS GPU (IndexFlatIP) - exact search")
        elif use_gpu and not GPU_AVAILABLE:
            print(f"   WARNING: GPU requested but not available, using CPU")
            print(f"   Using FAISS CPU (IndexFlatIP) - exact search")
        else:
            print(f"   Using FAISS CPU (IndexFlatIP) - exact search")

        index.add(embeddings_matrix)
    else:  # hnsw
        if not HNSW_AVAILABLE:
            raise ImportError("HNSW requested but not installed. Install with: pip install hnswlib")

        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=len(col_order), ef_construction=200, M=32)
        index.set_ef(100)
        index.add_items(embeddings_matrix, np.arange(len(col_order)))
        print(f"   Using HNSW (hnswlib) - approximate search")

    return index, embeddings_matrix, index_type


def search_single_query(query_col, col_to_embedding, col_order, index, embeddings_matrix, index_type="faiss", k=10):
    """
    Perform TabSketchFM 3-stage join search for a single query.

    Stage 1: KNNSEARCH - Find k×3 nearest columns
    Stage 2: COLUMNNEARESTTABLES - Group by table, track matches
    Stage 3: RANK - Sort by (# matches DESC, distance sum ASC)

    Returns:
        list: [(table_id, (num_matches, sum_dist)), ...]
    """
    query_table = column_to_table(query_col)

    if query_col not in col_to_embedding:
        return []

    query_idx = col_order.index(query_col)
    n_neighbors = min(k * 3, len(col_order))

    # Stage 1: KNNSEARCH
    if index_type == "faiss":
        query_emb = embeddings_matrix[query_idx:query_idx+1]
        similarities, indices = index.search(query_emb, n_neighbors)
        distances = 1.0 - similarities[0]  # Convert similarity to distance
        indices = indices[0]
    else:  # hnsw
        query_emb = embeddings_matrix[query_idx]
        indices, distances = index.knn_query(query_emb.reshape(1, -1), k=n_neighbors)
        distances = distances[0]
        indices = indices[0]

    # Stage 2: COLUMNNEARESTTABLES - Group by table
    table_to_cols = defaultdict(list)
    for dist, idx in zip(distances, indices):
        col_key = col_order[idx]
        table_name = column_to_table(col_key)

        # Skip self-matches (same table)
        if table_name == query_table:
            continue

        table_to_cols[table_name].append((col_key, dist))

    # Compute (num_matching_cols, sum_of_distances) per table
    table_scores = {}
    for table_name, col_dists in table_to_cols.items():
        # Sort by distance and keep unique columns
        col_dists_sorted = sorted(col_dists, key=lambda x: x[1])
        seen_cols = set()
        unique_dists = []
        for col, dist in col_dists_sorted:
            if col not in seen_cols:
                seen_cols.add(col)
                unique_dists.append(dist)

        num_cols = len(unique_dists)
        sum_dist = sum(unique_dists)
        table_scores[table_name] = (num_cols, sum_dist)

    # Stage 3: RANK - Sort by (# matches DESC, distance sum ASC)
    ranked_tables = sorted(
        table_scores.items(),
        key=lambda x: (-x[1][0], x[1][1])
    )

    return [(table, score) for table, score in ranked_tables[:k]]


def evaluate(results, ground_truth, k_values):
    """
    Evaluate join search results at table level.

    Args:
        results: dict {query_col: [(table, score), ...]}
        ground_truth: dict {query_col: [target_table, ...]}
        k_values: list of k values for evaluation

    Returns:
        dict: {k: {mean_f1, precision, recall, num_queries}}
    """
    metrics = {}

    for k in k_values:
        f1_scores = []
        precisions = []
        recalls = []

        for query_col in ground_truth:
            if query_col not in results:
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            predicted = [table for table, score in results[query_col][:k]]
            relevant = set(ground_truth[query_col])

            if len(relevant) == 0:
                continue

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
            'mean_f1': float(np.mean(f1_scores)),
            'precision': float(np.mean(precisions)),
            'recall': float(np.mean(recalls)),
            'num_queries': len(f1_scores)
        }

    return metrics


def print_metrics(metrics, k_values):
    """Print metrics in table format."""
    print()
    print("=" * 70)
    print("JOIN SEARCH EVALUATION METRICS (Table Level)")
    print("=" * 70)

    print(f"{'Metric':<15}", end="")
    for k in k_values:
        print(f"{'@' + str(k):>12}", end="")
    print()
    print("-" * (15 + 12 * len(k_values)))

    for metric_name in ['mean_f1', 'precision', 'recall']:
        print(f"{metric_name.replace('_', ' ').title():<15}", end="")
        for k in k_values:
            value = metrics[k][metric_name]
            print(f"{value:>12.4f}", end="")
        print()

    print()
    print(f"{'Queries':<15}", end="")
    for k in k_values:
        num = metrics[k]['num_queries']
        print(f"{num:>12,}", end="")
    print()
    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()

    # Parse k_values
    k_values = sorted(set(int(x) for x in args.k_values.split(',') if int(x) <= args.k))

    # Print header
    print("=" * 70)
    print("Join Search (TabSketchFM Methodology)")
    print("=" * 70)
    print(f"Query embeddings:    {args.query_emb}")
    print(f"Datalake embeddings: {args.datalake_emb}")
    print(f"Ground truth:        {args.ground_truth}")
    print(f"Min score:           {args.min_score}")
    print(f"K:                   {args.k}")
    print(f"K values:            {k_values}")
    print(f"Index type:          {args.index_type}")
    print(f"Use GPU:             {args.use_gpu} (available: {GPU_AVAILABLE})")
    print("=" * 70)

    # Step 1: Load embeddings
    print("\n[1/5] Loading query embeddings...")
    query_embeddings = load_embeddings(args.query_emb)

    print("\n[2/5] Loading datalake embeddings...")
    datalake_embeddings = load_embeddings(args.datalake_emb)

    # Merge all embeddings for index
    all_embeddings = {**datalake_embeddings}
    for k, v in query_embeddings.items():
        if k not in all_embeddings:
            all_embeddings[k] = v

    col_order = list(all_embeddings.keys())
    print(f"\n   Total unique columns in index: {len(col_order):,}")

    # Step 2: Load ground truth
    print("\n[3/5] Loading ground truth...")
    column_gt = load_ground_truth_csv(args.ground_truth, args.min_score)

    # Convert to table-level
    print("   Converting to table-level ground truth...")
    table_gt = convert_to_table_level_ground_truth(column_gt)

    num_queries = len(table_gt)
    total_tables = sum(len(tables) for tables in table_gt.values())
    print(f"   Target tables: {total_tables:,} (avg {total_tables/num_queries:.2f} per query)")

    # Check coverage
    queries_in_embeddings = sum(1 for q in table_gt if q in all_embeddings)
    print(f"   Queries with embeddings: {queries_in_embeddings:,}/{num_queries:,}")

    # Step 3: Build index
    print("\n[4/5] Building search index...")
    index, embeddings_matrix, index_type = build_index(
        all_embeddings, col_order, index_type=args.index_type, use_gpu=args.use_gpu
    )

    # Step 4: Search
    print(f"\n[5/5] Performing join search (k={args.k})...")
    results = {}

    missing_queries = 0
    for query_col in tqdm(table_gt.keys(), desc="Searching"):
        if query_col not in all_embeddings:
            missing_queries += 1
            continue
        ranked_tables = search_single_query(
            query_col, all_embeddings, col_order, index, embeddings_matrix,
            index_type=index_type, k=args.k
        )
        results[query_col] = ranked_tables

    print(f"\n   Queries processed: {len(results):,}")
    print(f"   Queries missing embeddings: {missing_queries:,}")

    # Step 5: Evaluate
    print("\nEvaluating results...")
    metrics = evaluate(results, table_gt, k_values)
    print_metrics(metrics, k_values)

    # Save results
    if args.output_results:
        os.makedirs(os.path.dirname(args.output_results) or ".", exist_ok=True)
        with open(args.output_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {args.output_results}")

    # Save metrics
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {args.output}")

    print("\nDone!")

    return metrics


if __name__ == '__main__':
    main()
