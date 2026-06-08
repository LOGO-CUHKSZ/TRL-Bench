#!/usr/bin/env python3
"""
Run join search using HNSW index.
Reuses the search logic from hnsw_search.py

Supports both legacy tuple format and unified dict format for embeddings:
- Legacy tuple format: [(table_name, col_idx, embedding), ...]
- Unified dict format: [{'table': ..., 'column_embeddings': {...}}, ...]
"""
import os
import pickle
import warnings

warnings.warn(
    "run_search.py is deprecated and has known bugs "
    "(module-level execution, missing str coercion, no duplicate column handling, "
    "no newline stripping, silent query dropping). "
    "For exact cosine search, use run_search_and_evaluate.py instead.",
    DeprecationWarning,
    stacklevel=1
)

# Resolve project root (two levels up from this script)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import hnswlib
from tqdm import tqdm
import sys
import argparse
from collections import defaultdict

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import unified format adapter (optional, with fallback)
try:
    from trl_bench.utils.unified_embedding_format import to_join_search_format
    HAS_UNIFIED_FORMAT = True
except ImportError:
    HAS_UNIFIED_FORMAT = False


def convert_to_tuples(embeddings):
    """
    Convert embeddings to tuple format if needed.

    Handles:
    - Legacy tuple format: [(table, col, emb), ...] - passed through
    - Unified dict format: [{'table': ..., 'column_embeddings': {...}}, ...]

    Returns:
        List of tuples: [(table_name, col_idx, embedding), ...]
    """
    if not embeddings:
        return []

    first = embeddings[0]

    # Already in tuple format — normalize table names to basenames
    # (legacy tuples may contain paths; basenames are needed for self-match filtering)
    if isinstance(first, tuple) and len(first) == 3:
        return [(os.path.basename(table), col, emb) for table, col, emb in embeddings]

    # Dict format - convert to tuples
    if isinstance(first, dict):
        result = []
        column_name_cache = {}
        for item in embeddings:
            # Prefer 'table' (path → basename with extension) over 'table_id' (may lack .csv)
            table_raw = item.get('table') or item.get('table_id') or item.get('table_name', '')

            # Normalize to basename (query CSVs use filenames, not paths)
            table = os.path.basename(table_raw)

            # Handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
            col_emb = item.get('column_embeddings') or item.get('column_embedding', {})

            if not col_emb:
                continue

            # Always load column_names metadata when available (unified format).
            col_names = item.get('column_names')

            # Fall back to CSV header for integer-keyed items without column_names.
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

            for col_idx, emb in col_emb.items():
                # Determine the column key
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

                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                elif isinstance(emb, np.ndarray):
                    emb = emb.astype(np.float32)

                result.append((table, col_key, emb))

        return result

    raise ValueError(f"Unknown embedding format: {type(first)}")


def aggregate_to_table_level(results_df, aggregation_method, k):
    """
    Aggregate column-level results to table level.

    Aggregates across candidate_column only, preserving query_column.
    For each (query_table, query_column), returns top-k candidate tables.

    Args:
        results_df: DataFrame with columns [query_table, query_column, candidate_table, candidate_column, similarity]
        aggregation_method: One of 'tabsketchfm', 'max', 'mean', 'sum'
        k: Number of top tables to return per (query_table, query_column)

    Returns:
        DataFrame with columns [query_table, query_column, candidate_table, similarity]
    """
    # Group by (query_table, query_column, candidate_table) - aggregate across candidate_column only
    grouped = results_df.groupby(['query_table', 'query_column', 'candidate_table'])

    table_results = []
    for (query_table, query_column, cand_table), group in grouped:
        num_matches = len(group)
        similarities = group['similarity'].values

        if aggregation_method == 'tabsketchfm':
            # TabSketchFM: primary sort by num_matches DESC, secondary by sum_similarity DESC
            score = sum(similarities)
            table_results.append({
                'query_table': query_table,
                'query_column': query_column,
                'candidate_table': cand_table,
                'similarity': score,  # Use 'similarity' for compatibility with evaluation
                'num_matches': num_matches
            })
        elif aggregation_method == 'max':
            score = max(similarities)
            table_results.append({
                'query_table': query_table,
                'query_column': query_column,
                'candidate_table': cand_table,
                'similarity': score,
                'num_matches': num_matches
            })
        elif aggregation_method == 'mean':
            score = np.mean(similarities)
            table_results.append({
                'query_table': query_table,
                'query_column': query_column,
                'candidate_table': cand_table,
                'similarity': score,
                'num_matches': num_matches
            })
        elif aggregation_method == 'sum':
            score = sum(similarities)
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

    # Sort and select top-k per (query_table, query_column)
    if aggregation_method == 'tabsketchfm':
        # TabSketchFM: sort by (num_matches DESC, similarity DESC)
        table_df = table_df.sort_values(
            ['query_table', 'query_column', 'num_matches', 'similarity'],
            ascending=[True, True, False, False]
        )
    else:
        # Other methods: sort by similarity DESC
        table_df = table_df.sort_values(
            ['query_table', 'query_column', 'similarity'],
            ascending=[True, True, False]
        )

    # Keep top-k per (query_table, query_column)
    top_k_results = table_df.groupby(['query_table', 'query_column']).head(k).reset_index(drop=True)

    # Drop num_matches column for cleaner output (optional, keep similarity)
    if 'num_matches' in top_k_results.columns:
        top_k_results = top_k_results.drop(columns=['num_matches'])

    return top_k_results

def parse_args():
    parser = argparse.ArgumentParser(description="Run join search using HNSW index")
    parser.add_argument("--query_list", type=str,
                        default=os.path.join(_PROJECT_ROOT, "datasets/opendata/queries/opendata_join/opendata_join_query.csv"),
                        help="Path to query list CSV file")
    parser.add_argument("--query_emb", type=str,
                        default=os.path.join(_PROJECT_ROOT, "embeddings/join_search/query_embeddings_hf.pkl"),
                        help="Path to query embeddings pickle file")
    parser.add_argument("--datalake_emb", type=str,
                        default=os.path.join(_PROJECT_ROOT, "embeddings/join_search/datalake_embeddings_hf.pkl"),
                        help="Path to data lake embeddings pickle file")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_PROJECT_ROOT, "results/evaluation/join_search/results.csv"),
                        help="Path to output results CSV file")
    parser.add_argument("--k", type=int, default=50,
                        help="Return top-K results per query (default: 50)")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Minimum similarity threshold (default: 0)")
    parser.add_argument("--aggregate_to_table", action="store_true",
                        help="Aggregate results to table level (instead of column level)")
    parser.add_argument("--aggregation", type=str, default="tabsketchfm",
                        choices=["tabsketchfm", "max", "mean", "sum"],
                        help="Aggregation method for table-level results (default: tabsketchfm)")
    return parser.parse_args()

args = parse_args()

print("="*60)
print("DeepJoin Join Search")
print("="*60)

# Load embeddings
print(f"\n[1/5] Loading query embeddings...")
with open(args.query_emb, "rb") as f:
    query_embeddings_raw = pickle.load(f)

# Convert to tuple format if needed (supports both unified dict and legacy tuple formats)
query_embeddings = convert_to_tuples(query_embeddings_raw)
print(f"✓ Loaded {len(query_embeddings)} query column embeddings")

print(f"\n[2/5] Loading data lake embeddings...")
with open(args.datalake_emb, "rb") as f:
    datalake_embeddings_raw = pickle.load(f)

# Convert to tuple format if needed
datalake_embeddings = convert_to_tuples(datalake_embeddings_raw)
print(f"✓ Loaded {len(datalake_embeddings)} data lake column embeddings")

# Create lookups
print(f"\n[3/5] Preparing data structures...")
query_lookup = {}
for table, col, emb in query_embeddings:
    query_lookup[(table, col)] = emb

datalake_list = []
for table, col, emb in datalake_embeddings:
    datalake_list.append((table, col, emb))

# Precompute column counts per table for adaptive self-match over-fetch
from collections import Counter
datalake_table_col_counts = Counter(table for table, _, _ in datalake_list)
max_same_table_cols = max(datalake_table_col_counts.values()) if datalake_table_col_counts else 0

print(f"  Query lookup: {len(query_lookup)} entries")
print(f"  Datalake tables: {len(datalake_table_col_counts)} (max {max_same_table_cols} cols/table)")

# Build HNSW index
print(f"\n[4/5] Building HNSW index...")
dim = datalake_list[0][2].shape[0]
print(f"  Dimension: {dim}")
print(f"  Number of columns: {len(datalake_list)}")

index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=len(datalake_list), ef_construction=200, M=32)
# ef must be >= k_search for any query
index.set_ef(max(args.k * 10, args.k + max_same_table_cols + 64))

# Add items to index
embeddings_array = np.array([emb for _, _, emb in datalake_list])
index.add_items(embeddings_array, np.arange(len(datalake_list)))
print(f"  HNSW index built successfully")

# Load query list and search
print(f"\n[5/5] Running join search...")
query_df = pd.read_csv(args.query_list, dtype={'query_table': str, 'query_column': str},
                       keep_default_na=False)
# Normalize query table names to basenames
query_df['query_table'] = query_df['query_table'].apply(os.path.basename)
print(f"  Total queries: {len(query_df)}")

results = []

for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Searching", ncols=80):
    query_table = row['query_table']
    query_column = row['query_column']

    key = (query_table, query_column)
    if key not in query_lookup:
        continue

    query_emb = query_lookup[key]

    # Adaptive over-fetch: account for same-table columns that will be filtered
    same_table_cols = datalake_table_col_counts.get(query_table, 0)
    k_search = min(args.k + same_table_cols + 32, len(datalake_list))
    labels, distances = index.knn_query(query_emb.reshape(1, -1), k=k_search)

    query_results = []
    for idx, dist in zip(labels[0], distances[0]):
        similarity = 1 - dist  # Convert cosine distance to similarity

        if similarity < args.threshold:
            continue

        cand_table, cand_col, _ = datalake_list[idx]

        # Filter self-table matches (entire query table excluded)
        if cand_table == query_table:
            continue

        query_results.append({
            'query_table': query_table,
            'query_column': query_column,
            'candidate_table': cand_table,
            'candidate_column': cand_col,
            'similarity': similarity
        })

        if len(query_results) >= args.k:
            break

    results.extend(query_results)

# Save results
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
_RESULT_COLUMNS = ['query_table', 'query_column', 'candidate_table', 'candidate_column', 'similarity']
results_df = pd.DataFrame(results, columns=_RESULT_COLUMNS) if results else pd.DataFrame(columns=_RESULT_COLUMNS)

# Apply table-level aggregation if requested
if args.aggregate_to_table:
    print(f"\n[6/6] Aggregating to table level (method: {args.aggregation})...")
    results_df = aggregate_to_table_level(results_df, args.aggregation, args.k)
    print(f"✓ Aggregated to {len(results_df)} table-level results")

results_df.to_csv(args.output, index=False)

print("="*60)
print("✅ COMPLETE!")
print(f"   Total results: {len(results_df)}")
print(f"   Output file: {args.output}")
if args.aggregate_to_table:
    print(f"   Aggregation: {args.aggregation}")
else:
    print(f"   Average results per query: {len(results_df) / len(query_df):.1f}")
print("="*60)
print("\nSample results:")
print(results_df.head(10).to_string(index=False))
