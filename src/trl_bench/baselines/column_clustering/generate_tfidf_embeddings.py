#!/usr/bin/env python3
"""
Generate TF-IDF column embeddings as a non-neural baseline for column clustering.

For each table in a dataset, builds a text representation of every column from
its header (if meaningful) and sampled cell values, then vectorizes with
character n-gram TF-IDF. The resulting vectors are saved in the unified v2.0
pkl format so the SLURM pipeline auto-discovers them as model="tfidf".

This baseline answers: "do learned embeddings beat a simple lexical similarity
heuristic?" — particularly important for column clustering where there is no
learned probe to provide an intermediate reference point.

Usage:
    # Generate for both clustering datasets
    python utils/baselines/column_clustering/generate_tfidf_embeddings.py \\
        --datasets sato sotab

    # Custom parameters
    python utils/baselines/column_clustering/generate_tfidf_embeddings.py \\
        --datasets sato --max_features 512 --ngram_range 3 5 --max_values 50

    # Dry run
    python utils/baselines/column_clustering/generate_tfidf_embeddings.py \\
        --datasets sato sotab --dry-run
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Generic header patterns (skip these — no semantic content)
_GENERIC_HEADER_RE = re.compile(
    r'^(col\d+|column\d+|unnamed[:\s].*|)$', re.IGNORECASE
)

# Null-like values to skip
_NULL_VALUES = frozenset({'', 'nan', 'none', 'null', 'n/a', 'na', '-', '--'})


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def is_generic_header(header: str) -> bool:
    """Check whether a column header is generic/meaningless."""
    return bool(_GENERIC_HEADER_RE.match(header.strip()))


def serialize_column(header: str, cell_values: list, max_values: int = 50) -> str:
    """Build a text representation of a single column.

    Concatenates header (if non-generic) + up to max_values non-null cell
    values, separated by spaces.
    """
    parts = []

    if header and not is_generic_header(header):
        parts.append(header.strip())

    count = 0
    for val in cell_values:
        if count >= max_values:
            break
        val_str = str(val).strip()
        if val_str.lower() not in _NULL_VALUES:
            parts.append(val_str)
            count += 1

    return " ".join(parts)


def read_table_columns(table_path: Path) -> tuple:
    """Read a CSV table and return (headers, columns_data, status).

    Returns:
        headers: list of header strings (one per column)
        columns_data: list of lists (one list of cell values per column)
        status: 'ok', 'missing', 'unreadable', or 'no_columns'
        Returns ([], [], status) on failure.
    """
    if not table_path.exists():
        return [], [], 'missing'

    try:
        df = pd.read_csv(
            table_path,
            dtype=str,
            on_bad_lines='skip',
            keep_default_na=False,
        )
    except Exception:
        return [], [], 'unreadable'

    if len(df.columns) == 0:
        return [], [], 'no_columns'

    headers = [str(c) for c in df.columns]
    # Header-only tables still produce valid (empty) column lists
    columns_data = [df.iloc[:, i].tolist() for i in range(len(headers))]
    return headers, columns_data, 'ok'


def resolve_table_path(table_id, tables_dir: Path) -> Path:
    """Resolve a table_id to a file path in the tables directory."""
    # Integer table_id (sato): table_0.csv, table_1.csv, ...
    if isinstance(table_id, (int, float, np.integer)):
        return tables_dir / f"table_{int(table_id)}.csv"

    table_id_str = str(table_id)

    # Already has .csv extension
    if table_id_str.endswith('.csv'):
        return tables_dir / table_id_str

    # Try with .csv appended
    with_csv = tables_dir / f"{table_id_str}.csv"
    if with_csv.exists():
        return with_csv

    # Try without extension (rare)
    return tables_dir / table_id_str


def generate_tfidf_for_dataset(
    dataset_name: str,
    project_root: Path,
    model_name: str = "tfidf",
    max_features: int = 256,
    ngram_range: tuple = (3, 5),
    max_values: int = 50,
    analyzer: str = "char_wb",
    dry_run: bool = False,
):
    """Generate TF-IDF column embeddings for one dataset."""
    dataset_dir = project_root / 'datasets' / dataset_name
    tables_dir = dataset_dir / 'tables'
    labels_path = dataset_dir / 'all.csv'

    if not labels_path.exists():
        print(f"  SKIP {dataset_name}: labels file not found at {labels_path}")
        return

    if not tables_dir.exists():
        print(f"  SKIP {dataset_name}: tables directory not found at {tables_dir}")
        return

    labels_df = pd.read_csv(labels_path)
    table_ids = labels_df['table_id'].unique()
    n_tables = len(table_ids)
    n_labeled_cols = len(labels_df)

    print(f"  Labels: {n_labeled_cols} columns across {n_tables} tables, "
          f"{labels_df['class'].nunique()} types")

    out_path = project_root / 'assets' / 'embeddings' / 'column' / model_name / f'{dataset_name}.pkl'

    if dry_run:
        print(f"  [DRY-RUN] Would generate: {out_path}")
        print(f"  Tables: {n_tables}, Features: {max_features}")
        return

    # Phase 1: Read all tables and serialize every column
    print(f"  Phase 1: Reading {n_tables} tables and serializing columns...")

    # Track: (table_id, col_idx) -> index in flat list
    flat_texts = []       # flat list of all column texts across all tables
    table_data = {}       # table_id -> {headers, n_cols, col_start_idx, table_path}
    missing_tables = []
    unreadable_tables = []

    for table_id in table_ids:
        table_path = resolve_table_path(table_id, tables_dir)
        headers, columns_data, status = read_table_columns(table_path)

        if status != 'ok':
            if status == 'missing':
                missing_tables.append(table_id)
            else:
                unreadable_tables.append((table_id, str(table_path), status))
            continue

        n_cols = len(headers)
        col_start_idx = len(flat_texts)

        for col_idx in range(n_cols):
            text = serialize_column(headers[col_idx], columns_data[col_idx], max_values)
            flat_texts.append(text)

        table_data[table_id] = {
            'headers': headers,
            'n_cols': n_cols,
            'col_start_idx': col_start_idx,
            'table_path': str(table_path),
        }

    n_missing = len(missing_tables)
    n_unreadable = len(unreadable_tables)
    n_skipped = n_missing + n_unreadable
    n_total_cols = len(flat_texts)
    n_empty = sum(1 for t in flat_texts if not t.strip())

    print(f"  Tables read: {len(table_data)}/{n_tables} "
          f"({n_missing} missing, {n_unreadable} unreadable)")
    if missing_tables:
        sample = missing_tables[:5]
        print(f"  Missing tables (first 5): {sample}")
    if unreadable_tables:
        for tid, path, reason in unreadable_tables[:5]:
            print(f"  Unreadable: {tid} ({path}) — {reason}")
    print(f"  Total columns: {n_total_cols} ({n_empty} empty text)")

    # Phase 2: Fit TF-IDF on all column texts
    if not flat_texts:
        print(f"  SKIP {dataset_name}: no readable tables/columns found")
        return

    print(f"  Phase 2: Fitting TF-IDF (analyzer={analyzer}, "
          f"ngram_range={ngram_range}, max_features={max_features})...")

    # Guard: if all texts are empty (e.g., tables with only null values and
    # generic headers), TfidfVectorizer raises ValueError("empty vocabulary").
    # Fall back to zero vectors with the requested dimensionality.
    non_empty_count = sum(1 for t in flat_texts if t.strip())
    if non_empty_count == 0:
        print(f"  WARNING: all {n_total_cols} column texts are empty — "
              f"producing zero vectors with dim={max_features}")
        dense_matrix = np.zeros((n_total_cols, max_features), dtype=np.float32)
        actual_dim = max_features
    else:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(flat_texts)
        dense_matrix = tfidf_matrix.toarray().astype(np.float32)
        actual_dim = dense_matrix.shape[1]

    print(f"  TF-IDF matrix: {dense_matrix.shape}")

    # Count zero vectors
    norms = np.linalg.norm(dense_matrix, axis=1)
    n_zero = int(np.sum(norms == 0))
    if n_zero > 0:
        print(f"  Zero-norm vectors: {n_zero}/{n_total_cols}")

    # Phase 3: Package as unified v2.0 pkl
    print(f"  Phase 3: Packaging as list-of-dicts pkl...")

    output_data = []
    for table_id, info in table_data.items():
        n_cols = info['n_cols']
        start = info['col_start_idx']
        headers = info['headers']
        table_path = info['table_path']

        # Build column_embeddings with int keys 0..n-1
        col_embeddings = {}
        for col_idx in range(n_cols):
            col_embeddings[col_idx] = dense_matrix[start + col_idx]

        # Table-level aggregation
        col_vecs = dense_matrix[start:start + n_cols]
        column_mean = col_vecs.mean(axis=0)

        # Derive table_name and canonical table_id from file stem.
        # Existing column embeddings (bert, tabsketchfm, etc.) use the CSV
        # file stem as table_id: "table_0" for sato, string IDs for SOTAB.
        # Downstream consumers (train_ct_mode4.py, csv_relation_pipeline.py)
        # look up embeddings by this canonical form.
        table_basename = os.path.basename(table_path)
        table_stem = table_basename[:-4] if table_basename.endswith('.csv') else table_basename

        output_data.append({
            'table_id': table_stem,
            'table': table_path,
            'table_name': table_stem,
            'model_name': model_name,
            'embedding_dim': actual_dim,
            'column_embeddings': col_embeddings,
            'column_names': headers,
            'table_embedding': {
                'column_mean': column_mean,
                'cls_embedding': None,
                'table_embedding': None,
                'token_mean': None,
            },
            'version': '2.0',
            'format': 'unified_table_embedding',
        })

    # Save
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"  Saved {len(output_data)} tables to {out_path}")
    print(f"  Embedding dim: {actual_dim}"
          + (f" (< max_features={max_features})" if actual_dim < max_features else ""))

    # Summary
    print(f"\n  Summary for {dataset_name}:")
    print(f"    Tables processed: {len(output_data)}")
    print(f"    Tables missing:   {n_missing}")
    print(f"    Tables unreadable:{n_unreadable}")
    print(f"    Total columns:    {n_total_cols}")
    print(f"    Empty texts:      {n_empty}")
    print(f"    Zero vectors:     {n_zero}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TF-IDF column embeddings as a non-neural baseline"
    )
    parser.add_argument("--datasets", nargs='+', default=['sato', 'sotab'],
                        help="Dataset names to process (default: sato sotab)")
    parser.add_argument("--model_name", default="tfidf",
                        help="Output model name (default: tfidf)")
    parser.add_argument("--max_features", type=int, default=256,
                        help="TF-IDF vocabulary size = embedding dimension (default: 256)")
    parser.add_argument("--ngram_range", nargs=2, type=int, default=[3, 5],
                        help="Character n-gram range (default: 3 5)")
    parser.add_argument("--max_values", type=int, default=50,
                        help="Max cell values per column (default: 50)")
    parser.add_argument("--analyzer", default="char_wb",
                        choices=["char_wb", "char", "word"],
                        help="TF-IDF analyzer type (default: char_wb)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would be generated")

    args = parser.parse_args()
    project_root = get_project_root()

    print(f"Project root: {project_root}")
    print(f"Model name: {args.model_name}")
    print(f"TF-IDF: analyzer={args.analyzer}, ngram_range={tuple(args.ngram_range)}, "
          f"max_features={args.max_features}, max_values={args.max_values}")
    print()

    for dataset in args.datasets:
        print(f"Processing {dataset}...")
        generate_tfidf_for_dataset(
            dataset_name=dataset,
            project_root=project_root,
            model_name=args.model_name,
            max_features=args.max_features,
            ngram_range=tuple(args.ngram_range),
            max_values=args.max_values,
            analyzer=args.analyzer,
            dry_run=args.dry_run,
        )
        print()


if __name__ == "__main__":
    main()
