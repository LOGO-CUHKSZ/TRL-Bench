"""Anchor dataset selection for TRL-EffBench Eff-Real.

Computes metadata for all candidate tables, clusters in metadata space,
and selects medoids + extreme cases as anchor datasets.

Usage::

    python -m effbench.select_anchors --mode row --output effbench/anchors_row.json
    python -m effbench.select_anchors --mode column --output effbench/anchors_column.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from effbench.spec import TableMetadata


# ---------------------------------------------------------------------------
# Metadata computation
# ---------------------------------------------------------------------------

def compute_row_table_metadata(table_dir: Path) -> TableMetadata:
    """Compute metadata for one row_data table (OpenML format)."""
    dataset_json = table_dir / "dataset.json"
    data_csv = table_dir / "data.csv"

    if not dataset_json.exists() or not data_csv.exists():
        raise FileNotFoundError(f"Missing files in {table_dir}")

    with open(dataset_json) as f:
        meta = json.load(f)

    table_id = table_dir.name
    data_meta = meta.get("data", {})
    n_rows = data_meta.get("n_rows", meta.get("n_rows", 0))
    n_columns = data_meta.get("n_columns", meta.get("n_columns", 0))

    # Read the data to compute detailed stats
    read_rows = min(n_rows, 10000) if n_rows > 0 else 10000
    try:
        df = pd.read_csv(data_csv, nrows=read_rows)
    except Exception:
        df = pd.read_csv(data_csv, nrows=read_rows, on_bad_lines="skip")

    # Identify label columns to exclude from feature metadata
    label_cols = set()
    for lbl in meta.get("labels", []):
        label_cols.add(lbl.get("column", ""))
    # Also check label_columns (schema v1.0)
    for col_name in meta.get("label_columns", []):
        label_cols.add(col_name)

    feature_cols = [c for c in df.columns if c not in label_cols]
    if not feature_cols:
        feature_cols = list(df.columns)

    df_feat = df[feature_cols]

    # Count numeric vs categorical
    n_numeric = 0
    n_categorical = 0
    cardinalities = []
    for col in df_feat.columns:
        try:
            pd.to_numeric(df_feat[col], errors="raise")
            n_numeric += 1
        except (ValueError, TypeError):
            n_categorical += 1
            nunique = df_feat[col].nunique()
            cardinalities.append(nunique)

    # Missingness
    total_cells = df_feat.shape[0] * df_feat.shape[1]
    missing_cells = df_feat.isna().sum().sum()
    missingness = missing_cells / max(total_cells, 1)

    # Cardinality stats
    avg_cardinality = float(np.mean(cardinalities)) if cardinalities else 0.0
    max_cardinality = max(cardinalities) if cardinalities else 0

    # Token-level estimates (whitespace-based approximation)
    cell_token_counts = []
    for col in df_feat.columns:
        for val in df_feat[col].dropna().head(200).astype(str):
            cell_token_counts.append(len(val.split()))
    avg_cell_tokens = float(np.mean(cell_token_counts)) if cell_token_counts else 1.0

    header_token_counts = [len(c.split("_")) for c in df_feat.columns]
    avg_header_tokens = float(np.mean(header_token_counts))

    # Serialized row length estimate: "col1: val1 | col2: val2 | ..."
    sample_rows = df_feat.head(100)
    row_lens = []
    for _, row in sample_rows.iterrows():
        parts = [f"{c}: {v}" for c, v in zip(df_feat.columns, row.values)]
        row_lens.append(len(" | ".join(parts).split()))
    avg_row_tokens = float(np.mean(row_lens)) if row_lens else 0.0

    # Serialized column length estimate: "header: val1, val2, ..."
    col_lens = []
    for col in df_feat.columns:
        vals = df_feat[col].dropna().head(100).astype(str).tolist()
        text = f"{col}: {', '.join(vals)}"
        col_lens.append(len(text.split()))
    avg_col_tokens = float(np.mean(col_lens)) if col_lens else 0.0

    file_size = data_csv.stat().st_size

    return TableMetadata(
        table_id=table_id,
        source="row_data",
        file_path=str(data_csv),
        n_rows=n_rows,
        n_columns=len(feature_cols),
        n_numeric=n_numeric,
        n_categorical=n_categorical,
        n_text=0,
        missingness=missingness,
        avg_cardinality=avg_cardinality,
        max_cardinality=max_cardinality,
        avg_cell_tokens=avg_cell_tokens,
        avg_header_tokens=avg_header_tokens,
        avg_row_tokens=avg_row_tokens,
        avg_col_tokens=avg_col_tokens,
        file_size_bytes=file_size,
    )


def compute_column_table_metadata(csv_path: Path, source: str = "unknown") -> TableMetadata:
    """Compute metadata for one column/table-level CSV."""
    try:
        df = pd.read_csv(csv_path, nrows=500, dtype=str)
    except Exception:
        df = pd.read_csv(csv_path, nrows=500, dtype=str, on_bad_lines="skip")

    n_rows = len(df)
    n_columns = len(df.columns)

    # All treated as text for column-level datasets
    n_numeric = 0
    n_categorical = 0
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors="raise")
            n_numeric += 1
        except (ValueError, TypeError):
            n_categorical += 1

    missingness = df.isna().sum().sum() / max(n_rows * n_columns, 1)

    cell_token_counts = []
    for col in df.columns:
        for val in df[col].dropna().head(100).astype(str):
            cell_token_counts.append(len(val.split()))
    avg_cell_tokens = float(np.mean(cell_token_counts)) if cell_token_counts else 1.0

    header_token_counts = [len(str(c).split()) for c in df.columns]
    avg_header_tokens = float(np.mean(header_token_counts))

    file_size = csv_path.stat().st_size

    return TableMetadata(
        table_id=csv_path.stem,
        source=source,
        file_path=str(csv_path),
        n_rows=n_rows,
        n_columns=n_columns,
        n_numeric=n_numeric,
        n_categorical=n_categorical,
        missingness=missingness,
        avg_cell_tokens=avg_cell_tokens,
        avg_header_tokens=avg_header_tokens,
        file_size_bytes=file_size,
    )


# ---------------------------------------------------------------------------
# Anchor selection via clustering
# ---------------------------------------------------------------------------

def select_anchors(
    metadata_list: List[TableMetadata],
    n_anchors: int = 8,
    feature_keys: List[str] | None = None,
) -> Tuple[List[int], np.ndarray]:
    """Select anchor tables via k-medoids in metadata space.

    Returns:
        (selected_indices, feature_matrix)
    """
    if feature_keys is None:
        feature_keys = [
            "n_rows", "n_columns", "n_numeric", "n_categorical",
            "missingness", "avg_cardinality", "avg_cell_tokens",
        ]

    # Build feature matrix
    raw = []
    for m in metadata_list:
        row = []
        for k in feature_keys:
            v = getattr(m, k, 0)
            # Log-transform counts
            if k in ("n_rows", "n_columns", "avg_cardinality", "max_cardinality"):
                v = math.log1p(v)
            row.append(float(v))
        raw.append(row)

    X = np.array(raw)

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    # Simple k-medoids (PAM-style greedy)
    n = len(X_norm)
    if n <= n_anchors:
        return list(range(n)), X

    # Initialize: pick the point closest to the global centroid
    centroid = X_norm.mean(axis=0)
    dists_to_centroid = np.linalg.norm(X_norm - centroid, axis=1)
    medoids = [int(np.argmin(dists_to_centroid))]

    # Greedy: add the point that is furthest from all current medoids
    for _ in range(n_anchors - 1):
        min_dists = np.full(n, np.inf)
        for m_idx in medoids:
            d = np.linalg.norm(X_norm - X_norm[m_idx], axis=1)
            min_dists = np.minimum(min_dists, d)
        # Mask already-selected
        for m_idx in medoids:
            min_dists[m_idx] = -1
        medoids.append(int(np.argmax(min_dists)))

    return medoids, X


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def scan_row_data(data_root: Path) -> List[TableMetadata]:
    """Scan all row_data tables and compute metadata."""
    row_data_dir = data_root / "datasets" / "row_data"
    tables = sorted(row_data_dir.glob("openml_*/dataset.json"))
    metadata = []
    for dj in tables:
        table_dir = dj.parent
        try:
            m = compute_row_table_metadata(table_dir)
            metadata.append(m)
        except Exception as e:
            print(f"  SKIP {table_dir.name}: {e}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Select EffBench anchor datasets")
    parser.add_argument("--mode", choices=["row", "column"], default="row")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--n-anchors", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.mode == "row":
        print("Scanning row_data tables...")
        metadata = scan_row_data(data_root)
        feature_keys = [
            "n_rows", "n_columns", "n_numeric", "n_categorical",
            "missingness", "avg_cardinality", "avg_cell_tokens",
        ]
    else:
        print("Column-level anchor selection not yet implemented.")
        print("Use --mode row for now.")
        return

    print(f"  Found {len(metadata)} tables")

    # Select anchors
    indices, X = select_anchors(metadata, n_anchors=args.n_anchors, feature_keys=feature_keys)

    print(f"\nSelected {len(indices)} anchor tables:")
    print(f"{'Table ID':<30} {'Rows':>8} {'Cols':>6} {'Num':>5} {'Cat':>5} {'Miss%':>6} {'AvgCard':>8}")
    print("-" * 75)
    for idx in indices:
        m = metadata[idx]
        print(f"{m.table_id:<30} {m.n_rows:>8} {m.n_columns:>6} "
              f"{m.n_numeric:>5} {m.n_categorical:>5} "
              f"{m.missingness:>5.1%} {m.avg_cardinality:>8.1f}")

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = str(PROJECT_ROOT / "effbench" / f"anchors_{args.mode}.json")

    result = {
        "mode": args.mode,
        "n_anchors": len(indices),
        "anchors": [metadata[i].to_dict() for i in indices],
        "all_metadata": [m.to_dict() for m in metadata],
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
