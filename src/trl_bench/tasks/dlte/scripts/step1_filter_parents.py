"""
Step 1: Filter parent tables from TabFact + WTQ for DLTE benchmark.

Criteria:
  - 5 <= n_cols <= 20  (after dropping artifact columns)
  - 30 <= n_rows <= 200
  - At least one column with uniqueness >= 0.80 and non_null_ratio >= 0.90
  - Prefer text columns as entity key; fallback to any column

Outputs:
  - datasets/dlte_v1/manifests/parents_filtered.jsonl
  - datasets/dlte_v1/manifests/splits.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


GLOBAL_SEED = 42
MIN_ROWS = 30
MAX_ROWS = 200
MIN_COLS = 5
MAX_COLS = 20
KEY_UNIQUENESS_THRESHOLD = 0.80
KEY_NON_NULL_THRESHOLD = 0.90

# Columns to drop before analysis (artifacts from HTML-to-CSV conversion)
ARTIFACT_COL_PATTERN = re.compile(r"^(Unnamed:\s*\d+)$")
# Columns to skip as entity key candidates (sequential row numbers)
NULL_COL_PATTERN = re.compile(r"^null(_\d+)?$", re.IGNORECASE)


def load_table(csv_path: str) -> pd.DataFrame | None:
    """Load a CSV table with robust parsing."""
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        return df
    except Exception as e:
        print(f"  WARN: Failed to parse {csv_path}: {e}")
        return None


def drop_artifact_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop artifact columns like 'Unnamed: 0' (row index leakage)."""
    cols_to_drop = [c for c in df.columns if ARTIFACT_COL_PATTERN.match(str(c))]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def compute_column_stats(df: pd.DataFrame) -> list[dict]:
    """Compute uniqueness and non-null ratio for each column."""
    stats = []
    for idx, col in enumerate(df.columns):
        series = df[col]
        non_null = series.dropna()
        n_non_null = len(non_null)
        n_total = len(series)

        if n_non_null == 0:
            stats.append({
                "col_name": str(col),
                "col_idx": idx,
                "uniqueness": 0.0,
                "non_null_ratio": 0.0,
                "is_text": False,
                "is_null_col": bool(NULL_COL_PATTERN.match(str(col))),
            })
            continue

        # Normalize strings for uniqueness computation
        if series.dtype == object:
            normalized = non_null.astype(str).str.strip().str.lower()
            n_unique = normalized.nunique()
            is_text = True
        else:
            n_unique = non_null.nunique()
            is_text = False

        stats.append({
            "col_name": str(col),
            "col_idx": idx,
            "uniqueness": n_unique / n_non_null if n_non_null > 0 else 0.0,
            "non_null_ratio": n_non_null / n_total if n_total > 0 else 0.0,
            "is_text": is_text,
            "is_null_col": bool(NULL_COL_PATTERN.match(str(col))),
        })
    return stats


def select_key_column(col_stats: list[dict]) -> dict | None:
    """Select the best entity/key column.

    Priority:
    1. Text columns (non-null pattern) with uniqueness >= threshold
    2. Any column (non-null pattern) with uniqueness >= threshold
    3. None if nothing qualifies
    """
    candidates = [
        s for s in col_stats
        if s["uniqueness"] >= KEY_UNIQUENESS_THRESHOLD
        and s["non_null_ratio"] >= KEY_NON_NULL_THRESHOLD
        and not s["is_null_col"]
    ]

    if not candidates:
        # Try including null-pattern columns as last resort
        candidates = [
            s for s in col_stats
            if s["uniqueness"] >= KEY_UNIQUENESS_THRESHOLD
            and s["non_null_ratio"] >= KEY_NON_NULL_THRESHOLD
        ]

    if not candidates:
        return None

    # Prefer text columns, then sort by uniqueness descending
    text_candidates = [c for c in candidates if c["is_text"]]
    if text_candidates:
        return max(text_candidates, key=lambda c: c["uniqueness"])
    return max(candidates, key=lambda c: c["uniqueness"])


def process_dataset(tables_dir: Path, dataset_name: str,
                    project_root: Path = None) -> list[dict]:
    """Process all tables in a dataset directory."""
    tables_dir = Path(tables_dir)
    csv_files = sorted(tables_dir.glob("*.csv"))
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}: {len(csv_files)} CSV files")
    print(f"{'='*60}")

    results = []
    skipped = {"parse_error": 0, "too_few_rows": 0, "too_many_rows": 0,
               "too_few_cols": 0, "too_many_cols": 0, "no_key_col": 0}

    for i, csv_path in enumerate(csv_files):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(csv_files)}...")

        df = load_table(str(csv_path))
        if df is None:
            skipped["parse_error"] += 1
            continue

        # Drop artifact columns
        df = drop_artifact_columns(df)

        n_rows, n_cols = df.shape

        # Size filters
        if n_rows < MIN_ROWS:
            skipped["too_few_rows"] += 1
            continue
        if n_rows > MAX_ROWS:
            skipped["too_many_rows"] += 1
            continue
        if n_cols < MIN_COLS:
            skipped["too_few_cols"] += 1
            continue
        if n_cols > MAX_COLS:
            skipped["too_many_cols"] += 1
            continue

        # Compute column statistics and find key column
        col_stats = compute_column_stats(df)
        key_col = select_key_column(col_stats)

        if key_col is None:
            skipped["no_key_col"] += 1
            continue

        # Derive table_id to match embedding pickle files
        csv_stem = csv_path.stem  # e.g., "1-10006830-1.html" or "t_200_0"
        if csv_stem.endswith(".html"):
            # TabFact: table_id in embeddings is the stem (includes .html)
            table_id_in_embeddings = csv_stem
        else:
            table_id_in_embeddings = csv_stem

        parent_id = f"{dataset_name}__{csv_stem}"

        results.append({
            "parent_id": parent_id,
            "dataset": dataset_name,
            "csv_path": str(csv_path.relative_to(project_root)) if project_root else str(csv_path.resolve()),
            "csv_stem": csv_stem,
            "table_id_in_embeddings": table_id_in_embeddings,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "key_col": key_col["col_name"],
            "key_col_idx": key_col["col_idx"],
            "key_uniqueness": round(key_col["uniqueness"], 4),
            "key_non_null_ratio": round(key_col["non_null_ratio"], 4),
            "key_is_text": key_col["is_text"],
        })

    print(f"\n  Results for {dataset_name}:")
    print(f"    Total CSVs:    {len(csv_files)}")
    print(f"    Passed filter: {len(results)}")
    print(f"    Skipped:")
    for reason, count in skipped.items():
        if count > 0:
            print(f"      {reason}: {count}")

    return results


def create_splits(parent_ids: list[str], seed: int = GLOBAL_SEED) -> dict:
    """Split parent IDs into train/dev/test (60/15/25%)."""
    # First split: 75% train+dev, 25% test
    train_dev, test = train_test_split(
        parent_ids, test_size=0.25, random_state=seed
    )
    # Second split: from train+dev, 20% becomes dev (= 15% of total)
    train, dev = train_test_split(
        train_dev, test_size=0.20, random_state=seed
    )
    return {"train": sorted(train), "dev": sorted(dev), "test": sorted(test)}


def main():
    parser = argparse.ArgumentParser(description="Step 1: Filter parent tables for DLTE")
    parser.add_argument("--project-root", type=str,
                        default=None,
                        help="Project root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {project_root}/datasets/dlte_v1/manifests)")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "datasets" / "dlte_v1" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process both datasets
    all_parents = []

    tabfact_dir = project_root / "datasets" / "tabfact" / "tables"
    if tabfact_dir.exists():
        all_parents.extend(process_dataset(tabfact_dir, "tabfact", project_root))
    else:
        print(f"WARNING: TabFact directory not found: {tabfact_dir}")

    wtq_dir = project_root / "datasets" / "wtq" / "tables"
    if wtq_dir.exists():
        all_parents.extend(process_dataset(wtq_dir, "wtq", project_root))
    else:
        print(f"WARNING: WTQ directory not found: {wtq_dir}")

    # Sort by parent_id for reproducibility
    all_parents.sort(key=lambda x: x["parent_id"])

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total filtered parents: {len(all_parents)}")

    # Per-dataset counts
    for ds in ["tabfact", "wtq"]:
        count = sum(1 for p in all_parents if p["dataset"] == ds)
        print(f"  {ds}: {count}")

    # Stats
    rows = [p["n_rows"] for p in all_parents]
    cols = [p["n_cols"] for p in all_parents]
    print(f"\nRow distribution: min={min(rows)}, median={np.median(rows):.0f}, "
          f"max={max(rows)}, mean={np.mean(rows):.1f}")
    print(f"Col distribution: min={min(cols)}, median={np.median(cols):.0f}, "
          f"max={max(cols)}, mean={np.mean(cols):.1f}")

    # Key column analysis
    text_keys = sum(1 for p in all_parents if p["key_is_text"])
    print(f"\nKey column type: {text_keys} text ({text_keys/len(all_parents)*100:.1f}%), "
          f"{len(all_parents)-text_keys} numeric")

    # Top key column names
    from collections import Counter
    key_names = Counter(p["key_col"] for p in all_parents)
    print(f"\nTop 10 key column names:")
    for name, count in key_names.most_common(10):
        print(f"  {name}: {count}")

    # Write parents manifest
    parents_path = output_dir / "parents_filtered.jsonl"
    with open(parents_path, "w") as f:
        for parent in all_parents:
            f.write(json.dumps(parent) + "\n")
    print(f"\nWritten: {parents_path} ({len(all_parents)} entries)")

    # Create and write splits
    parent_ids = [p["parent_id"] for p in all_parents]
    splits = create_splits(parent_ids)
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Written: {splits_path}")
    print(f"  train: {len(splits['train'])}, dev: {len(splits['dev'])}, test: {len(splits['test'])}")

    # Spot-check: print 10 random samples
    rng = np.random.RandomState(GLOBAL_SEED)
    sample_indices = rng.choice(len(all_parents), min(10, len(all_parents)), replace=False)
    print(f"\n{'='*60}")
    print(f"SPOT CHECK: 10 random samples")
    print(f"{'='*60}")
    for idx in sample_indices:
        p = all_parents[idx]
        print(f"  {p['parent_id']}: {p['n_rows']}×{p['n_cols']}, "
              f"key='{p['key_col']}' (uniq={p['key_uniqueness']}, text={p['key_is_text']})")


if __name__ == "__main__":
    main()
