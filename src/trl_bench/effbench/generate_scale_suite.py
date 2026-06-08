"""Generate the Eff-Scale controlled scaling suite.

Creates semi-synthetic tables by varying one factor at a time from a baseline,
using real column distributions from anchor tables as seeds.

Row track sweeps: n_rows, n_features, cat_share, cat_cardinality, missingness.
Column track sweeps: n_columns, n_context_rows, avg_cell_tokens, type_mix.

Usage::

    python -m effbench.generate_scale_suite --output-dir effbench/scale_suite
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from effbench.spec import COL_BASELINE, COL_SWEEPS, ROW_BASELINE, ROW_SWEEPS


# ---------------------------------------------------------------------------
# Row table generation
# ---------------------------------------------------------------------------

def _generate_numeric_column(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a realistic numeric column (normal with some structure)."""
    mu = rng.uniform(-10, 100)
    sigma = rng.uniform(0.1, 50)
    return rng.normal(mu, sigma, size=n).astype(np.float32)


def _generate_categorical_column(
    n: int, cardinality: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a categorical column with realistic value names."""
    # Create category labels like "cat_0", "cat_1", etc.
    cats = [f"cat_{i}" for i in range(cardinality)]
    # Zipf-like distribution for realistic category frequencies
    probs = 1.0 / np.arange(1, cardinality + 1, dtype=float)
    probs /= probs.sum()
    return rng.choice(cats, size=n, p=probs)


def _inject_missingness(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Inject NaN values at the given rate."""
    if rate <= 0:
        return df
    mask = rng.random(df.shape) < rate
    df = df.copy()
    df[mask] = np.nan
    return df


def generate_row_table(
    n_rows: int = 10_000,
    n_features: int = 32,
    cat_share: float = 0.5,
    cat_cardinality: int = 32,
    missingness: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a semi-synthetic table for row-level efficiency testing.

    Returns:
        (dataframe, metadata_dict)
    """
    rng = np.random.default_rng(seed)

    n_cat = int(n_features * cat_share)
    n_num = n_features - n_cat

    columns = {}
    col_names = []

    # Numeric columns
    for i in range(n_num):
        name = f"num_{i}"
        columns[name] = _generate_numeric_column(n_rows, rng)
        col_names.append(name)

    # Categorical columns
    for i in range(n_cat):
        name = f"cat_{i}"
        columns[name] = _generate_categorical_column(n_rows, cat_cardinality, rng)
        col_names.append(name)

    # Add a dummy binary label column (needed by some models)
    columns["_label"] = rng.integers(0, 2, size=n_rows)

    df = pd.DataFrame(columns)

    # Inject missingness (not in label)
    if missingness > 0:
        feature_cols = [c for c in df.columns if c != "_label"]
        df_feat = df[feature_cols]
        df_feat = _inject_missingness(df_feat, missingness, rng)
        df[feature_cols] = df_feat

    metadata = {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_numeric": n_num,
        "n_categorical": n_cat,
        "cat_share": cat_share,
        "cat_cardinality": cat_cardinality,
        "missingness": missingness,
        "seed": seed,
        "label_column": "_label",
    }

    return df, metadata


# ---------------------------------------------------------------------------
# Column/table generation
# ---------------------------------------------------------------------------

def _generate_text_cell(avg_tokens: int, rng: np.random.Generator) -> str:
    """Generate a realistic text cell with approximately avg_tokens words."""
    # Sample actual token count from Poisson around the target
    n_tokens = max(1, rng.poisson(avg_tokens))
    # Generate words of varying length
    words = []
    for _ in range(n_tokens):
        word_len = rng.integers(2, 10)
        word = "".join(rng.choice(list(string.ascii_lowercase), size=word_len))
        words.append(word)
    return " ".join(words)


def _generate_header(avg_tokens: int, idx: int, rng: np.random.Generator) -> str:
    """Generate a realistic column header."""
    if avg_tokens <= 1:
        return f"col_{idx}"
    prefixes = ["total", "avg", "count", "max", "min", "sum", "pct", "rate", "num", "is"]
    suffixes = ["value", "amount", "score", "type", "status", "name", "date", "id", "code", "flag"]
    return f"{rng.choice(prefixes)}_{rng.choice(suffixes)}_{idx}"


def generate_column_table(
    n_columns: int = 16,
    n_context_rows: int = 16,
    avg_cell_tokens: int = 4,
    type_mix: str = "mixed",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a semi-synthetic table for column/table-level efficiency testing.

    Args:
        type_mix: "numeric" | "mixed" | "text"

    Returns:
        (dataframe, metadata_dict)
    """
    rng = np.random.default_rng(seed)

    columns = {}
    headers = []

    for i in range(n_columns):
        header = _generate_header(2, i, rng)
        headers.append(header)

        if type_mix == "numeric":
            columns[header] = _generate_numeric_column(n_context_rows, rng)
        elif type_mix == "text":
            columns[header] = [
                _generate_text_cell(avg_cell_tokens, rng) for _ in range(n_context_rows)
            ]
        else:  # mixed
            if rng.random() < 0.4:
                columns[header] = _generate_numeric_column(n_context_rows, rng)
            elif rng.random() < 0.5:
                card = rng.integers(3, 20)
                columns[header] = _generate_categorical_column(n_context_rows, card, rng)
            else:
                columns[header] = [
                    _generate_text_cell(avg_cell_tokens, rng) for _ in range(n_context_rows)
                ]

    df = pd.DataFrame(columns)

    metadata = {
        "n_columns": n_columns,
        "n_context_rows": n_context_rows,
        "avg_cell_tokens": avg_cell_tokens,
        "type_mix": type_mix,
        "seed": seed,
    }

    return df, metadata


# ---------------------------------------------------------------------------
# Suite generation
# ---------------------------------------------------------------------------

def generate_row_sweep_suite(output_dir: Path, seed: int = 42) -> List[Dict]:
    """Generate all one-factor row sweep tables."""
    manifest = []
    baseline = ROW_BASELINE.copy()

    for factor, levels in ROW_SWEEPS.items():
        for level in levels:
            params = baseline.copy()
            params[factor] = level
            sweep_id = f"row_sweep_{factor}_{level}"

            df, meta = generate_row_table(**params, seed=seed)
            meta["sweep_factor"] = factor
            meta["sweep_level"] = level
            meta["sweep_id"] = sweep_id

            table_dir = output_dir / "row" / sweep_id
            table_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(table_dir / "data.csv", index=False)
            with open(table_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            manifest.append(meta)
            print(f"  {sweep_id}: {params['n_rows']} rows x {params['n_features']} feats")

    return manifest


def generate_column_sweep_suite(output_dir: Path, seed: int = 42) -> List[Dict]:
    """Generate all one-factor column sweep tables."""
    manifest = []
    baseline = COL_BASELINE.copy()

    for factor, levels in COL_SWEEPS.items():
        for level in levels:
            params = baseline.copy()
            params[factor] = level
            sweep_id = f"col_sweep_{factor}_{level}"

            df, meta = generate_column_table(**params, seed=seed)
            meta["sweep_factor"] = factor
            meta["sweep_level"] = level
            meta["sweep_id"] = sweep_id

            table_dir = output_dir / "column" / sweep_id
            table_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(table_dir / "data.csv", index=False)
            with open(table_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            manifest.append(meta)
            print(f"  {sweep_id}: {meta['n_columns']} cols x {meta['n_context_rows']} rows")

    return manifest


def generate_bridge_tables(output_dir: Path, seed: int = 42) -> List[Dict]:
    """Generate 3 bridge tables valid for both row and column workloads."""
    manifest = []
    configs = [
        {"name": "bridge_small",  "n_rows": 2000, "n_cols": 16, "cat_share": 0.3},
        {"name": "bridge_medium", "n_rows": 10000, "n_cols": 32, "cat_share": 0.5},
        {"name": "bridge_wide",   "n_rows": 5000,  "n_cols": 64, "cat_share": 0.4},
    ]

    for cfg in configs:
        # Generate as row table (numeric + categorical, short text)
        df, meta = generate_row_table(
            n_rows=cfg["n_rows"],
            n_features=cfg["n_cols"],
            cat_share=cfg["cat_share"],
            cat_cardinality=16,
            missingness=0.05,
            seed=seed,
        )
        meta["bridge_id"] = cfg["name"]
        meta["is_bridge"] = True

        table_dir = output_dir / "bridge" / cfg["name"]
        table_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(table_dir / "data.csv", index=False)
        with open(table_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        manifest.append(meta)
        print(f"  {cfg['name']}: {cfg['n_rows']} rows x {cfg['n_cols']} cols")

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Eff-Scale suite")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "effbench" / "scale_suite"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Row Sweep Suite ===")
    row_manifest = generate_row_sweep_suite(output_dir, seed=args.seed)

    print("\n=== Generating Column Sweep Suite ===")
    col_manifest = generate_column_sweep_suite(output_dir, seed=args.seed)

    print("\n=== Generating Bridge Tables ===")
    bridge_manifest = generate_bridge_tables(output_dir, seed=args.seed)

    # Save full manifest
    manifest = {
        "row_sweeps": row_manifest,
        "column_sweeps": col_manifest,
        "bridge_tables": bridge_manifest,
        "row_baseline": ROW_BASELINE,
        "column_baseline": COL_BASELINE,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Row sweeps: {len(row_manifest)} tables")
    print(f"Column sweeps: {len(col_manifest)} tables")
    print(f"Bridge tables: {len(bridge_manifest)} tables")
    print(f"Total: {len(row_manifest) + len(col_manifest) + len(bridge_manifest)} tables")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
