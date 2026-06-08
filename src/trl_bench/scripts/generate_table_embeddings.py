#!/usr/bin/env python3
"""Generate table-level embeddings from per-column embedding pickles.

This is the Stage-2 aggregator in the release pipeline:

    Stage 1  models/<m>/generate_column_embeddings.py
                -> embeddings/column/<m>/<dataset>.pkl
    Stage 2  THIS SCRIPT
                -> embeddings/table/<m>/<dataset>.pkl
    Stage 3  utils/downstream/run_task.py reads the Stage-2 pickle.

For each table in the Stage-1 pickle, four table-level embedding variants
are produced inside ``table_embedding`` (a dict):

    - cls_embedding    : CLS embedding (if the model produced one)
    - table_embedding  : native table-level embedding (if any)
    - column_mean      : mean-pooled column embeddings (recomputed if absent)
    - token_mean       : whole-table token mean (cannot be recomputed)

The aggregator does no I/O beyond reading the column pickle and writing the
table pickle; it is pure NumPy, deterministic, and free of randomness.

USAGE
-----
Programmatic (from inside the test suite, or another script)::

    from trl_bench.scripts.generate_table_embeddings import (
        extract_table_embeddings, process_model_dataset,
    )

CLI::

    python -m trl_bench.scripts.generate_table_embeddings \\
        --models bert --datasets spider_join

    # Default paths are cwd-relative:
    #   --column-embeddings-dir ./embeddings/column
    #   --output-dir            ./embeddings/table

    python -m trl_bench.scripts.generate_table_embeddings --force --dry-run
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from trl_bench.utils.aggregation.aggregator import (
    aggregate_embeddings,
    AggregationMethod,
)


_SHARD_RE = re.compile(r".*_shard\d+of\d+\.pkl$")


def _is_shard_file(pkl_name: str) -> bool:
    return bool(_SHARD_RE.match(pkl_name))


# Default paths relative to cwd (release convention).
DEFAULT_COL_EMB_DIR   = Path("embeddings/column")
DEFAULT_TABLE_EMB_DIR = Path("embeddings/table")


def discover_column_embeddings(
    col_emb_dir: Path,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Discover available column-embedding pkl files.

    Returns a dict mapping model name -> list of dataset names found.
    Skips checkpoint files and shard files (``*.checkpoint.pkl``,
    ``*_shardNofN.pkl``).
    """
    discovered: Dict[str, List[str]] = {}
    if not col_emb_dir.exists():
        return discovered

    for model_dir in sorted(col_emb_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if models and model_name not in models:
            continue

        model_datasets = []
        for pkl_file in sorted(model_dir.glob("*.pkl")):
            if pkl_file.name.endswith(".checkpoint.pkl"):
                continue
            if _is_shard_file(pkl_file.name):
                continue
            ds_name = pkl_file.stem
            if datasets and ds_name not in datasets:
                continue
            model_datasets.append(ds_name)

        if model_datasets:
            discovered[model_name] = model_datasets

    return discovered


def extract_table_embeddings(column_pkl_path: Path) -> List[Dict]:
    """Read a column-embedding pickle and emit a table-embedding list.

    Each output row is a dict::

        {
            "table_id": ...,
            "table_embedding": {
                "cls_embedding":   np.ndarray or None,
                "table_embedding": np.ndarray or None,  # native table emb
                "column_mean":     np.ndarray or None,
                "token_mean":      np.ndarray or None,
            },
            "model_name": ...,
            "embedding_dim": int,
        }
    """
    with open(column_pkl_path, "rb") as f:
        column_data = pickle.load(f)

    results: List[Dict] = []
    for item in column_data:
        table_id = item.get("table_id", "")
        model_name = item.get("model_name", "")
        embedding_dim = item.get("embedding_dim", 0)

        col_emb = item.get("column_embeddings") or item.get("column_embedding") or {}

        table_emb_block = item.get("table_embedding")
        table_emb_dict = table_emb_block if isinstance(table_emb_block, dict) else {}

        # cls_embedding
        cls_embedding = table_emb_dict.get("cls_embedding")
        if cls_embedding is None:
            cls_embedding = item.get("cls_embedding")
        if cls_embedding is not None:
            cls_embedding = np.asarray(cls_embedding, dtype=np.float32)

        # native table_embedding
        native_table_emb = table_emb_dict.get("table_embedding")
        if native_table_emb is not None:
            native_table_emb = np.asarray(native_table_emb, dtype=np.float32)

        # column_mean: prefer pre-computed, fall back to recompute from columns.
        column_mean = table_emb_dict.get("column_mean")
        if column_mean is not None:
            column_mean = np.asarray(column_mean, dtype=np.float32)
        elif col_emb:
            column_mean = aggregate_embeddings(col_emb, AggregationMethod.MEAN)

        # token_mean: forward-pass artefact; never recomputable from columns.
        token_mean = table_emb_dict.get("token_mean")
        if token_mean is not None:
            token_mean = np.asarray(token_mean, dtype=np.float32)

        if not embedding_dim:
            for emb in (column_mean, cls_embedding):
                if emb is not None:
                    embedding_dim = int(emb.shape[0])
                    break

        results.append({
            "table_id": table_id,
            "table_embedding": {
                "cls_embedding":   cls_embedding,
                "table_embedding": native_table_emb,
                "column_mean":     column_mean,
                "token_mean":      token_mean,
            },
            "model_name": model_name,
            "embedding_dim": embedding_dim,
        })

    return results


def process_model_dataset(
    model: str,
    dataset: str,
    col_emb_dir: Path,
    table_emb_dir: Path,
    *, force: bool = False, dry_run: bool = False,
) -> bool:
    """Run extraction for one (model, dataset) pair.

    Returns True if successfully processed (or correctly skipped), False
    if the column input is missing.
    """
    col_pkl = col_emb_dir / model / f"{dataset}.pkl"
    out_pkl = table_emb_dir / model / f"{dataset}.pkl"

    if not col_pkl.exists():
        print(f"  FAIL: column embeddings not found: {col_pkl}")
        return False

    if out_pkl.exists() and not force:
        print(f"  SKIP: already exists: {out_pkl}")
        return True

    if dry_run:
        print(f"  [DRY-RUN] would generate: {out_pkl}")
        return True

    t0 = time.time()
    table_data = extract_table_embeddings(col_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(table_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    n_tables = len(table_data)
    size_mb = out_pkl.stat().st_size / (1024 * 1024)
    print(f"  OK: {n_tables} tables, {size_mb:.1f}MB, {elapsed:.1f}s -> {out_pkl}")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate table embeddings from column embedding pickles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to process (default: all discovered)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to process (default: all discovered)")
    parser.add_argument("--column-embeddings-dir", type=Path,
                        default=DEFAULT_COL_EMB_DIR,
                        help=f"Column embeddings directory (default: {DEFAULT_COL_EMB_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TABLE_EMB_DIR,
                        help=f"Table embeddings output directory (default: {DEFAULT_TABLE_EMB_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing table embeddings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing")
    args = parser.parse_args(argv)

    print("Table Embedding Generation")
    print("=" * 60)

    available = discover_column_embeddings(
        args.column_embeddings_dir, args.models, args.datasets,
    )
    if not available:
        print("Error: no column embeddings found.")
        print(f"  Searched: {args.column_embeddings_dir}")
        return 1

    total_pairs = sum(len(ds) for ds in available.values())
    print(f"Found {len(available)} models, {total_pairs} (model, dataset) pairs\n")

    succeeded = 0
    failed = 0
    for model in sorted(available):
        datasets = available[model]
        print(f"\n{model} ({len(datasets)} datasets):")
        for dataset in datasets:
            ok = process_model_dataset(
                model, dataset,
                args.column_embeddings_dir, args.output_dir,
                force=args.force, dry_run=args.dry_run,
            )
            if ok:
                succeeded += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"Succeeded: {succeeded}, Failed: {failed}")
    if not args.dry_run:
        print(f"Output: {args.output_dir}")
    print("=" * 60)
    return 1 if failed > 0 else 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
