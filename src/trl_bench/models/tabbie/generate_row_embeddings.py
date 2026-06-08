"""
TABBIE Row Embedding Generation

Processes a directory of CSV files and produces an aggregate pickle
containing row embeddings for each table. Each data row is embedded
by creating a 2-row mini-table (header + data row), running it through
TABBIE's dual transformers, and extracting the CLS intersection.

Output format: List[dict] pickle at --output_path, one entry per table.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from project-level utils BEFORE adding TABBIE's dir to sys.path
from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)
from trl_bench.utils.table_list import load_table_list, filter_csv_files

# Now remove project root AND the cached utils modules so TABBIE's
# internal imports resolve correctly.
sys.path.remove(project_root)
for mod_name in list(sys.modules):
    if mod_name == "utils" or mod_name.startswith("utils."):
        del sys.modules[mod_name]

# TABBIE's internal imports need this directory on path
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv as csv_mod
import tempfile

import numpy as np

from csv_to_embeddings import TABBIEEmbedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using TABBIE"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output path for aggregate pickle",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to TABBIE weights.pt checkpoint",
    )
    parser.add_argument(
        "--device_id", type=int, default=None,
        help="GPU device ID (None=auto, -1=CPU)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=50,
        help="Save intermediate results every N tables (default: 50)",
    )
    parser.add_argument(
        "--label_columns", type=str, nargs="*", default=None,
        help="Label columns to exclude from features",
    )
    parser.add_argument("--table_list", default=None, help="Path to table list file for shard filtering")
    parser.add_argument(
        "--bert_model_name", type=str, default="bert-base-uncased",
        help="BERT model name or local path (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--row_batch_size", type=int, default=32,
        help="Number of rows to batch in each TABBIE forward pass (default: 32)",
    )
    return parser.parse_args()


def _write_csv_without_columns(src_path, dst_path, drop_columns):
    """Copy a CSV file, dropping specified columns while preserving raw cell values."""
    drop_set = set(drop_columns)
    with open(src_path, "r", newline="") as fin, open(dst_path, "w", newline="") as fout:
        reader = csv_mod.reader(fin)
        writer = csv_mod.writer(fout)
        header = next(reader)
        keep_indices = [i for i, col in enumerate(header) if col not in drop_set]
        writer.writerow([header[i] for i in keep_indices])
        for row in reader:
            padded = row + [""] * max(0, len(header) - len(row))
            writer.writerow([padded[i] for i in keep_indices])


def embed_table(embedder, csv_path, label_columns=None, row_batch_size=32):
    """Embed a single CSV file using TABBIE row-by-row processing.

    Returns a table result dict, or None if the table cannot be processed.
    """
    try:
        with open(csv_path, "r") as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            n_rows = sum(1 for _ in reader)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: cannot read CSV: {e}")
        return None

    if n_rows < 1:
        print(f"  SKIP {csv_path.name}: no data rows")
        return None

    label_set = set(label_columns) if label_columns else set()
    column_names = [c for c in header if c not in label_set]

    # TABBIE's embedder takes a CSV path, so write a temp CSV
    # when label columns are excluded; otherwise use original.
    embed_path = None
    try:
        if label_columns:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False,
            )
            embed_path = tmp.name
            tmp.close()
            _write_csv_without_columns(csv_path, embed_path, label_set)
        else:
            embed_path = str(csv_path)

        embeddings = embedder.csv_to_row_embeddings(
            csv_path=embed_path,
            row_batch_size=row_batch_size,
            output_format="numpy",
        )
        embeddings = embeddings.astype(np.float32)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None
    finally:
        if label_columns and embed_path and os.path.exists(embed_path):
            os.unlink(embed_path)

    return build_table_result(
        table_path=str(csv_path),
        row_embeddings=embeddings,
        column_names=column_names,
        model_name="tabbie",
    )


def main():
    args = parse_args()

    print("=" * 80)
    print("TABBIE Row Embedding Generation")
    print("=" * 80)

    # Initialize TABBIE embedder (reused across all tables)
    print(f"Loading TABBIE model from {args.model_path}...")
    embedder = TABBIEEmbedder(
        model_path=args.model_path,
        device_id=args.device_id,
        bert_model_name=args.bert_model_name,
    )

    # Discover tables (apply table_list filtering directly since TABBIE's
    # sys.path removal breaks the lazy import inside discover_csv_files)
    csv_files = discover_csv_files(args.input_dir)
    if args.table_list:
        table_list = load_table_list(args.table_list)
        csv_files = filter_csv_files(csv_files, table_list)
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")

    if not csv_files:
        sys.exit(0)

    # Resume support
    results = load_existing_results(args.output_path)
    completed = get_completed_table_ids(results)
    register_save_on_signal(results, args.output_path)
    if completed:
        print(f"Resuming: {len(completed)} tables already processed")

    # Process tables
    newly_processed = 0
    for i, csv_path in enumerate(csv_files):
        table_id = csv_path.stem
        if table_id in completed:
            continue

        print(f"\n[{i + 1}/{len(csv_files)}] Processing {csv_path.name}...")
        result = embed_table(
            embedder, csv_path,
            label_columns=args.label_columns,
            row_batch_size=args.row_batch_size,
        )

        if result is not None:
            results.append(result)
            newly_processed += 1
            print(
                f"  Embedded: {result['num_rows']} rows x {result['embedding_dim']} dim"
            )

        # Periodic checkpoint
        if newly_processed > 0 and newly_processed % args.checkpoint_interval == 0:
            save_aggregate_pickle(results, args.output_path)
            print(f"  Checkpoint saved ({len(results)} tables total)")

    # Final save
    if newly_processed > 0:
        save_aggregate_pickle(results, args.output_path)

    print(f"\n{'=' * 80}")
    print(f"Done. {len(results)} tables in {args.output_path}")
    print(f"  Newly processed: {newly_processed}")
    print(f"  Previously completed: {len(completed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
