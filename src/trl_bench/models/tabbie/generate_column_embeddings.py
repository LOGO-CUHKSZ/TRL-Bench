"""
TABBIE Column Embedding Generation

Processes a directory of CSV files and produces a unified v2.0 column
embedding pickle using TABBIE's dual-transformer architecture.

Per-column embeddings are extracted from the CLS row of the final column
transformer. The CLS table embedding (row/col intersection average) is
also preserved. column_mean is derived via aggregation.

Output format: List[dict] pickle at --output, one entry per table.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from project-level utils BEFORE adding TABBIE's dir to sys.path
from trl_bench.utils.aggregation import aggregate_embeddings
from trl_bench.utils.table_list import load_table_list

# Now remove project root AND the cached utils modules so TABBIE's
# internal imports resolve correctly.
sys.path.remove(project_root)
for mod_name in list(sys.modules):
    if mod_name == "utils" or mod_name.startswith("utils."):
        del sys.modules[mod_name]

# TABBIE's internal imports need this directory on path
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from csv_to_embeddings import TABBIEEmbedder


# =============================================================================
# Checkpoint/Resume Support
# =============================================================================

def load_checkpoint_data(output_path):
    """Load existing checkpoint or output file for resume support."""
    checkpoint_path = Path(output_path).with_suffix(".checkpoint.pkl")
    existing_results = []
    processed_tables = set()

    for path, label in [(checkpoint_path, "checkpoint"), (Path(output_path), "output")]:
        if path.exists() and not processed_tables:
            print(f"\nFound {label} file: {path}")
            try:
                with open(path, "rb") as f:
                    existing_results = pickle.load(f)
                if isinstance(existing_results, list):
                    processed_tables = {
                        e.get("table_name", e.get("table_id"))
                        for e in existing_results
                    }
                    print(f"  Loaded {len(existing_results)} already-processed tables from {label}")
            except Exception as e:
                print(f"  Warning: Failed to load {label}: {e}")
                existing_results = []
                processed_tables = set()

    return existing_results, processed_tables, checkpoint_path


def save_checkpoint(results, checkpoint_path):
    """Save current progress to checkpoint file."""
    try:
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(results, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate column embeddings for a directory of CSV files using TABBIE"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to CSV file or directory of CSV files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output pickle file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to TABBIE weights.pt checkpoint"
    )
    parser.add_argument(
        "--max_rows", type=int, default=30,
        help="Max data rows per table (default: 30, TABBIE's architectural limit)"
    )
    parser.add_argument(
        "--device_id", type=int, default=None, help="GPU device ID (None=auto, -1=CPU)"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=100,
        help="Save checkpoint every N tables (default: 100)"
    )
    parser.add_argument(
        "--bert_model_name", type=str, default="bert-base-uncased",
        help="BERT model name or local path (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--table_list", type=str, default=None,
        help="Path to file listing CSV basenames to process (for sharded runs)"
    )
    return parser.parse_args()


def read_column_names(csv_path):
    """Read column names from the first row of a CSV file."""
    df = pd.read_csv(csv_path, header=None, nrows=1, keep_default_na=False, dtype=str)
    return [str(df.iloc[0, j]) for j in range(len(df.columns))]


def embed_table_columns(embedder, csv_path, model_name="tabbie"):
    """Extract column embeddings from a single CSV file.

    Returns a v2.0 format dict, or None on failure.
    """
    try:
        result = embedder.csv_to_embeddings(
            csv_path=str(csv_path),
            aggregate="column",
            output_format="numpy",
        )
    except Exception as e:
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None

    col_embs = result["column_embeddings"]
    cls_emb = result["cls_embedding"]

    # Read column names from the header row
    try:
        column_names = read_column_names(csv_path)
        # Truncate to match the number of columns the embedder actually processed
        column_names = column_names[:len(col_embs)]
    except Exception:
        column_names = [str(j) for j in range(len(col_embs))]

    # Derive table embeddings via aggregation
    mean_emb = aggregate_embeddings(col_embs, "mean")

    table_name = Path(csv_path).stem

    return {
        "version": "2.0",
        "format": "unified_table_embedding",
        "table": str(csv_path),
        "table_id": table_name,
        "column_embeddings": col_embs,
        "column_names": column_names,
        "table_embedding": {
            "cls_embedding": cls_emb,
            "table_embedding": None,
            "column_mean": mean_emb,
        },
        "table_name": table_name,
        "model_name": model_name,
        "embedding_dim": cls_emb.shape[0],
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("TABBIE Column Embedding Generation")
    print("=" * 80)

    # Initialize embedder
    print(f"Loading TABBIE model from {args.model_path}...")
    embedder = TABBIEEmbedder(
        model_path=args.model_path,
        device_id=args.device_id,
        bert_model_name=args.bert_model_name,
        max_rows=args.max_rows,
    )

    is_directory = os.path.isdir(args.input)

    if not is_directory:
        # Single file mode
        print(f"\nProcessing file: {args.input}")
        result = embed_table_columns(embedder, Path(args.input))
        if result is None:
            print("Failed to generate embeddings.")
            sys.exit(1)
        with open(args.output, "wb") as f:
            pickle.dump(result, f, protocol=4)
        print(f"Saved to {args.output}")
        return

    # Directory mode
    csv_files = sorted(Path(args.input).glob("*.csv"))
    if args.table_list:
        _table_list = load_table_list(args.table_list)
        csv_files = [f for f in csv_files if f.name in _table_list]
    print(f"Found {len(csv_files)} CSV files in {args.input}")

    if not csv_files:
        print("No CSV files found.")
        sys.exit(0)

    # Resume support
    existing_results, processed_tables, checkpoint_path = load_checkpoint_data(args.output)
    results = list(existing_results)

    if processed_tables:
        original_count = len(csv_files)
        csv_files = [f for f in csv_files if f.stem not in processed_tables]
        skipped = original_count - len(csv_files)
        if skipped > 0:
            print(f"Skipping {skipped} already-processed tables")

    if not csv_files:
        print("All tables already processed")
        return

    start_time = time.time()
    tables_since_checkpoint = 0

    for i, csv_path in enumerate(tqdm(csv_files, desc="Encoding tables")):
        result = embed_table_columns(embedder, csv_path)
        if result is not None:
            results.append(result)
            tables_since_checkpoint += 1

        if (
            args.checkpoint_interval > 0
            and tables_since_checkpoint >= args.checkpoint_interval
        ):
            save_checkpoint(results, checkpoint_path)
            tables_since_checkpoint = 0

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f, protocol=4)

    # Clean up checkpoint
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print("Checkpoint file removed (processing complete).")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint: {e}")

    elapsed = time.time() - start_time
    new_tables = len(results) - len(existing_results)

    print(f"\n{'=' * 80}")
    print("TABBIE COLUMN EMBEDDING EXTRACTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Tables processed: {len(results)} total ({new_tables} new)")
    print(f"Embedding dimension: 768")
    print(f"Output saved to: {args.output}")
    print(f"Inference time: {elapsed:.2f} seconds")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
