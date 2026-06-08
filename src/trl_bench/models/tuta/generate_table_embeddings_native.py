"""
TUTA Native Table Embedding Generation

Processes a directory of CSV files and produces a table embedding pickle
directly using TUTA's native [CLS] token. Unlike the column→table derivation
pipeline, this script runs TUTA's forward pass on whole tables and extracts
the [CLS] token as a table-level representation.

This aligns with how TUTA was pre-trained (TCR objective) and fine-tuned
(TTC task) — both use the [CLS] token as the table representation.

Output format: List[dict] pickle at --output_path, one entry per table.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from project-level utils BEFORE adding TUTA's dir to sys.path,
# because TUTA has its own utils/ that would shadow the project-level one.
from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)
from trl_bench.utils.table_list import load_table_list, filter_csv_files

# Now remove project root AND the cached utils modules so TUTA's
# internal `from utils import ...` resolves to tuta/utils.py,
# not the project-level utils/ package.
sys.path.remove(project_root)
for mod_name in list(sys.modules):
    if mod_name == "utils" or mod_name.startswith("utils."):
        del sys.modules[mod_name]

# TUTA's internal imports need the tuta subdirectory on path
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np

from csv_to_embeddings import TUTAEmbedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate native table embeddings for a directory of CSV files using TUTA [CLS]"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for aggregate pickle (e.g., embeddings/table/tuta/dataset.pkl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to TUTA .bin checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="tuta",
        choices=["tuta", "tuta_explicit", "base"],
        help="TUTA model variant (default: tuta)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="GPU device ID (None=auto, -1=CPU)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save intermediate results every N tables (default: 50)",
    )
    parser.add_argument(
        "--table_list",
        type=str,
        default=None,
        help="Path to file listing CSV basenames to process (for sharded runs)",
    )
    return parser.parse_args()


def embed_table_native(embedder, csv_path, model_name="tuta"):
    """Extract native [CLS] table embedding from a single CSV file.

    Returns a table embedding result dict. Raises on failure — tables
    must never be silently skipped.
    """
    # aggregate='cls' uses multi-sequence aggregation → (1, 768)
    cls_emb = embedder.csv_to_embeddings(
        csv_path=str(csv_path),
        aggregate="cls",
        output_format="numpy",
    )
    # Squeeze from (1, 768) to (768,)
    cls_emb = cls_emb.squeeze(0).astype(np.float32)

    table_id = csv_path.stem

    return {
        "table_id": table_id,
        "table_embedding": {
            "cls_embedding": cls_emb,
            "table_embedding": None,
            "column_mean": None,
        },
        "model_name": model_name,
        "embedding_dim": cls_emb.shape[0],
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("TUTA Native Table Embedding Generation")
    print("=" * 80)

    # Initialize TUTA embedder (reused across all tables)
    print(f"Loading TUTA model from {args.model_path}...")
    embedder = TUTAEmbedder(
        model_path=args.model_path,
        target=args.model_type,
        device_id=args.device_id,
    )

    # Discover tables (optionally filtered by table list for sharded runs)
    csv_files = discover_csv_files(args.input_dir)
    if args.table_list:
        table_list = load_table_list(args.table_list)
        csv_files = filter_csv_files(csv_files, table_list)
        print(f"Shard: {len(csv_files)} CSV files (from {args.table_list})")
    else:
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
        result = embed_table_native(embedder, csv_path, model_name=args.model_type)
        results.append(result)
        newly_processed += 1
        print(f"  Embedded: {result['embedding_dim']} dim")

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
