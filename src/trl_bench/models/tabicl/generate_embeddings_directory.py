"""
TabICL Directory-Mode Row Embedding Generation

Processes a directory of CSV files and produces an aggregate pickle
containing row embeddings for each table. No train/test split — each
CSV is embedded as a whole.

Embeddings are extracted from Stage 2 (row_interactor) using a forward
hook + predict_proba(), which ensures the full EnsembleGenerator
preprocessing pipeline runs (StandardScaler + PowerTransform +
OutlierRemover). Calling sub-modules directly would skip this and
produce NaN.

Output format: List[dict] pickle at --output_path, one entry per table.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse

import numpy as np
import pandas as pd
import torch

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)

MAX_TEST_CHUNK = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using TabICL"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for aggregate pickle",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1,
        help="Number of estimators (default: 1, >1 incoherent for embeddings due to RoPE)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--checkpoint_version",
        type=str,
        default="tabicl-classifier-v1.1-0506.ckpt",
        help="TabICL checkpoint version (default: tabicl-classifier-v1.1-0506.ckpt)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save intermediate results every N tables (default: 50)",
    )
    parser.add_argument(
        "--label_columns", type=str, nargs='*', default=None,
        help="Label columns to exclude from features",
    )
    parser.add_argument("--table_list", default=None, help="Path to table list file for shard filtering")
    return parser.parse_args()


def embed_table(clf, csv_path, label_columns=None):
    """Embed a single CSV file using TabICL forward hook pattern.

    Returns a table result dict, or None if the table cannot be processed.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: cannot read CSV: {e}")
        return None

    n = len(df)
    if n < 2:
        print(f"  SKIP {csv_path.name}: too few rows ({n})")
        return None

    X = df.copy()
    if label_columns:
        cols_to_drop = [c for c in label_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
    column_names = list(X.columns)

    # Replace inf with NaN — some web tables have division-by-zero values
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

    # Dummy 2-class labels — TabICLClassifier.fit() requires ≥2 classes.
    # Labels don't affect Stage 2 embeddings.
    dummy_y = np.zeros(n, dtype=int)
    dummy_y[n // 2 :] = 1

    try:
        clf.fit(X, dummy_y)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: fit failed: {e}")
        return None

    model = clf.model_
    captured_reps = []

    def capture_row_reps(module, input, output):
        captured_reps.append(output.detach().cpu())

    hook_handle = model.row_interactor.register_forward_hook(capture_row_reps)
    try:
        if n <= MAX_TEST_CHUNK:
            # Small table — embed in one pass.
            # predict_proba sees [context=X, test=X] so output is (1, 2*n, 512).
            # We want the second half (the "test" rows = our full table).
            captured_reps.clear()
            with torch.no_grad():
                clf.predict_proba(X)
            reps = captured_reps[0][0].numpy()  # (2*n, 512)
            embeddings = reps[n:]  # Second half = test embeddings
        else:
            # Large table — process in chunks.
            # First get a baseline pass with minimal test to anchor train context,
            # then chunk the full table as "test" data.
            chunks = []
            n_chunks = (n + MAX_TEST_CHUNK - 1) // MAX_TEST_CHUNK

            for i in range(n_chunks):
                start = i * MAX_TEST_CHUNK
                end = min(start + MAX_TEST_CHUNK, n)

                captured_reps.clear()
                with torch.no_grad():
                    clf.predict_proba(X.iloc[start:end])

                # After predict_proba, reps shape is (1, n_context + chunk_size, 512)
                # n_context = n (the fitted training data), chunk rows start at index n
                chunk_emb = captured_reps[0][0, n:].numpy()
                chunks.append(chunk_emb)
                print(f"    Chunk {i + 1}/{n_chunks}: rows {start}-{end - 1}")

            embeddings = np.concatenate(chunks, axis=0)
    except Exception as e:
        hook_handle.remove()
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None
    finally:
        hook_handle.remove()

    return build_table_result(
        table_path=str(csv_path),
        row_embeddings=embeddings,
        column_names=column_names,
        model_name="TabICL",
    )


def main():
    args = parse_args()

    print("=" * 80)
    print("TabICL Directory-Mode Row Embedding Generation")
    print("=" * 80)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Discover tables
    csv_files = discover_csv_files(args.input_dir, table_list_path=args.table_list)
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")

    if not csv_files:
        print("No CSV files found. Exiting.")
        sys.exit(0)

    # Resume support
    results = load_existing_results(args.output_path)
    completed = get_completed_table_ids(results)
    register_save_on_signal(results, args.output_path)
    if completed:
        print(f"Resuming: {len(completed)} tables already processed")

    # Initialize TabICL — re-used across tables (fit() is called per table)
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(
        n_estimators=args.n_estimators,
        device=device,
    )

    # Process tables
    newly_processed = 0
    for i, csv_path in enumerate(csv_files):
        table_id = csv_path.stem
        if table_id in completed:
            continue

        print(f"\n[{i + 1}/{len(csv_files)}] Processing {csv_path.name}...")
        result = embed_table(clf, csv_path, label_columns=args.label_columns)

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
