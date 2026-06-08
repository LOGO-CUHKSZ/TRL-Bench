#!/usr/bin/env python3
"""
Generate column embeddings from CSV files using Starmie model.

This script outputs embeddings in the unified v2.0 format, compatible with
all downstream tasks and other models in the TRL benchmark.

Starmie uses contrastive learning (BarlowTwinsSimCLR) to learn column
representations from table data.

RESUME SUPPORT
==============
This script supports resuming from interruptions:
- Checkpoint file (.checkpoint.pkl) saved alongside output every N tables
- On restart, automatically detects and loads checkpoint
- Skips already-processed tables
- Checkpoint removed after successful completion

OUTPUT FORMAT (Unified v2.0)
============================
Pickle file containing list of dicts:

    [
        {
            'table': 'path/to/table.csv',
            'table_id': 'table_name',
            'column_embeddings': {0: array, 1: array, ...},
            'column_names': ['col1', 'col2', ...],
            'table_embedding': {
                'cls_embedding': None,
                'table_embedding': None,
                'column_mean': array,
            }
        },
        ...
    ]

USAGE
=====
    # Basic usage
    python generate_column_embeddings.py \
        --model_path path/to/model.pt \
        --input_dir path/to/tables/ \
        --output_path embeddings.pkl

    # With checkpoint interval (for large datasets)
    python generate_column_embeddings.py \
        --model_path path/to/model.pt \
        --input_dir path/to/tables/ \
        --output_path embeddings.pkl \
        --checkpoint_interval 100

LEGACY FORMAT CONVERSION
========================
To convert unified format to Starmie's legacy formats for union/join search:

    python utils/convert_unified_to_starmie.py \
        --input embeddings.pkl \
        --output_format union_search \
        --output union_search_embeddings.pkl

Reference: "Starmie: Data Discovery with Column Annotations" (Fan et al., VLDB 2023)
"""

import os
import sys
import csv
import glob
import pickle
import time
import argparse
from pathlib import Path

# Increase CSV field size limit to handle large fields (e.g., GeoJSON data)
# before truncating them to a reasonable size for embeddings
csv.field_size_limit(sys.maxsize)

# Maximum field size in characters for embedding purposes
# Fields larger than this will be truncated (default: 100KB)
MAX_FIELD_SIZE = 102400

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path. Also add this model's own directory so the
# vendored ``sdd`` package (Starmie/PVLDB-2023's upstream module name) is
# importable as a top-level module — matches the tuta/tabbie/tabsketchfm
# pattern for wrappers that inherit absolute imports from their upstream
# repos.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sdd.pretrain import load_checkpoint, inference_on_tables
from trl_bench.utils.aggregation import aggregate_embeddings


def extract_table_id(table_path: str) -> str:
    """
    Extract table_id from a table path.

    Removes directory path and file extension to get canonical identifier.
    Handles compression suffixes (.gz, .bz2) and double extensions (.csv.gz).
    """
    basename = os.path.basename(table_path)
    # Strip compression suffixes first
    for ext in ['.gz', '.bz2']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
    # Then strip data format extensions
    for ext in ['.csv', '.json', '.tsv', '.parquet']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
    return basename


def load_single_table(file_path: str, max_rows: int = 1000,
                      max_field_size: int = MAX_FIELD_SIZE) -> pd.DataFrame:
    """
    Load a single CSV file into a DataFrame.

    Args:
        file_path: Path to CSV file
        max_rows: Maximum rows to read
        max_field_size: Maximum characters per field (larger fields are truncated)

    Returns:
        DataFrame containing the table data
    """
    df = pd.read_csv(
        file_path,
        encoding='utf-8',
        on_bad_lines='skip',
        engine='python',
        encoding_errors='ignore',
        nrows=max_rows,
        dtype=str  # Prevent pandas from parsing IDs like "1E123" as scientific notation
    )
    # Clean column names
    df.columns = df.columns.str.strip().str.rstrip('\r\n')

    # Handle duplicate column names
    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index.tolist()
            cols.iloc[dup_indices] = [
                f"{dup}.{i}" if i > 0 else dup
                for i in range(len(dup_indices))
            ]
        df.columns = cols

    # Truncate large fields to prevent memory issues and improve embedding quality
    # (e.g., GeoJSON geometry data can be megabytes but isn't useful for text embeddings)
    if max_field_size > 0:
        for col in df.columns:
            mask = df[col].str.len() > max_field_size
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].str[:max_field_size] + '...[truncated]'

    return df


def build_result_entry(file_path: str, df: pd.DataFrame, vectors: list) -> dict:
    """
    Build a unified v2.0 format result entry for a table.

    Args:
        file_path: Path to the source table
        df: DataFrame containing the table data
        vectors: List of column embedding vectors

    Returns:
        Dict in unified v2.0 format
    """
    column_names = [str(c) for c in df.columns.tolist()]

    # Validate vector count
    if len(vectors) != len(column_names):
        num_cols = min(len(vectors), len(column_names))
        column_names = column_names[:num_cols]
        vectors = vectors[:num_cols]
    else:
        num_cols = len(column_names)

    # Build column_embeddings dict
    column_embeddings = {
        col_idx: np.array(vectors[col_idx], dtype=np.float32)
        for col_idx in range(num_cols)
    }

    # Compute table-level embeddings
    table_embedding = {
        'cls_embedding': None,
        'table_embedding': None,
        'column_mean': aggregate_embeddings(column_embeddings, 'mean'),
    }

    return {
        'table': file_path,
        'table_id': extract_table_id(file_path),
        'column_embeddings': column_embeddings,
        'column_names': column_names,
        'table_embedding': table_embedding,
    }


def load_checkpoint_data(output_path: str):
    """
    Load existing checkpoint or output file.

    Args:
        output_path: Path to the output file

    Returns:
        Tuple of (existing_results, processed_tables, checkpoint_path)
    """
    checkpoint_path = Path(output_path).with_suffix('.checkpoint.pkl')
    existing_results = []
    processed_tables = set()

    # Check for checkpoint first, then final output
    if checkpoint_path.exists():
        print(f"\nFound checkpoint file: {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                existing_results = pickle.load(f)
            processed_tables = {e['table'] for e in existing_results}
            print(f"  Loaded {len(existing_results)} already-processed tables from checkpoint")
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint: {e}")
            existing_results = []
            processed_tables = set()
    elif Path(output_path).exists():
        print(f"\nFound existing output file: {output_path}")
        try:
            with open(output_path, 'rb') as f:
                existing_results = pickle.load(f)
            processed_tables = {e['table'] for e in existing_results}
            print(f"  Loaded {len(existing_results)} already-processed tables from output")
        except Exception as e:
            print(f"  Warning: Failed to load output file: {e}")
            existing_results = []
            processed_tables = set()

    return existing_results, processed_tables, checkpoint_path


def save_checkpoint(results: list, checkpoint_path: Path):
    """
    Save current progress to checkpoint file.

    Args:
        results: List of result entries
        checkpoint_path: Path to checkpoint file
    """
    try:
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        with open(tmp_path, 'wb') as f:
            pickle.dump(results, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings using Starmie (unified v2.0 format)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained Starmie checkpoint (.pt)')
    parser.add_argument('--input_dir', '--input', type=str, required=True, dest='input_dir',
                        help='Path to directory containing CSV files')
    parser.add_argument('--output_path', '--output', type=str, required=True, dest='output_path',
                        help='Output pickle file path')
    parser.add_argument('--max_rows', type=int, default=1000,
                        help='Maximum rows to read per table (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for model inference (default: 1024)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N tables (default: 100). Set to 0 to disable.')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("STARMIE EMBEDDING GENERATION (Unified v2.0 Format)")
    print("=" * 70)
    print(f"Model checkpoint:     {args.model_path}")
    print(f"Input directory:      {args.input_dir}")
    print(f"Output path:          {args.output_path}")
    print(f"Max rows/table:       {args.max_rows}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Checkpoint interval:  {args.checkpoint_interval} tables")
    print("=" * 70)

    # =========================================================================
    # RESUME SUPPORT: Load existing checkpoint if available
    # =========================================================================
    existing_results, processed_tables, checkpoint_path = load_checkpoint_data(args.output_path)

    # Get list of all CSV files (including dot-prefixed files)
    all_csv_files = sorted(
        entry.path
        for entry in os.scandir(args.input_dir)
        if entry.is_file() and entry.name.endswith(".csv")
    )
    if args.table_list:
        from trl_bench.utils.table_list import load_table_list
        _table_list = load_table_list(args.table_list)
        all_csv_files = [f for f in all_csv_files if os.path.basename(f) in _table_list]
    total_files = len(all_csv_files)

    # Filter out already-processed files
    if processed_tables:
        csv_files = [f for f in all_csv_files if f not in processed_tables]
        skipped = total_files - len(csv_files)
        print(f"\nResume mode:")
        print(f"  Total files:     {total_files}")
        print(f"  Already done:    {skipped}")
        print(f"  Remaining:       {len(csv_files)}")

        if not csv_files:
            print("\nAll tables already processed. Nothing to do.")
            # Ensure final output exists
            if not Path(args.output_path).exists():
                print(f"Saving final output to {args.output_path}...")
                with open(args.output_path, 'wb') as f:
                    pickle.dump(existing_results, f, protocol=4)
            # Clean up checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Checkpoint file removed.")
            return
    else:
        csv_files = all_csv_files
        print(f"\nTotal files to process: {len(csv_files)}")

    # =========================================================================
    # Load model once
    # =========================================================================
    print("\nLoading Starmie model...")
    ckpt = torch.load(args.model_path, map_location=torch.device('cuda'), weights_only=False)
    # The checkpoint's ``hp.data_path``/``hp.task`` is baked-in to the upstream
    # author's filesystem (``/u3/...``); override to the CLI ``--input_dir``
    # so the dataset constructor walks our local CSVs.
    model, trainset = load_checkpoint(ckpt, ds_path_override=args.input_dir)
    model.eval()
    print("Model loaded.")

    # =========================================================================
    # Process tables in batches with checkpointing
    # =========================================================================
    print("\nProcessing tables...")
    start_time = time.time()

    # Combine existing results with new ones
    results = list(existing_results)  # Make a copy
    tables_since_checkpoint = 0

    # Process in batches for efficiency
    batch_dfs = []
    batch_paths = []

    pbar = tqdm(csv_files, desc="Extracting embeddings")
    for file_path in pbar:
        # Load table
        df = load_single_table(file_path, max_rows=args.max_rows)
        batch_dfs.append(df)
        batch_paths.append(file_path)

        # Process batch when it reaches batch_size or at the end
        if len(batch_dfs) >= args.batch_size or file_path == csv_files[-1]:
            if batch_dfs:
                # Run inference on batch
                column_vectors = inference_on_tables(
                    batch_dfs, model, trainset,
                    batch_size=args.batch_size,
                    total=len(batch_dfs)
                )

                # Build results for each table in batch
                for i, (path, df) in enumerate(zip(batch_paths, batch_dfs)):
                    result_entry = build_result_entry(path, df, column_vectors[i])
                    results.append(result_entry)
                    tables_since_checkpoint += 1

                # Clear batch
                batch_dfs = []
                batch_paths = []

                # Save checkpoint at regular intervals
                if args.checkpoint_interval > 0 and tables_since_checkpoint >= args.checkpoint_interval:
                    save_checkpoint(results, checkpoint_path)
                    pbar.set_postfix({'saved': len(results)})
                    tables_since_checkpoint = 0

    pbar.close()
    inference_time = time.time() - start_time

    # =========================================================================
    # Save final results and clean up
    # =========================================================================
    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"Checkpoint file removed (processing complete).")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint file: {e}")

    # Summary
    embedding_dim = len(list(results[0]['column_embeddings'].values())[0]) if results else 0
    total_columns = sum(len(r['column_embeddings']) for r in results)
    new_tables = len(results) - len(existing_results)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Tables processed:    {len(results)} total ({new_tables} new)")
    print(f"  Total columns:       {total_columns}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Output format:       Unified v2.0")
    print(f"  Output file:         {args.output_path}")
    print(f"  Inference time:      {inference_time:.2f} seconds")
    print("=" * 70)
    print("\nTo convert to Starmie legacy formats, use:")
    print("  python utils/convert_unified_to_starmie.py --help")


if __name__ == '__main__':
    main()
