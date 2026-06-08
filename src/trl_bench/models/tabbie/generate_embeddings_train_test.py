#!/usr/bin/env python3
"""
TABBIE Split-Aware Row Embedding Generation

Generates row-level embeddings using TABBIE's dual-transformer architecture
for canonical datasets with train/test splits (datasets/row_data/).

Each data row is embedded by creating a 2-row mini-table (header + data row),
running through TABBIE's 12 alternating row/column transformers, and extracting
the CLS intersection.

Supports multiple input modes:
- Pre-split: --data_dir with train.csv and test.csv (or canonical dataset.json)
- Single directory: canonical or legacy layouts auto-detected

Two modes:
1. With label_column: Generates embeddings WITHOUT the label column, extracts labels separately
2. Without label_column: Generates embeddings on full tables

Output: JSON metadata (metadata.json) with v2.0 split-aware format

Usage:
    python generate_embeddings_train_test.py \
        --data_dir datasets/row_data/openml_1486 \
        --embedding_dir embeddings/row_prediction/tabbie/openml_1486 \
        --model_path checkpoints/tabbie/weights.pt \
        --label_policy manifest
"""

import argparse
import os
import sys
import tempfile
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path FIRST for project-level utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, is_regression_label

# Now remove project root AND cached utils modules so TABBIE's
# internal imports resolve correctly.
sys.path.remove(project_root)
for mod_name in list(sys.modules):
    if mod_name == "utils" or mod_name.startswith("utils."):
        del sys.modules[mod_name]

# TABBIE's internal imports need this directory on path
sys.path.insert(0, os.path.dirname(__file__))

from csv_to_embeddings import TABBIEEmbedder

logger = logging.getLogger(__name__)


def write_split_csv(view, label_cols, temp_dir):
    """Write a SplitView to a temporary CSV file, without any label columns."""
    if label_cols and view.y is not None:
        df = view.X.copy()
        # Reconstruct label columns
        if isinstance(view.y, pd.DataFrame):
            for col in label_cols:
                if col in view.y.columns:
                    df[col] = view.y[col]
        else:
            df[label_cols[0]] = view.y.values if hasattr(view.y, 'values') else view.y
        # Drop ALL label columns
        df_out = df.drop(columns=[c for c in label_cols if c in df.columns])
    else:
        df_out = view.X

    temp_path = os.path.join(temp_dir, f"temp_{os.getpid()}_{id(view)}.csv")
    df_out.to_csv(temp_path, index=False)
    return temp_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate TABBIE row-level embeddings for datasets'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data (supports canonical, legacy, or single CSV layouts)')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory to save embeddings and labels')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to TABBIE weights.pt checkpoint')
    parser.add_argument('--label_column', type=str, default=None,
                        help='Name of label column (if classification task)')
    parser.add_argument(
        '--label_policy',
        type=str,
        default='auto',
        choices=['auto', 'none', 'manifest', 'cli'],
        help=(
            "Label resolution policy: auto (legacy behavior), none (force unlabeled), "
            "manifest (use dataset.json labels), cli (require --label_column)."
        ),
    )
    parser.add_argument('--device_id', type=int, default=None,
                        help='GPU device ID (None=auto-detect, -1=CPU only)')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased',
                        help='BERT model name or local path (default: bert-base-uncased)')
    parser.add_argument('--row_batch_size', type=int, default=32,
                        help='Number of rows to batch in each TABBIE forward pass (default: 32)')
    parser.add_argument('--ignore_fingerprint', action='store_true',
                        help='Skip SHA256 fingerprint verification for canonical datasets')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TABBIE Row Embeddings Generation (Split-Aware)")
    print(f"{'='*80}")
    print(f"Dataset: {args.data_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Label column: {args.label_column if args.label_column else 'None (full table)'}")
    print(f"{'='*80}\n")

    # ========================================================================
    # 1. Load data via TableDataset
    # ========================================================================
    print("1. Loading data...")
    label_columns_cli = resolve_label_columns_cli(args.label_column, args.label_policy)
    dataset = load_table_dataset(
        args.data_dir,
        label_columns_cli=label_columns_cli,
        ignore_fingerprint=args.ignore_fingerprint,
    )

    # Single-CSV / all-only layout: split into train/test before embedding
    if dataset.split_names == ["all"]:
        dataset.apply_train_test_split(stratify_on_label=bool(dataset.label_columns))
        print("   Split single-CSV dataset into train/test (ratio=0.8, seed=42)")

    print(f"   Dataset: {dataset}")

    # Resolve effective label columns
    if args.label_policy == 'manifest':
        label_cols = list(dataset.label_columns)
    elif args.label_column:
        if args.label_column in dataset.label_columns:
            label_cols = [args.label_column]
        elif args.label_policy == 'cli':
            raise ValueError(
                f"Label column '{args.label_column}' not found in dataset. "
                f"Available labels: {dataset.label_columns}"
            )
        else:
            print(f"   Warning: label column '{args.label_column}' was not found; continuing without labels")
            label_cols = []
    else:
        label_cols = []

    has_labels = bool(label_cols)

    # ========================================================================
    # 2. Initialize TABBIE embedder
    # ========================================================================
    print("\n2. Initializing TABBIE embedder...")
    embedder = TABBIEEmbedder(
        model_path=args.model_path,
        device_id=args.device_id,
        bert_model_name=args.bert_model_name,
    )
    print("   Embedder initialized")

    # ========================================================================
    # 3. Prepare label encoders
    # ========================================================================
    label_encoders = {}
    if has_labels:
        print(f"\n3. Preparing label encoders...")
        full_view = dataset.get_full()
        for col in label_cols:
            y_col = full_view.y[col] if isinstance(full_view.y, pd.DataFrame) else full_view.y
            if is_regression_label(y_col, dataset.label_task_types, col):
                label_encoders[col] = None
                print(f"   Label '{col}': regression (raw values will be saved)")
            else:
                le = LabelEncoder()
                le.fit(y_col)
                label_encoders[col] = le
                print(f"   Label '{col}': classification ({le.classes_.tolist()})")
    else:
        print(f"\n3. No label columns (full table mode)")

    # ========================================================================
    # 4. Generate embeddings per split
    # ========================================================================
    print(f"\n4. Generating embeddings per split...")

    emb_dict = {}
    lbl_dict = {}
    idx_dict = {}

    temp_dir = tempfile.mkdtemp()

    try:
        for split_name in dataset.split_names:
            view = dataset.get_split(split_name)
            print(f"\n   Processing split '{split_name}' ({len(view)} samples)...")

            # Write split to temp CSV (without label column if specified)
            temp_csv = write_split_csv(view, label_cols, temp_dir)

            # Generate row embeddings
            embeddings = embedder.csv_to_row_embeddings(
                temp_csv,
                row_batch_size=args.row_batch_size,
                output_format='numpy',
            )
            embeddings = embeddings.astype(np.float32)
            emb_dict[split_name] = embeddings
            print(f"   Embeddings shape: {embeddings.shape}")

            # Clean up temp file
            os.unlink(temp_csv)

            # Encode labels if available
            if has_labels and view.y is not None:
                per_col = {}
                for col in label_cols:
                    y_col = view.y[col] if isinstance(view.y, pd.DataFrame) else view.y
                    le = label_encoders.get(col)
                    per_col[col] = encode_label_column(y_col, le, split_name, col, logger)
                lbl_dict[split_name] = per_col

            # Row indices if available
            if view.row_indices is not None:
                idx_dict[split_name] = view.row_indices

    finally:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    # ========================================================================
    # 5. Save with v2.0 split-aware format
    # ========================================================================
    print(f"\n5. Saving embeddings...")

    first_split = next(iter(emb_dict.values()))
    feature_columns = list(dataset.feature_columns)

    generation_config = {
        'data_source': str(args.data_dir),
        'model_path': str(args.model_path),
        'label_policy': args.label_policy,
        'row_batch_size': args.row_batch_size,
        'bert_model_name': args.bert_model_name,
    }

    dataset_info = {
        'source_path': dataset.source_path,
        'layout': dataset.layout,
    }
    if dataset.fingerprint:
        dataset_info['fingerprint_sha256'] = dataset.fingerprint

    metadata = RowEmbeddingMetadataV2(
        model_name='tabbie',
        embedding_dim=first_split.shape[1],
        label_columns=label_cols,
        label_task_types={c: dataset.label_task_types.get(c, '') for c in label_cols},
        feature_columns=feature_columns,
        generation_config=generation_config,
        dataset=dataset_info,
        checkpoint_path=str(args.model_path),
    )

    output_files = save_split_embeddings(
        embeddings=emb_dict,
        metadata=metadata,
        output_dir=str(output_dir),
        labels=lbl_dict if lbl_dict else None,
        row_indices=idx_dict if idx_dict else None,
    )

    print(f"   Saved:")
    for file_type, file_path in output_files.items():
        print(f"      {file_path}")

    print(f"\n{'='*80}")
    print("Embedding generation completed successfully!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Embedding dimension: {first_split.shape[1]}")
    for name, emb in emb_dict.items():
        print(f"  {name} embeddings: {emb.shape}")
    if has_labels:
        print(f"  Label columns: {label_cols}")
        for col in label_cols:
            le = label_encoders.get(col)
            if le is not None:
                print(f"    '{col}': classification ({le.classes_.tolist()})")
            else:
                print(f"    '{col}': regression (raw values saved)")
    else:
        print(f"  No labels (full table mode)")
    print(f"\nAll files in: {output_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
