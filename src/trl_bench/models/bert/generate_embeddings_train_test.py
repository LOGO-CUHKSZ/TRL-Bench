#!/usr/bin/env python3
"""
BERT Split-Aware Row Embedding Generation

Generates row-level embeddings using BERT for canonical datasets with
train/test splits (datasets/row_data/). Each data row is serialized as
"col1: val1 | col2: val2 | ..." and encoded through BERT, extracting
the [CLS] token embedding.

Output: JSON metadata (metadata.json) with v2.0 split-aware format

Usage:
    python generate_embeddings_train_test.py \
        --data_dir datasets/row_data/openml_1486 \
        --embedding_dir embeddings/row_prediction/bert/openml_1486 \
        --label_policy manifest
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, is_regression_label

import torch
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)


def serialize_row(columns, row_values, max_chars_per_cell=50):
    """Serialize a single data row as 'col: val | col: val | ...'."""
    pairs = []
    for col, val in zip(columns, row_values):
        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val)[:max_chars_per_cell]
        pairs.append(f"{col}: {val_str}")
    return " | ".join(pairs)


class BERTRowEmbedder:
    """Encodes table rows as text and produces [CLS] embeddings via BERT."""

    def __init__(self, model_name="bert-base-uncased", device="auto",
                 max_length=512, max_chars_per_cell=50):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.max_chars_per_cell = max_chars_per_cell

    def encode_rows(self, df, batch_size=64):
        """Encode all rows of a DataFrame into embeddings.

        Args:
            df: DataFrame with feature columns only (no label columns).
            batch_size: Number of rows per forward pass.

        Returns:
            np.ndarray of shape (n_rows, 768) with float32 dtype.
        """
        texts = [
            serialize_row(df.columns, row, self.max_chars_per_cell)
            for _, row in df.iterrows()
        ]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls)

        if not all_embeddings:
            dim = self.model.config.hidden_size
            return np.empty((0, dim), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Generate BERT row-level embeddings for datasets'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data (supports canonical, legacy, or single CSV layouts)')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory to save embeddings and labels')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='HuggingFace BERT model name or path (default: bert-base-uncased)')
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
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, etc. (default: auto)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum token length for BERT tokenizer (default: 512)')
    parser.add_argument('--max_chars_per_cell', type=int, default=50,
                        help='Max characters per cell value before truncation (default: 50)')
    parser.add_argument('--row_batch_size', type=int, default=64,
                        help='Number of rows per forward pass (default: 64)')
    parser.add_argument('--ignore_fingerprint', action='store_true',
                        help='Skip SHA256 fingerprint verification for canonical datasets')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"BERT Row Embeddings Generation (Split-Aware)")
    print(f"{'='*80}")
    print(f"Dataset: {args.data_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Max length: {args.max_length}")
    print(f"Max chars/cell: {args.max_chars_per_cell}")
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
    # 2. Initialize BERT embedder
    # ========================================================================
    print("\n2. Initializing BERT embedder...")
    embedder = BERTRowEmbedder(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        max_chars_per_cell=args.max_chars_per_cell,
    )
    print(f"   Embedder initialized on {embedder.device}")

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

    for split_name in dataset.split_names:
        view = dataset.get_split(split_name)
        print(f"\n   Processing split '{split_name}' ({len(view)} samples)...")

        # Encode rows directly from the DataFrame — no temp CSV needed
        embeddings = embedder.encode_rows(view.X, batch_size=args.row_batch_size)
        emb_dict[split_name] = embeddings
        print(f"   Embeddings shape: {embeddings.shape}")

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

    # ========================================================================
    # 5. Save with v2.0 split-aware format
    # ========================================================================
    print(f"\n5. Saving embeddings...")

    first_split = next(iter(emb_dict.values()))
    feature_columns = list(dataset.feature_columns)

    generation_config = {
        'data_source': str(args.data_dir),
        'model': args.model,
        'max_length': args.max_length,
        'max_chars_per_cell': args.max_chars_per_cell,
        'row_batch_size': args.row_batch_size,
        'label_policy': args.label_policy,
        'device': str(embedder.device),
    }

    dataset_info = {
        'source_path': dataset.source_path,
        'layout': dataset.layout,
    }
    if dataset.fingerprint:
        dataset_info['fingerprint_sha256'] = dataset.fingerprint

    metadata = RowEmbeddingMetadataV2(
        model_name='bert',
        embedding_dim=first_split.shape[1],
        label_columns=label_cols,
        label_task_types={c: dataset.label_task_types.get(c, '') for c in label_cols},
        feature_columns=feature_columns,
        generation_config=generation_config,
        dataset=dataset_info,
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
