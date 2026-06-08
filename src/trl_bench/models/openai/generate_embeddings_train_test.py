#!/usr/bin/env python3
"""
OpenAI Split-Aware Row Embedding Generation

Generates row-level embeddings using OpenAI text-embedding-3-small for
canonical datasets with train/test splits (datasets/row_data/). Each data
row is serialized as "col1: val1 | col2: val2 | ..." — same as GTE.

Output: JSON metadata (metadata.json) with v2.0 split-aware format

Usage:
    python generate_embeddings_train_test.py \
        --data_dir datasets/row_data/openml_1486 \
        --embedding_dir embeddings/row_prediction/openai/openml_1486 \
        --label_policy manifest
"""

import argparse
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import tiktoken
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../' * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.openai.client import create_client, resolve_model_name
from trl_bench.utils.unified_embedding_format import RowEmbeddingMetadataV2, save_split_embeddings, encode_label_column
from trl_bench.utils.table_dataset import load_table_dataset, resolve_label_columns_cli, is_regression_label

logger = logging.getLogger(__name__)

OPENAI_MAX_TOKENS = 8191
_enc = None

def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding('cl100k_base')
    return _enc

def truncate_text(text, max_tokens=OPENAI_MAX_TOKENS):
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def serialize_row(columns, row_values, max_chars_per_cell=100):
    pairs = []
    for col, val in zip(columns, row_values):
        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val)[:max_chars_per_cell]
        pairs.append(f"{col}: {val_str}")
    return " | ".join(pairs)


class OpenAIRowEmbedder:
    def __init__(self, model_name="text-embedding-3-small",
                 dimensions=None, max_chars_per_cell=100):
        from models.openai.client import get_model_info
        self.client, provider = create_client()
        self.model_name = resolve_model_name(model_name, provider)
        _, native_dim, self._supports_dimensions = get_model_info(model_name)
        self.max_chars_per_cell = max_chars_per_cell

        if dimensions is not None and self._supports_dimensions:
            self.dimensions = dimensions
        else:
            self.dimensions = native_dim if not self._supports_dimensions else (dimensions or native_dim)

        print(f"  Row embedder ({provider}): model={self.model_name}, dim={self.dimensions}")

    def _make_kwargs(self, input_texts):
        kwargs = dict(input=input_texts, model=self.model_name)
        if self._supports_dimensions:
            kwargs['dimensions'] = self.dimensions
        return kwargs

    def _embed_batch(self, batch):
        """Embed a batch of texts, splitting on token limit errors."""
        import openai
        try:
            response = self.client.embeddings.create(**self._make_kwargs(batch))
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]
        except (openai.BadRequestError, ValueError) as e:
            if len(batch) > 1:
                mid = len(batch) // 2
                left = self._embed_batch(batch[:mid])
                right = self._embed_batch(batch[mid:])
                return left + right
            # Singleton: re-raise hard API errors (bad model, bad dimensions, etc.)
            if isinstance(e, openai.BadRequestError):
                raise
            # ValueError (e.g. "No embedding data received") — zero vector fallback
            print(f"  WARNING: single-text embedding failed ({type(e).__name__}: {e}), using zero vector")
            return [np.zeros(self.dimensions, dtype=np.float32)]

    def encode_rows(self, df, batch_size=256):
        texts = [
            serialize_row(df.columns, row, self.max_chars_per_cell)
            for _, row in df.iterrows()
        ]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [truncate_text(t) for t in texts[i:i + batch_size]]
            batch = [t if t.strip() else " " for t in batch]
            batch_embs = self._embed_batch(batch)
            all_embeddings.extend(batch_embs)

        if not all_embeddings:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Generate OpenAI row-level embeddings for datasets'
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--embedding_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='text-embedding-3-small')
    parser.add_argument('--dimensions', type=int, default=768)
    parser.add_argument('--label_column', type=str, default=None)
    parser.add_argument('--label_policy', type=str, default='auto',
                        choices=['auto', 'none', 'manifest', 'cli'])
    parser.add_argument('--max_chars_per_cell', type=int, default=100)
    parser.add_argument('--row_batch_size', type=int, default=256)
    parser.add_argument('--ignore_fingerprint', action='store_true')

    args = parser.parse_args()

    output_dir = Path(args.embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"OpenAI Row Embeddings Generation (Split-Aware)")
    print(f"{'='*80}")
    print(f"Dataset: {args.data_dir}")
    print(f"Output: {output_dir}")

    # 1. Load data
    print("1. Loading data...")
    label_columns_cli = resolve_label_columns_cli(args.label_column, args.label_policy)
    dataset = load_table_dataset(
        args.data_dir,
        label_columns_cli=label_columns_cli,
        ignore_fingerprint=args.ignore_fingerprint,
    )

    if dataset.split_names == ["all"]:
        dataset.apply_train_test_split(stratify_on_label=bool(dataset.label_columns))

    print(f"   Dataset: {dataset}")

    # Resolve labels
    if args.label_policy == 'manifest':
        label_cols = list(dataset.label_columns)
    elif args.label_column:
        if args.label_column in dataset.label_columns:
            label_cols = [args.label_column]
        else:
            label_cols = []
    else:
        label_cols = []

    has_labels = bool(label_cols)

    # 2. Initialize embedder
    print("\n2. Initializing OpenAI embedder...")
    embedder = OpenAIRowEmbedder(
        model_name=args.model,
        dimensions=args.dimensions,
        max_chars_per_cell=args.max_chars_per_cell,
    )

    # 3. Prepare label encoders
    label_encoders = {}
    if has_labels:
        print(f"\n3. Preparing label encoders...")
        full_view = dataset.get_full()
        for col in label_cols:
            y_col = full_view.y[col] if isinstance(full_view.y, pd.DataFrame) else full_view.y
            if is_regression_label(y_col, dataset.label_task_types, col):
                label_encoders[col] = None
            else:
                le = LabelEncoder()
                le.fit(y_col)
                label_encoders[col] = le
    else:
        print(f"\n3. No label columns")

    # 4. Generate embeddings per split
    print(f"\n4. Generating embeddings per split...")

    emb_dict = {}
    lbl_dict = {}
    idx_dict = {}

    for split_name in dataset.split_names:
        view = dataset.get_split(split_name)
        print(f"\n   Processing split '{split_name}' ({len(view)} samples)...")

        embeddings = embedder.encode_rows(view.X, batch_size=args.row_batch_size)
        emb_dict[split_name] = embeddings
        print(f"   Embeddings shape: {embeddings.shape}")

        if has_labels and view.y is not None:
            per_col = {}
            for col in label_cols:
                y_col = view.y[col] if isinstance(view.y, pd.DataFrame) else view.y
                le = label_encoders.get(col)
                per_col[col] = encode_label_column(y_col, le, split_name, col, logger)
            lbl_dict[split_name] = per_col

        if view.row_indices is not None:
            idx_dict[split_name] = view.row_indices

    # 5. Save
    print(f"\n5. Saving embeddings...")

    first_split = next(iter(emb_dict.values()))
    feature_columns = list(dataset.feature_columns)

    generation_config = {
        'data_source': str(args.data_dir),
        'model': embedder.model_name,
        'dimensions': args.dimensions,
        'max_chars_per_cell': args.max_chars_per_cell,
        'row_batch_size': args.row_batch_size,
        'label_policy': args.label_policy,
    }

    dataset_info = {
        'source_path': dataset.source_path,
        'layout': dataset.layout,
    }
    if dataset.fingerprint:
        dataset_info['fingerprint_sha256'] = dataset.fingerprint

    metadata = RowEmbeddingMetadataV2(
        model_name='openai',
        embedding_dim=first_split.shape[1],
        label_columns=label_cols,
        label_task_types={c: dataset.label_task_types.get(c, '') for c in label_cols},
        feature_columns=feature_columns,
        generation_config=generation_config,
        dataset=dataset_info,
    )

    save_split_embeddings(
        embeddings=emb_dict,
        metadata=metadata,
        output_dir=str(output_dir),
        labels=lbl_dict if lbl_dict else None,
        row_indices=idx_dict if idx_dict else None,
    )

    print(f"\n{'='*80}")
    print(f"Done. Dim={first_split.shape[1]}")
    for name, emb in emb_dict.items():
        print(f"  {name}: {emb.shape}")
    print(f"Output: {output_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
