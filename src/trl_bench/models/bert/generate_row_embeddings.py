"""
BERT Row Embedding Generation (Directory Mode)

Processes a directory of CSV files and produces an aggregate pickle
containing row embeddings for each table. Each data row is serialized
as "col1: val1 | col2: val2 | ..." and encoded through BERT, extracting
the [CLS] token embedding.

Output format: List[dict] pickle at --output_path, one entry per table.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer


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


def embed_table(embedder, csv_path, label_columns=None, row_batch_size=64,
                max_rows=None):
    """Embed a single CSV file using BERT row-level encoding.

    Returns a table result dict, or None if the table cannot be processed.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: cannot read CSV: {e}")
        return None

    if len(df) < 1:
        print(f"  SKIP {csv_path.name}: no data rows")
        return None

    # Drop label columns
    label_set = set(label_columns) if label_columns else set()
    feature_cols = [c for c in df.columns if c not in label_set]
    df_features = df[feature_cols]

    if max_rows is not None and len(df_features) > max_rows:
        df_features = df_features.iloc[:max_rows]

    column_names = list(df_features.columns)

    try:
        embeddings = embedder.encode_rows(df_features, batch_size=row_batch_size)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: embedding failed: {e}")
        return None

    result = build_table_result(
        table_path=str(csv_path),
        row_embeddings=embeddings,
        column_names=column_names,
        model_name="bert",
    )
    result["generation_config"] = {
        "model": embedder.tokenizer.name_or_path,
        "max_length": embedder.max_length,
        "max_chars_per_cell": embedder.max_chars_per_cell,
        "row_batch_size": row_batch_size,
        "device": str(embedder.device),
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using BERT"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output path for aggregate pickle",
    )
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased",
        help="HuggingFace BERT model name or path (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, cuda:0, etc. (default: auto)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum token length for BERT tokenizer (default: 512)",
    )
    parser.add_argument(
        "--max_chars_per_cell", type=int, default=50,
        help="Max characters per cell value before truncation (default: 50)",
    )
    parser.add_argument(
        "--row_batch_size", type=int, default=64,
        help="Number of rows per forward pass (default: 64)",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Maximum rows to embed per table (default: all)",
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
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("BERT Row Embedding Generation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Max length: {args.max_length}")
    print(f"Max chars/cell: {args.max_chars_per_cell}")
    print(f"Device: {args.device}")

    # Initialize embedder
    print(f"\nLoading BERT model: {args.model}...")
    embedder = BERTRowEmbedder(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        max_chars_per_cell=args.max_chars_per_cell,
    )
    print(f"Model loaded on {embedder.device}")

    # Discover tables
    csv_files = discover_csv_files(args.input_dir, table_list_path=args.table_list)
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
            max_rows=args.max_rows,
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
