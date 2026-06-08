#!/usr/bin/env python
"""
OpenAI Row Embedding Generation (Directory Mode)

Processes a directory of CSV files and produces an aggregate pickle
containing row embeddings for each table. Each data row is serialized
as "col1: val1 | col2: val2 | ..." and encoded through OpenAI API.

Same serialization as GTE row embeddings, with dimensions=768 by default.

Output format: List[dict] pickle at --output_path, one entry per table.
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()

import tiktoken

OPENAI_MAX_TOKENS = 8191
_enc = None

def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding('cl100k_base')
    return _enc

def truncate_text(text: str, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

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


def serialize_row(columns, row_values, max_chars_per_cell=100):
    """Serialize a single data row as 'col: val | col: val | ...'."""
    pairs = []
    for col, val in zip(columns, row_values):
        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val)[:max_chars_per_cell]
        pairs.append(f"{col}: {val_str}")
    return " | ".join(pairs)


class OpenAIRowEmbedder:
    """Encodes table rows as text and produces embeddings via OpenAI API."""

    def __init__(self, model_name="text-embedding-3-small",
                 dimensions=None, max_chars_per_cell=100):
        from models.openai.client import create_client, resolve_model_name, get_model_info

        self.client, provider = create_client()
        self.model_name = resolve_model_name(model_name, provider)
        _, native_dim, self._supports_dimensions = get_model_info(model_name)
        self.max_chars_per_cell = max_chars_per_cell

        if dimensions is not None and self._supports_dimensions:
            self.dimensions = dimensions
        else:
            self.dimensions = native_dim if not self._supports_dimensions else (dimensions or native_dim)

        print(f"Row embedder ({provider}): model={self.model_name}, dimensions={self.dimensions}")

    def _embed_batch(self, batch):
        """Embed a batch of texts, splitting on token limit errors."""
        import openai
        kwargs = dict(input=batch, model=self.model_name)
        if self._supports_dimensions:
            kwargs['dimensions'] = self.dimensions
        try:
            response = self.client.embeddings.create(**kwargs)
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
        """Encode all rows of a DataFrame into embeddings.

        Args:
            df: DataFrame with feature columns only (no label columns).
            batch_size: Number of rows per API call.

        Returns:
            np.ndarray of shape (n_rows, embedding_dim) with float32 dtype.
        """
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


def embed_table(embedder, csv_path, label_columns=None, row_batch_size=256,
                max_rows=None):
    """Embed a single CSV file using OpenAI row-level encoding.

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
        model_name="openai",
    )
    result["generation_config"] = {
        "model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "max_chars_per_cell": embedder.max_chars_per_cell,
        "row_batch_size": row_batch_size,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row embeddings for a directory of CSV files using OpenAI"
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
        "--model", type=str, default="text-embedding-3-small",
        help="OpenAI model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--dimensions", type=int, default=768,
        help="Embedding dimensions (default: 768, matching GTE-base)",
    )
    parser.add_argument(
        "--max_chars_per_cell", type=int, default=100,
        help="Max characters per cell value before truncation (default: 100)",
    )
    parser.add_argument(
        "--row_batch_size", type=int, default=256,
        help="Number of rows per API call (default: 256)",
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
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel table workers (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("OpenAI Row Embedding Generation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Max chars/cell: {args.max_chars_per_cell}")

    embedder = OpenAIRowEmbedder(
        model_name=args.model,
        dimensions=args.dimensions,
        max_chars_per_cell=args.max_chars_per_cell,
    )

    csv_files = discover_csv_files(args.input_dir, table_list_path=args.table_list)
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")

    if not csv_files:
        sys.exit(0)

    results = load_existing_results(args.output_path)
    completed = get_completed_table_ids(results)
    register_save_on_signal(results, args.output_path)
    if completed:
        print(f"Resuming: {len(completed)} tables already processed")

    # Filter to pending tables
    pending = [cp for cp in csv_files if cp.stem not in completed]
    print(f"Tables to process: {len(pending)}")

    if not pending:
        print("All tables already processed")
        return

    if args.workers <= 1:
        # Sequential
        newly_processed = 0
        for i, csv_path in enumerate(pending):
            print(f"\n[{i + 1}/{len(pending)}] Processing {csv_path.name}...")
            result = embed_table(
                embedder, csv_path,
                label_columns=args.label_columns,
                row_batch_size=args.row_batch_size,
                max_rows=args.max_rows,
            )
            if result is not None:
                results.append(result)
                newly_processed += 1
            if newly_processed > 0 and newly_processed % args.checkpoint_interval == 0:
                save_aggregate_pickle(results, args.output_path)
                print(f"  Checkpoint saved ({len(results)} tables total)")
    else:
        # Parallel: multiple tables concurrently, each making its own API calls
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        def _process_table(csv_path):
            return embed_table(
                embedder, csv_path,
                label_columns=args.label_columns,
                row_batch_size=args.row_batch_size,
                max_rows=args.max_rows,
            )

        newly_processed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_table, cp): cp for cp in pending}
            with tqdm(total=len(pending), desc="Embedding tables") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        newly_processed += 1
                    pbar.update(1)
                    if newly_processed > 0 and newly_processed % args.checkpoint_interval == 0:
                        save_aggregate_pickle(results, args.output_path)

    if newly_processed > 0:
        save_aggregate_pickle(results, args.output_path)

    print(f"\n{'=' * 80}")
    print(f"Done. {len(results)} tables in {args.output_path}")
    print(f"  Newly processed: {newly_processed}")
    print(f"  Previously completed: {len(completed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
