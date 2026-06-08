#!/usr/bin/env python
"""
Generate table embeddings from CSV files using any HuggingFace text encoder.

Shared Stage-1 table-DIRECT script for models that don't have a per-column
extraction step: each table is linearized to a flat text string and encoded
with any HuggingFace text encoder (MPNet, Sentence-T5, DeBERTa, ...). The
output pickle is table-level (one entry per table), so Stage-2 aggregation
is skipped entirely for models that route through this script.

Supports two pooling strategies:
  - cls:  Use position-0 ([CLS]) token -- for BERT-like models (DeBERTa, GTE)
  - mean: Mean-pool all non-padding tokens -- for models without a trained
          [CLS] token (Sentence-T5, MPNet)

Produces all table-level variants in one pass:
  - cls_embedding:  CLS/mean-pooled from linearized table text
  - column_mean:    Mean of per-column embeddings (used by hybrid mode +
                    column-axis settings)
  - token_mean:     Mean-pooled from linearized table text

Output format: List[dict] pickle, one entry per table:
    {table_id, table_embedding: {cls_embedding, column_mean, token_mean, ...},
     model_name, embedding_dim}

Usage::

    # Sentence-T5 (mean pooling, T5 encoder)
    python -m trl_bench.utils.generate_table_embeddings_text_encoder \\
        --input_dir datasets/nq_tables/csv/tables \\
        --output_path embeddings/table/sentence_t5/nq_tables.pkl \\
        --model sentence-transformers/sentence-t5-base --pooling mean

    # MPNet (mean pooling)
    python -m trl_bench.utils.generate_table_embeddings_text_encoder \\
        --input_dir datasets/nq_tables/csv/tables \\
        --output_path embeddings/table/mpnet/nq_tables.pkl \\
        --model sentence-transformers/all-mpnet-base-v2 --pooling mean
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)
from trl_bench.utils.table_list import load_table_list, filter_csv_files


# =============================================================================
# Table serialization (same conventions as GTE/BERT column embedding scripts)
# =============================================================================

def linearize_table(df: pd.DataFrame) -> str:
    """Linearize a DataFrame into a flat string for encoding.

    Format: "col1 | col2 | col3 . val1 | val2 | val3 . ..."
    """
    rows = [' | '.join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        rows.append(' | '.join(str(v) for v in row))
    return ' . '.join(rows)


def serialize_column(col_name: str, values: pd.Series) -> str:
    """Serialize a single column as 'header: val1, val2, ...'."""
    vals = ', '.join(str(v) for v in values)
    return f"{col_name}: {vals}"


# =============================================================================
# TextEncoderEmbedder
# =============================================================================

class TextEncoderEmbedder:
    """Generic text encoder embedder for table-level embeddings from CSV files.

    Works with any HuggingFace model that supports AutoModel and produces
    last_hidden_state. Supports CLS and mean pooling strategies.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = 'cls',
        device: str = None,
        max_length: int = 512,
    ):
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.max_length = max_length
        self.model_name = model_name
        self.pooling = pooling

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # T5 loads as encoder-decoder; we only need the encoder
        if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
            self.model = self.model.encoder
        self.model = self.model.to(device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded -- device: {device}, dim: {self.embedding_dim}, pooling: {pooling}")

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text and return a single embedding vector."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state[0]  # (seq_len, dim)

        if self.pooling == 'cls':
            return hidden_states[0].cpu().numpy().astype(np.float32)
        else:  # mean
            mask = inputs['attention_mask'][0].unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-9)
            return pooled.cpu().numpy().astype(np.float32)

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Mean-pool hidden states over non-padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-9)
        return pooled.cpu().numpy().astype(np.float32)

    def encode_csv(
        self,
        csv_path: str,
        max_rows: int = 100,
        delimiter: str = None,
    ) -> dict:
        """Generate table-level embeddings for a CSV file."""
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV with auto-delimiter detection
        if delimiter:
            df = pd.read_csv(csv_path, nrows=max_rows, delimiter=delimiter, dtype=str)
        else:
            try:
                df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)
            except Exception:
                df = None
                for delim in [',', '#', '\t', ';']:
                    try:
                        df = pd.read_csv(csv_path, nrows=max_rows, delimiter=delim, dtype=str)
                        if len(df.columns) > 1:
                            break
                    except Exception:
                        continue
                if df is None:
                    df = pd.read_csv(csv_path, nrows=max_rows, dtype=str, engine='python')

        df = df.head(max_rows).copy()
        df = df.reset_index(drop=True)
        df = df.fillna('')
        df = df.astype(str)
        df.columns = [str(c) for c in df.columns]

        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        column_names = list(df.columns)

        # 1) Table-level embeddings from linearized table
        table_text = linearize_table(df)
        inputs = self.tokenizer(
            table_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state[0]  # (seq_len, dim)
        attention_mask = inputs['attention_mask'][0]

        if self.pooling == 'cls':
            cls_embedding = hidden_states[0].cpu().numpy().astype(np.float32)
        else:
            cls_embedding = self._mean_pool(hidden_states, attention_mask)

        token_mean = self._mean_pool(hidden_states, attention_mask)

        # 2) Per-column embeddings -> column_mean
        col_embeddings = []
        for col in column_names:
            col_text = serialize_column(col, df[col])
            col_embeddings.append(self._encode_text(col_text))
        column_mean = np.mean(col_embeddings, axis=0).astype(np.float32)

        return {
            'table_id': table_name,
            'table_embedding': {
                'cls_embedding': cls_embedding,
                'table_embedding': None,
                'column_mean': column_mean,
                'token_mean': token_mean,
            },
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate table embeddings for CSV files using a HuggingFace text encoder"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output path for aggregate pickle (e.g., embeddings/table/sentence_t5/nq_tables.pkl)",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--pooling", type=str, default="cls", choices=["cls", "mean"],
        help="Pooling strategy: cls (position 0) or mean (default: cls)",
    )
    parser.add_argument(
        "--max_rows", type=int, default=100,
        help="Maximum rows to load per CSV (default: 100)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=50,
        help="Save intermediate results every N tables (default: 50)",
    )
    parser.add_argument(
        "--table_list", type=str, default=None,
        help="Path to file listing CSV basenames to process (for sharded runs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Text Encoder Table Embedding Generation")
    print("=" * 80)

    embedder = TextEncoderEmbedder(
        model_name=args.model,
        pooling=args.pooling,
        max_length=args.max_length,
    )

    # Discover tables
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
    remaining = [f for f in csv_files if f.stem not in completed]
    for csv_path in tqdm(remaining, desc="Encoding tables"):
        result = embedder.encode_csv(str(csv_path), max_rows=args.max_rows)
        results.append(result)
        newly_processed += 1

        if newly_processed > 0 and newly_processed % args.checkpoint_interval == 0:
            save_aggregate_pickle(results, args.output_path)

    # Final save
    if newly_processed > 0:
        save_aggregate_pickle(results, args.output_path)

    print(f"\n{'=' * 80}")
    print(f"Done. {len(results)} tables in {args.output_path}")
    print(f"  Model: {args.model}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Newly processed: {newly_processed}")
    print(f"  Previously completed: {len(completed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
