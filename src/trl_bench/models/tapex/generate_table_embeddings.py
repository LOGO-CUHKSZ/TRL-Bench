"""
TAPEX Table Embedding Generation

Processes a directory of CSV files and produces a table embedding pickle
using TAPEX (microsoft/tapex-base), a BART-based encoder-decoder model
pre-trained on SQL execution.

TAPEX produces table-level embeddings only (no per-column embeddings).
The encoder's token-level hidden states are mean-pooled across all
non-padding tokens to produce the primary table representation.

Output format: List[dict] pickle at --output_path, one entry per table.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../" * 2))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl_bench.utils.row_embedding.directory import (
    discover_csv_files,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
)
from trl_bench.utils.table_list import load_table_list, filter_csv_files

import argparse
import warnings

import torch
import numpy as np
import pandas as pd

# Suppress FutureWarnings from tokenizer internals
warnings.filterwarnings('ignore', category=FutureWarning)


def _linearize_table(df: pd.DataFrame) -> str:
    """Linearize a DataFrame into the TAPEX format.

    Replicates the TapexTokenizer serialization (removed in transformers 5.x):
        col : col1 | col2 row 1 : val1 | val2 row 2 : val3 | val4
    """
    header = "col : " + " | ".join(str(c) for c in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = " | ".join(str(v) for v in row.values)
        rows.append(f"row {i + 1} : {cells}")
    return header + " " + " ".join(rows)


class TAPEXEmbedder:
    """
    TAPEX embedder for table-level embeddings using HuggingFace Transformers.

    Uses the TAPEX encoder (BART encoder) to produce token-level hidden states,
    then mean-pools all non-padding tokens for a table-level representation.

    TapexTokenizer was removed in transformers 5.x, so we linearize the table
    manually and tokenize with BartTokenizer (which TAPEX's tokenizer wraps).
    """

    def __init__(
        self,
        model_name: str = 'microsoft/tapex-base',
        device: str = None,
        max_length: int = 1024,
    ):
        from transformers import BartTokenizer, BartModel

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.max_length = max_length
        self.model_name = model_name

        print(f"Loading TAPEX model: {model_name}")

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        self.embedding_dim = self.model.config.d_model

        print(f"Model loaded successfully")
        print(f"Device: {device}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Max sequence length: {max_length}")

    def _prepare_table(self, df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
        """Prepare DataFrame for TAPEX encoding."""
        df = df.head(max_rows).copy()
        df = df.reset_index(drop=True)
        df = df.fillna('')
        df = df.astype(str)
        df.columns = [str(c) for c in df.columns]
        return df

    def encode_csv(
        self,
        csv_path: str,
        max_rows: int = 100,
        delimiter: str = None,
    ) -> dict:
        """
        Generate table-level embeddings for a CSV file.

        Args:
            csv_path: Path to CSV file
            max_rows: Maximum rows to load from CSV
            delimiter: CSV delimiter (default: auto-detect, falls back to ',')

        Returns:
            dict with table_id, table_embedding, model_name, embedding_dim
        """
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

        df = self._prepare_table(df, max_rows)
        table_name = os.path.splitext(os.path.basename(csv_path))[0]

        # Linearize table into TAPEX format and tokenize with BART tokenizer
        linear_text = _linearize_table(df)
        inputs = self.tokenizer(
            linear_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass through encoder only (critical for BART-family models)
        with torch.no_grad():
            encoder = self.model.get_encoder()
            outputs = encoder(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, seq_len, d_model)

        # BOS embedding (position 0) as cls_embedding
        cls_embedding = hidden_states[0, 0].cpu().numpy().astype(np.float32)

        # Mean-pool all non-padding encoder tokens (including BOS)
        attention_mask = inputs['attention_mask'][0]  # (seq_len,)
        mask = attention_mask.unsqueeze(-1).float()  # (seq_len, 1)
        summed = (hidden_states[0] * mask).sum(dim=0)  # (d_model,)
        count = mask.sum()
        table_embedding = (summed / count).cpu().numpy().astype(np.float32)

        return {
            'table_id': table_name,
            'table_embedding': {
                'cls_embedding': cls_embedding,
                'table_embedding': table_embedding,
                'column_mean': None,
                'token_mean': table_embedding,  # same as table_embedding (both are mean-pooled encoder tokens)
            },
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate table embeddings for a directory of CSV files using TAPEX"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for aggregate pickle (e.g., embeddings/table/tapex/dataset.pkl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/tapex-base",
        help="HuggingFace model name (default: microsoft/tapex-base)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=100,
        help="Maximum rows to load per CSV (default: 100)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
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


def main():
    args = parse_args()

    print("=" * 80)
    print("TAPEX Table Embedding Generation")
    print("=" * 80)

    # Initialize embedder
    print(f"Loading TAPEX model: {args.model}")
    embedder = TAPEXEmbedder(
        model_name=args.model,
        max_length=args.max_length,
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
        result = embedder.encode_csv(
            str(csv_path),
            max_rows=args.max_rows,
        )
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
