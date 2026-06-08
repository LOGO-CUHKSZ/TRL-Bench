#!/usr/bin/env python
"""
Generate column embeddings from CSV files using pretrained GTE.

GTE (General Text Embeddings) is a contrastively-trained encoder with the same
BERT-like architecture (768-dim, CLS pooling) but much better embedding quality
out-of-the-box. It uses a 512-token context like BERT.

This script uses two complementary encoding strategies:

1. **Table-level CLS**: Linearize the whole table (cells joined by '|',
   rows joined by '.') into a single string, run one forward pass, and
   take the CLS embedding.

2. **Per-column embeddings**: Encode each column as "header: val1, val2, ..."
   and take the CLS embedding per column. These are then aggregated via
   utils/aggregation to produce column_mean table embeddings.

Output format: unified v2.0 pickle — list of dicts, each with:
    table_id, table_embedding (dict), column_embeddings (dict), column_names,
    table_name, model_name, embedding_dim

RESUME SUPPORT
==============
Checkpoint file (.checkpoint.pkl) saved alongside output every N tables.
On restart, automatically detects and loads checkpoint, skipping processed tables.

Usage:
    # Single CSV
    python generate_column_embeddings.py --input table.csv --output emb.pkl

    # Directory of CSVs
    python generate_column_embeddings.py --input /path/to/csvs/ --output emb.pkl

    # With options
    python generate_column_embeddings.py --input /path/to/csvs/ --output emb.pkl \
        --model thenlper/gte-base --device cuda --max_rows 50 --checkpoint_interval 200
"""

import os
import sys
import pickle
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trl_bench.utils.aggregation import aggregate_embeddings


# =============================================================================
# Checkpoint/Resume Support
# =============================================================================

def load_checkpoint_data(output_path: str):
    """Load existing checkpoint or output file for resume support."""
    checkpoint_path = Path(output_path).with_suffix('.checkpoint.pkl')
    existing_results = []
    processed_tables = set()

    for path, label in [(checkpoint_path, "checkpoint"), (Path(output_path), "output")]:
        if path.exists() and not processed_tables:
            print(f"\nFound {label} file: {path}")
            try:
                with open(path, 'rb') as f:
                    existing_results = pickle.load(f)
                if isinstance(existing_results, dict):
                    existing_results = [
                        {**v, 'table_name': v.get('table_name', k)}
                        for k, v in existing_results.items()
                    ]
                processed_tables = {e['table_name'] for e in existing_results}
                print(f"  Loaded {len(existing_results)} already-processed tables from {label}")
            except Exception as e:
                print(f"  Warning: Failed to load {label}: {e}")
                existing_results = []
                processed_tables = set()

    return existing_results, processed_tables, checkpoint_path


def save_checkpoint(results: list, checkpoint_path: Path):
    """Save current progress to checkpoint file."""
    try:
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        with open(tmp_path, 'wb') as f:
            pickle.dump(results, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


# =============================================================================
# Table serialization
# =============================================================================

def linearize_table(df: pd.DataFrame) -> str:
    """
    Linearize a DataFrame into a flat string for GTE encoding.

    Format: "col1 | col2 | col3 . val1 | val2 | val3 . ..."
    Header row first, then data rows, separated by ' . '.
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
# GTEEmbedder
# =============================================================================

class GTEEmbedder:
    """
    GTE embedder for table and column embeddings from CSV files.

    Uses thenlper/gte-base by default (768-dim). Encodes tables as
    linearized text and columns individually.
    """

    def __init__(
        self,
        model_name: str = 'thenlper/gte-base',
        device: str = None,
        max_length: int = 512,
    ):
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.max_length = max_length
        self.model_name = model_name

        print(f"Loading GTE model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded — device: {device}, dim: {self.embedding_dim}")

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text and return CLS embedding as numpy array."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 0].cpu().numpy().astype(np.float32)

    def encode_csv(
        self,
        csv_path: str,
        max_rows: int = 100,
        delimiter: str = None,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a CSV file.

        Returns dict in unified v2.0 format with table_id, table_embedding,
        column_embeddings, column_names, etc.
        """
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV
        if delimiter:
            df = pd.read_csv(csv_path, nrows=max_rows, delimiter=delimiter, dtype=str)
        else:
            try:
                df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)
            except Exception:
                try:
                    df = pd.read_csv(csv_path, nrows=max_rows, engine='python', dtype=str)
                except Exception:
                    for delim in [',', '#', '\t', ';']:
                        try:
                            df = pd.read_csv(csv_path, nrows=max_rows, delimiter=delim, dtype=str)
                            if len(df.columns) > 1:
                                break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"Failed to parse CSV with any delimiter: {csv_path}")

        df = df.head(max_rows).copy()
        df = df.reset_index(drop=True)
        df = df.fillna('')
        df = df.astype(str)
        df.columns = [str(c) for c in df.columns]

        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        column_names = list(df.columns)

        # 1) Table-level CLS + token_mean from linearized table
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
        cls_embedding = hidden_states[0].cpu().numpy().astype(np.float32)

        attention_mask = inputs['attention_mask'][0]  # (seq_len,)
        mask = attention_mask.unsqueeze(-1).float()  # (seq_len, 1)
        token_mean = ((hidden_states * mask).sum(dim=0) / mask.sum()).cpu().numpy().astype(np.float32)

        # 2) Per-column embeddings
        col_embeddings = {}
        for i, col in enumerate(column_names):
            col_text = serialize_column(col, df[col])
            col_embeddings[i] = self._encode_text(col_text)

        table_embedding = {
            'cls_embedding': cls_embedding,
            'table_embedding': None,
            'column_mean': aggregate_embeddings(col_embeddings, 'mean'),
            'token_mean': token_mean,
        }

        return {
            'version': '2.0',
            'format': 'unified_table_embedding',
            'table_id': table_name,
            'table': csv_path,
            'table_embedding': table_embedding,
            'column_embeddings': col_embeddings,
            'column_names': column_names,
            'table_name': table_name,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }

    def encode_directory(
        self,
        csv_dir: str,
        max_rows: int = 100,
        show_progress: bool = True,
        existing_results: List[Dict] = None,
        processed_tables: set = None,
        checkpoint_path: Path = None,
        checkpoint_interval: int = 100,
        table_list: set = None,
    ) -> List[Dict]:
        """Generate embeddings for all CSV files in a directory."""
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
        if table_list is not None:
            csv_files = [f for f in csv_files if f in table_list]
        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_dir}")

        results = list(existing_results) if existing_results else []
        processed_tables = processed_tables or set()

        if processed_tables:
            original_count = len(csv_files)
            csv_files = [f for f in csv_files
                         if os.path.splitext(f)[0] not in processed_tables]
            skipped = original_count - len(csv_files)
            if skipped > 0:
                print(f"Skipping {skipped} already-processed tables")

        if not csv_files:
            print("All tables already processed")
            return results

        tables_since_checkpoint = 0
        iterator = tqdm(csv_files, desc="Encoding tables") if show_progress else csv_files

        for csv_file in iterator:
            csv_path = os.path.join(csv_dir, csv_file)
            result = self.encode_csv(csv_path, max_rows=max_rows)
            results.append(result)
            tables_since_checkpoint += 1

            if checkpoint_interval > 0 and checkpoint_path and tables_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, checkpoint_path)
                tables_since_checkpoint = 0

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings from CSV file(s) using GTE'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to CSV file or directory of CSV files')
    parser.add_argument('--model', type=str, default='thenlper/gte-base',
                        help='HuggingFace model name (default: thenlper/gte-base)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file (default: auto-generated)')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows to load from CSV (default: 100)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N tables (default: 100)')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()
    is_directory = os.path.isdir(args.input)

    if args.output is None:
        if is_directory:
            args.output = 'gte_embeddings.pkl'
        else:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}_gte_embeddings.pkl"

    embedder = GTEEmbedder(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
    )

    table_list = None
    if args.table_list:
        from trl_bench.utils.table_list import load_table_list
        table_list = load_table_list(args.table_list)

    if is_directory:
        print(f"\nProcessing directory: {args.input}")
        existing_results, processed_tables, checkpoint_path = load_checkpoint_data(args.output)

        start_time = time.time()
        results = embedder.encode_directory(
            args.input,
            max_rows=args.max_rows,
            existing_results=existing_results,
            processed_tables=processed_tables,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            table_list=table_list,
        )
        elapsed = time.time() - start_time

        with open(args.output, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print("Checkpoint file removed (processing complete).")
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint: {e}")

        new_tables = len(results) - len(existing_results)
        print(f"\n{'=' * 60}")
        print("BATCH EMBEDDING EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Model: {args.model}")
        print(f"Tables processed: {len(results)} total ({new_tables} new)")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        print(f"Output saved to: {args.output}")
        print(f"Inference time: {elapsed:.2f} seconds")
        print(f"{'=' * 60}")
    else:
        print(f"\nProcessing file: {args.input}")
        result = embedder.encode_csv(args.input, max_rows=args.max_rows)

        with open(args.output, 'wb') as f:
            pickle.dump(result, f)

        print(f"\n{'=' * 60}")
        print("EMBEDDING EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Table: {result['table_name']}")
        print(f"Model: {result['model_name']}")
        print(f"Columns: {len(result['column_embeddings'])}")
        print(f"Column names: {result['column_names']}")
        print(f"Embedding dimension: {result['embedding_dim']}")
        print(f"Output saved to: {args.output}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
