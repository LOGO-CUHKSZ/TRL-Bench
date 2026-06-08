#!/usr/bin/env python
"""
Generate column embeddings from CSV files using OpenAI text-embedding-3-small.

Uses the same serialization strategy as GTE:

1. **Table-level embedding**: Linearize the whole table (cells joined by '|',
   rows joined by '.') into a single string, embed via OpenAI API.

2. **Per-column embeddings**: Encode each column as "header: val1, val2, ..."
   and embed via OpenAI API. Aggregated via utils/aggregation to produce
   column_mean table embeddings.

By default, dimensions=768 to match GTE-base. No explicit context truncation
is applied; the API truncates at 8191 tokens internally.

Output format: unified v2.0 pickle — list of dicts, each with:
    table_id, table_embedding (dict), column_embeddings (dict), column_names,
    table_name, model_name, embedding_dim

PARALLELIZATION
===============
For large directories, --workers N enables cross-table batching: texts from
many tables are packed into large API calls (up to 2048 texts each) and sent
concurrently from N threads. This cuts 170K single-table API calls down to
~500 large batched calls running in parallel.

RESUME SUPPORT
==============
Checkpoint file (.checkpoint.pkl) saved alongside output every N tables.
On restart, automatically detects and loads checkpoint, skipping processed tables.

Usage:
    # Single CSV
    python generate_column_embeddings.py --input table.csv --output emb.pkl

    # Directory of CSVs (parallel)
    python generate_column_embeddings.py --input /path/to/csvs/ --output emb.pkl \
        --model text-embedding-3-small --dimensions 768 --max_rows 100 --workers 32
"""

import os
import sys
import pickle
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()

import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm

OPENAI_MAX_TOKENS = 8191
_enc = None

def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding('cl100k_base')
    return _enc

def truncate_text(text: str, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
    """Truncate text to max_tokens using the cl100k_base tokenizer."""
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

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
# Table serialization (same as GTE)
# =============================================================================

def linearize_table(df: pd.DataFrame) -> str:
    """
    Linearize a DataFrame into a flat string.

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
# CSV reading helper
# =============================================================================

def read_csv_robust(csv_path: str, max_rows: int, delimiter: str = None) -> pd.DataFrame:
    """Read a CSV file with fallback delimiters. Returns cleaned DataFrame."""
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
    return df


# =============================================================================
# OpenAIEmbedder
# =============================================================================

class OpenAIEmbedder:
    """
    OpenAI embedder for table and column embeddings from CSV files.

    Uses text-embedding-3-small by default with dimensions=768 to match
    GTE-base. Encodes tables as linearized text and columns individually.
    """

    def __init__(
        self,
        model_name: str = 'text-embedding-3-small',
        dimensions: int = None,
    ):
        from models.openai.client import create_client, resolve_model_name, get_model_info, supports_dimensions

        self.client, provider = create_client()
        self.model_name = resolve_model_name(model_name, provider)
        _, native_dim, self._supports_dimensions = get_model_info(model_name)

        if dimensions is not None and self._supports_dimensions:
            self.dimensions = dimensions
        elif dimensions is not None and not self._supports_dimensions:
            print(f"  Warning: {model_name} does not support dimensions param, using native {native_dim}")
            self.dimensions = native_dim
        else:
            self.dimensions = native_dim

        self.embedding_dim = self.dimensions

        print(f"Embedder ({provider}): model={self.model_name}, dimensions={self.dimensions}")

    def _embed_texts(self, texts: List[str], _retries: int = 5) -> List[np.ndarray]:
        """Embed a list of texts in one API call.

        Each text is truncated to OPENAI_MAX_TOKENS. If the batch exceeds
        the per-request token limit (300K), it is automatically split in
        half and retried recursively. Rate limit errors (429) are retried
        with exponential backoff.
        """
        import openai

        texts = [truncate_text(t) for t in texts]
        # Replace empty strings with a space (API rejects empty input)
        texts = [t if t.strip() else " " for t in texts]

        if len(texts) == 0:
            return []

        try:
            kwargs = dict(input=texts, model=self.model_name)
            if self._supports_dimensions:
                kwargs['dimensions'] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]
        except (openai.BadRequestError, ValueError) as e:
            # ValueError: "No embedding data received" — split and retry
            if len(texts) > 1:
                mid = len(texts) // 2
                left = self._embed_texts(texts[:mid])
                right = self._embed_texts(texts[mid:])
                return left + right
            raise
        except openai.RateLimitError:
            if _retries <= 0:
                raise
            wait = 2 ** (5 - _retries) + 1  # 2, 3, 5, 9, 17 seconds
            time.sleep(wait)
            return self._embed_texts(texts, _retries=_retries - 1)

    def _assemble_table_result(
        self,
        csv_path: str,
        column_names: List[str],
        embeddings: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Assemble a unified v2.0 result dict from embeddings.

        Args:
            csv_path: Path to the source CSV.
            column_names: List of column names.
            embeddings: List of embeddings — index 0 is the table-level CLS,
                        indices 1..N correspond to columns.
        """
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        table_cls = embeddings[0]
        col_embeddings = {i: embeddings[i + 1] for i in range(len(column_names))}

        table_embedding = {
            'cls_embedding': table_cls,
            'table_embedding': None,
            'column_mean': aggregate_embeddings(col_embeddings, 'mean'),
            'token_mean': None,
        }

        return {
            'version': '2.0',
            'format': 'unified_table_embedding',
            'table_id': table_name,
            'table': os.path.abspath(csv_path),
            'table_embedding': table_embedding,
            'column_embeddings': col_embeddings,
            'column_names': column_names,
            'table_name': table_name,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }

    def encode_csv(
        self,
        csv_path: str,
        max_rows: int = 100,
        delimiter: str = None,
    ) -> Dict[str, Any]:
        """Generate embeddings for a single CSV file."""
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = read_csv_robust(csv_path, max_rows, delimiter)
        column_names = list(df.columns)

        texts = [linearize_table(df)]
        for col in column_names:
            texts.append(serialize_column(col, df[col]))

        embeddings = self._embed_texts(texts)
        return self._assemble_table_result(csv_path, column_names, embeddings)

    def encode_directory(
        self,
        csv_dir: str,
        max_rows: int = 100,
        workers: int = 1,
        api_batch_size: int = 2048,
        show_progress: bool = True,
        existing_results: List[Dict] = None,
        processed_tables: set = None,
        checkpoint_path: Path = None,
        checkpoint_interval: int = 500,
        table_list: set = None,
    ) -> List[Dict]:
        """Generate embeddings for all CSV files in a directory.

        When workers > 1, texts from multiple tables are packed into large API
        calls (up to api_batch_size texts each) and sent from a thread pool,
        dramatically reducing HTTP overhead.
        """
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

        if workers <= 1:
            return self._encode_directory_sequential(
                csv_dir, csv_files, max_rows, show_progress,
                results, checkpoint_path, checkpoint_interval,
            )
        else:
            return self._encode_directory_parallel(
                csv_dir, csv_files, max_rows, workers, api_batch_size,
                results, checkpoint_path, checkpoint_interval,
            )

    # -----------------------------------------------------------------
    # Sequential path (workers=1) — simple, one table per API call
    # -----------------------------------------------------------------

    def _encode_directory_sequential(
        self, csv_dir, csv_files, max_rows, show_progress,
        results, checkpoint_path, checkpoint_interval,
    ):
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

    # -----------------------------------------------------------------
    # Parallel path — cross-table batching + thread pool
    # -----------------------------------------------------------------

    def _encode_directory_parallel(
        self, csv_dir, csv_files, max_rows, workers, api_batch_size,
        results, checkpoint_path, checkpoint_interval,
    ):
        # Step 1: Read all CSVs and serialize texts (CPU-only)
        print(f"Step 1/3: Serializing {len(csv_files)} tables...")
        table_specs = []   # (csv_path, column_names, num_texts)
        all_texts = []     # flat list of all texts across tables

        for csv_file in tqdm(csv_files, desc="Reading CSVs"):
            csv_path = os.path.join(csv_dir, csv_file)
            try:
                df = read_csv_robust(csv_path, max_rows)
            except Exception as e:
                print(f"  SKIP {csv_file}: {e}")
                continue

            column_names = list(df.columns)
            start_idx = len(all_texts)

            all_texts.append(linearize_table(df))
            for col in column_names:
                all_texts.append(serialize_column(col, df[col]))

            num_texts = len(column_names) + 1  # table + columns
            table_specs.append((csv_path, column_names, start_idx, num_texts))

        total_texts = len(all_texts)
        print(f"  {len(table_specs)} tables -> {total_texts:,} texts")

        # Step 2: Embed all texts in parallel batched API calls
        print(f"Step 2/3: Embedding with {workers} workers, "
              f"batch_size={api_batch_size}...")
        all_embeddings = [None] * total_texts

        # Build batch ranges
        batch_ranges = []
        for i in range(0, total_texts, api_batch_size):
            batch_ranges.append((i, min(i + api_batch_size, total_texts)))

        def _embed_batch(start_end):
            start, end = start_end
            embs = self._embed_texts(all_texts[start:end])
            return start, embs

        completed_texts = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_embed_batch, br): br for br in batch_ranges}
            with tqdm(total=total_texts, desc="Embedding", unit="text") as pbar:
                for future in as_completed(futures):
                    start, embs = future.result()
                    for i, emb in enumerate(embs):
                        all_embeddings[start + i] = emb
                    pbar.update(len(embs))
                    completed_texts += len(embs)

        # Step 3: Assemble per-table results
        print(f"Step 3/3: Assembling {len(table_specs)} table results...")
        tables_since_checkpoint = 0
        for csv_path, column_names, start_idx, num_texts in table_specs:
            embeddings = all_embeddings[start_idx:start_idx + num_texts]
            result = self._assemble_table_result(csv_path, column_names, embeddings)
            results.append(result)
            tables_since_checkpoint += 1

            if checkpoint_interval > 0 and checkpoint_path and tables_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, checkpoint_path)
                tables_since_checkpoint = 0

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings from CSV file(s) using OpenAI'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to CSV file or directory of CSV files')
    parser.add_argument('--model', type=str, default='text-embedding-3-small',
                        help='OpenAI model name (default: text-embedding-3-small)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file (default: auto-generated)')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows to load from CSV (default: 100)')
    parser.add_argument('--dimensions', type=int, default=None,
                        help='Embedding dimensions (default: 768 for v3, 1536 for ada-002)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel API threads (default: 8)')
    parser.add_argument('--api_batch_size', type=int, default=1024,
                        help='Max texts per API call (default: 1024, auto-splits if too large)')
    parser.add_argument('--checkpoint_interval', type=int, default=500,
                        help='Save checkpoint every N tables (default: 500)')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()
    is_directory = os.path.isdir(args.input)

    if args.output is None:
        if is_directory:
            args.output = 'openai_embeddings.pkl'
        else:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}_openai_embeddings.pkl"

    embedder = OpenAIEmbedder(
        model_name=args.model,
        dimensions=args.dimensions,
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
            workers=args.workers,
            api_batch_size=args.api_batch_size,
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
        print(f"Dimensions: {args.dimensions}")
        print(f"Workers: {args.workers}")
        print(f"Tables processed: {len(results)} total ({new_tables} new)")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        print(f"Output saved to: {args.output}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
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
        print(f"Dimensions: {args.dimensions}")
        print(f"Columns: {len(result['column_embeddings'])}")
        print(f"Column names: {result['column_names']}")
        print(f"Embedding dimension: {result['embedding_dim']}")
        print(f"Output saved to: {args.output}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
