#!/usr/bin/env python
"""
Generate column embeddings from CSV files using TaBERT.

Supports single file and batch (directory) modes. Model is loaded once.
Supports three context modes:
  - 'context': Use provided question/context text (context-aware embeddings)
  - 'header': Use column names as pseudo-context (header-aware embeddings)
  - 'column': Empty context, pure column embeddings

RESUME SUPPORT
==============
This script supports resuming from interruptions:
- Checkpoint file (.checkpoint.pkl) saved alongside output every N tables
- On restart, automatically detects and loads checkpoint
- Skips already-processed tables
- Checkpoint removed after successful completion

Usage:
    # Single file - column-only mode (default)
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
        --output embeddings.pkl

    # With context (question)
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
        --context "What is the population of Tokyo?" \
        --context_mode context \
        --output embeddings.pkl

    # Header-aware mode
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
        --context_mode header \
        --output embeddings.pkl

    # Batch mode (directory of CSVs)
    python generate_column_embeddings.py \
        --input /path/to/csv_directory/ \
        --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
        --output all_embeddings.pkl

    # With checkpoint interval (for large datasets)
    python generate_column_embeddings.py \
        --input /path/to/csv_directory/ \
        --checkpoint checkpoints/tabert/tabert_large_k3/model.bin \
        --output all_embeddings.pkl \
        --checkpoint_interval 100

Note:
    - Supports both K=1 (VanillaTableBert) and K>1 (VerticalAttentionTableBert) models
    - K>1 models require table data rows for content snapshot sampling
"""

import os
import sys
import pickle
import argparse
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add TaBERT modules to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from table_bert.table_bert import TableBertModel
from table_bert.table import Table, Column
from trl_bench.utils.aggregation import aggregate_embeddings


# =============================================================================
# Checkpoint/Resume Support Functions
# =============================================================================

def load_checkpoint_data(output_path: str):
    """
    Load existing checkpoint or output file for resume support.

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
            # Support both old dict format and new list format
            if isinstance(existing_results, dict):
                # Convert dict to list, injecting table_name from key if missing
                existing_results = [
                    {**v, 'table_name': v.get('table_name', k)}
                    for k, v in existing_results.items()
                ]
            processed_tables = {e['table_name'] for e in existing_results}
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
            # Support both old dict format and new list format
            if isinstance(existing_results, dict):
                # Convert dict to list, injecting table_name from key if missing
                existing_results = [
                    {**v, 'table_name': v.get('table_name', k)}
                    for k, v in existing_results.items()
                ]
            processed_tables = {e['table_name'] for e in existing_results}
            print(f"  Loaded {len(existing_results)} already-processed tables from output")
        except Exception as e:
            print(f"  Warning: Failed to load output file: {e}")
            existing_results = []
            processed_tables = set()

    return existing_results, processed_tables, checkpoint_path


def save_checkpoint(results: dict, checkpoint_path: Path):
    """
    Save current progress to checkpoint file.

    Args:
        results: Dict mapping table_name -> embeddings
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


def _is_numeric_str(val: str) -> bool:
    """Return True if val can be parsed as a float (pure-Python check)."""
    try:
        float(val)
        return True
    except Exception:
        return False


def infer_column_type(series: pd.Series) -> str:
    """Infer column type from pandas series for TaBERT.

    Avoid pandas numeric inference to prevent segfaults on certain CSVs.
    """
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        if _is_numeric_str(s):
            return 'real'
    return 'text'


def select_content_snapshot_rows(
    table_data: List[Dict],
    context_tokens: List[str],
    num_rows: int = 3
) -> List[Dict]:
    """
    Select rows based on n-gram overlap with context (TaBERT content snapshot).

    This implements the question-biased row sampling from the TaBERT paper,
    selecting rows whose cell values have the highest overlap with the context.

    Args:
        table_data: List of row dicts {col_name: [cell_value], ...}
        context_tokens: Tokenized context/question
        num_rows: Number of rows to select

    Returns:
        Selected rows (list of dicts)
    """
    if not context_tokens or len(table_data) <= num_rows:
        return table_data[:num_rows]

    context_set = set(tok.lower() for tok in context_tokens)

    # Score each row by n-gram overlap with context
    row_scores = []
    for row_id, row in enumerate(table_data):
        score = 0
        for col_name, cell_tokens in row.items():
            if not cell_tokens:
                continue
            # Check for token overlap
            cell_set = set(tok.lower() for tok in cell_tokens if tok)
            overlap = len(context_set & cell_set)
            if overlap > 0:
                score = max(score, overlap)
        row_scores.append((row_id, score))

    # Sort by score (descending), then by row_id (ascending) for stability
    row_scores.sort(key=lambda x: (-x[1], x[0]))

    # Select top rows
    selected_ids = [row_id for row_id, _ in row_scores[:num_rows]]
    selected_ids.sort()  # Keep original order

    return [table_data[i] for i in selected_ids]


def csv_to_table(
    csv_path: str,
    table_id: str = None,
    max_rows: int = 100,
    sample_row_num: int = None,
    context_tokens: List[str] = None
) -> Table:
    """
    Convert a CSV file to TaBERT Table object.

    Args:
        csv_path: Path to CSV file
        table_id: Optional table identifier (defaults to filename)
        max_rows: Maximum rows to load from CSV (for initial candidate pool)
        sample_row_num: Number of rows to keep for encoding (default: all loaded rows).
                        If provided with context_tokens, uses content snapshot selection.
        context_tokens: Optional tokenized context for content snapshot row selection

    Returns:
        Table object ready for tokenization
    """
    # Read WITH dtype=str to prevent numeric inference segfaults on certain CSVs
    try:
        df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)
    except Exception:
        # Fall back to python engine for CSVs with embedded \r in quoted fields
        # that trigger C parser buffer overflow
        df = pd.read_csv(csv_path, nrows=max_rows, dtype=str, engine='python')
    if df.shape[0] == 0 or df.shape[1] == 0:
        msg = (
            f"Empty table after read_csv: {csv_path} "
            f"(rows={df.shape[0]}, cols={df.shape[1]})"
        )
        print(msg, file=sys.stderr)
        raise ValueError(msg)
    col_types = {col: infer_column_type(df[col]) for col in df.columns}

    if table_id is None:
        table_id = os.path.splitext(os.path.basename(csv_path))[0]

    # Create columns with sample values, using pre-inferred types
    columns = []
    for col_name in df.columns:
        col_type = col_types.get(col_name, 'text')  # Use pre-inferred type
        # Get first non-null value as sample
        non_null = df[col_name].dropna()
        sample_value = str(non_null.iloc[0]) if len(non_null) > 0 else ''

        columns.append(Column(
            name=str(col_name),
            type=col_type,
            sample_value=sample_value
        ))

    # Convert data to list of dicts with column name keys
    # Values should be lists of tokens (will be tokenized later)
    data = []
    for _, row in df.iterrows():
        row_dict = {str(col): [str(row[col])] for col in df.columns}
        data.append(row_dict)

    # Apply content snapshot selection if sample_row_num is specified
    if sample_row_num is not None and len(data) > sample_row_num:
        data = select_content_snapshot_rows(data, context_tokens, sample_row_num)

    table = Table(
        id=table_id,
        header=columns,
        data=data
    )

    return table


class TaBERTEmbedder:
    """
    TaBERT embedder that loads the model once for batch processing.

    Supports three context modes:
    - 'context': Use provided question/context (context-aware embeddings)
    - 'header': Use column names as pseudo-context (header-aware embeddings)
    - 'column': Empty context, pure column embeddings
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Load TaBERT model.

        Args:
            checkpoint_path: Path to TaBERT checkpoint (model.bin)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading TaBERT model: {checkpoint_path}")
        self.model = TableBertModel.from_pretrained(checkpoint_path)
        self.model = self.model.to(device)
        self.model.eval()

        # Determine model type and get sample_row_num
        self.model_type = type(self.model).__name__
        self.embedding_dim = self.model.output_size

        # Get sample_row_num from config (for K>1 models)
        self.sample_row_num = getattr(self.model.config, 'sample_row_num', 1)

        print(f"Model type: {self.model_type}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Sample row num: {self.sample_row_num}")

    def encode_csv(
        self,
        csv_path: str,
        context: str = None,
        context_mode: str = 'column',
        max_rows: int = 100,
        trim_long_table: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a CSV file.

        OPTIMIZATION: Only tokenizes the rows that will actually be used for encoding
        (sample_row_num rows selected via content snapshot), not all loaded rows.

        Args:
            csv_path: Path to CSV file
            context: Optional question/context text (for 'context' mode)
            context_mode: One of:
                - 'context': Use provided context (question) with table
                - 'header': Use column names as pseudo-context
                - 'column': Empty context, pure column embeddings (default)
            max_rows: Maximum rows to load as candidate pool for content snapshot

        Returns:
            dict with:
                - 'table_embedding': numpy array (embedding_dim,) - mean-pooled
                - 'column_embeddings': dict {col_idx: numpy array (embedding_dim,), ...}
                - 'context_embedding': numpy array (seq_len, embedding_dim) if context provided
                - 'column_names': list of column names
                - 'table_name': table identifier
                - 'model_type': 'VanillaTableBert' or 'VerticalAttentionTableBert'
                - 'embedding_dim': 768 or 1024
        """
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Prepare context tokens and row selection tokens
        #
        # Row selection behavior:
        # - 'context': Question-biased selection (rows matching question score higher)
        #              This matches the NSM semantic parsing pipeline behavior.
        # - 'header': First K rows (header text is for encoding context, not row selection)
        # - 'column': First K rows (matches core TaBERT inference behavior)
        #
        if context_mode == 'context':
            if context is None:
                raise ValueError("context_mode='context' requires context text")
            context_tokens = self.model.tokenizer.tokenize(context)
            # Use question for row selection (matches NSM semantic parsing)
            row_selection_tokens = context_tokens
        elif context_mode == 'header':
            # For header mode, we'll get column names from csv_to_table later
            # to avoid reading the CSV twice
            context_tokens = None  # Will be set after csv_to_table
            row_selection_tokens = None
        elif context_mode == 'column':
            context_tokens = []
            row_selection_tokens = None
        else:
            raise ValueError(f"Unknown context_mode: {context_mode}")

        # Convert CSV to Table with content snapshot selection
        # Only keep sample_row_num rows to minimize tokenization cost
        table = csv_to_table(
            csv_path,
            max_rows=max_rows,
            sample_row_num=self.sample_row_num,
            context_tokens=row_selection_tokens
        )
        table_name = table.id
        column_names = [col.name for col in table.header]

        # For header mode, set context_tokens from column names (after csv_to_table)
        if context_mode == 'header':
            header_text = ' '.join(column_names)
            context_tokens = self.model.tokenizer.tokenize(header_text)

        # Tokenize table (now only tokenizes sample_row_num rows!)
        table.tokenize(self.model.tokenizer)

        # Encode
        with torch.no_grad():
            context_encoding, column_encoding, info = self.model.encode(
                contexts=[context_tokens],
                tables=[table],
                trim_long_table=trim_long_table,
            )

        # Extract embeddings (remove batch dimension)
        context_emb = context_encoding[0].cpu().numpy()  # (seq_len, hidden_size)
        column_emb = column_encoding[0].cpu().numpy()    # (num_cols, hidden_size)

        # Build column embeddings dict
        col_embeddings = {}
        for i in range(len(column_names)):
            if i < column_emb.shape[0]:
                col_embeddings[i] = column_emb[i].astype(np.float32)

        # Compute table-level embedding variants using aggregation module
        # TaBERT native support:
        # - cls_embedding: No (TaBERT doesn't use CLS for table-level representation)
        # - table_embedding: No (no native table-level output)
        # - column_mean: Computed via aggregation
        table_embedding = {
            'cls_embedding': None,  # TaBERT doesn't use CLS for table-level
            'table_embedding': None,  # No native support
            'column_mean': aggregate_embeddings(col_embeddings, 'mean'),
        }

        result = {
            'table_id': table_name,  # Canonical identifier for downstream lookup
            'table': csv_path,  # Full path for legacy compatibility
            'table_embedding': table_embedding,
            'column_embeddings': col_embeddings,
            'column_names': column_names,
            'table_name': table_name,
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
        }

        # Include context embedding if context was provided
        if context_mode in ('context', 'header') and context_emb.shape[0] > 0:
            result['context_embedding'] = context_emb.astype(np.float32)

        return result

    def encode_directory(
        self,
        csv_dir: str,
        context_mode: str = 'column',
        max_rows: int = 100,
        show_progress: bool = True,
        existing_results: List[Dict] = None,
        processed_tables: set = None,
        checkpoint_path: Path = None,
        checkpoint_interval: int = 100,
        table_list: set = None,
    ) -> List[Dict]:
        """
        Generate embeddings for all CSV files in a directory.

        Supports resume from checkpoint: provide existing_results and processed_tables
        to skip already-processed files.

        Args:
            csv_dir: Path to directory containing CSV files
            context_mode: Context mode for all files
            max_rows: Maximum rows to load per table
            show_progress: Whether to show progress bar
            existing_results: List of already-processed results (for resume)
            processed_tables: Set of table names already processed (for resume)
            checkpoint_path: Path to save checkpoint file
            checkpoint_interval: Save checkpoint every N tables (0 to disable)
            table_list: Optional set of CSV basenames to restrict processing to

        Returns:
            List of embedding dicts (unified v2.0 format)
        """
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
        if table_list is not None:
            csv_files = [f for f in csv_files if f in table_list]
        if not csv_files:
            raise ValueError(f"No CSV files found in {csv_dir}")

        # Initialize results with existing data (unified v2.0 list format)
        if existing_results:
            if isinstance(existing_results, dict):
                # Convert dict to list, injecting table_name from key if missing
                results = [
                    {**v, 'table_name': v.get('table_name', k)}
                    for k, v in existing_results.items()
                ]
            else:
                results = list(existing_results)
        else:
            results = []
        processed_tables = processed_tables or set()

        # Filter out already-processed files
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

        for idx, csv_file in enumerate(iterator):
            csv_path = os.path.join(csv_dir, csv_file)
            result = self.encode_csv(
                csv_path,
                context_mode=context_mode,
                max_rows=max_rows
            )
            results.append(result)
            tables_since_checkpoint += 1

            # Save checkpoint at regular intervals
            if checkpoint_interval > 0 and checkpoint_path and tables_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, checkpoint_path)
                if show_progress:
                    iterator.set_postfix({'saved': len(results)})
                tables_since_checkpoint = 0

        return results

    def encode_with_questions(
        self,
        csv_path: str,
        questions: List[str],
        max_rows: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate context-aware embeddings for multiple questions on the same table.

        Efficient for semantic parsing where multiple questions use the same table.

        Args:
            csv_path: Path to CSV file
            questions: List of question strings
            max_rows: Maximum rows to load

        Returns:
            List of embedding dicts, one per question
        """
        csv_path = os.path.abspath(csv_path)

        # Convert CSV to Table once
        table = csv_to_table(csv_path, max_rows=max_rows)
        table_name = table.id
        column_names = [col.name for col in table.header]

        results = []

        for question in questions:
            # Need to re-tokenize table for each question (TaBERT requirement)
            table_copy = csv_to_table(csv_path, table_id=table_name, max_rows=max_rows)
            table_copy.tokenize(self.model.tokenizer)

            context_tokens = self.model.tokenizer.tokenize(question)

            with torch.no_grad():
                context_encoding, column_encoding, info = self.model.encode(
                    contexts=[context_tokens],
                    tables=[table_copy]
                )

            context_emb = context_encoding[0].cpu().numpy()
            column_emb = column_encoding[0].cpu().numpy()

            col_embeddings = {}
            for i in range(len(column_names)):
                if i < column_emb.shape[0]:
                    col_embeddings[i] = column_emb[i].astype(np.float32)

            # Compute table-level embedding variants using aggregation module
            table_embedding = {
                'cls_embedding': None,  # TaBERT doesn't use CLS for table-level
                'table_embedding': None,  # No native support
                'column_mean': aggregate_embeddings(col_embeddings, 'mean'),
            }

            results.append({
                'table_id': table_name,  # Canonical identifier for downstream lookup
                'table': csv_path,  # Full path for legacy compatibility
                'question': question,
                'table_embedding': table_embedding,
                'column_embeddings': col_embeddings,
                'context_embedding': context_emb.astype(np.float32),
                'column_names': column_names,
                'table_name': table_name,
                'model_type': self.model_type,
                'embedding_dim': self.embedding_dim,
            })

        return results


# Backward-compatible function API
def get_column_embeddings(
    csv_path: str,
    checkpoint_path: str,
    context: str = None,
    context_mode: str = 'column',
    device: str = None
) -> Dict:
    """
    Generate column embeddings from a single CSV file.

    For batch processing, use TaBERTEmbedder class directly.
    """
    embedder = TaBERTEmbedder(checkpoint_path, device)
    return embedder.encode_csv(csv_path, context=context, context_mode=context_mode)


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings from CSV file(s) using TaBERT'
    )
    parser.add_argument('--input', '--csv', type=str, required=True, dest='input',
                        help='Path to CSV file or directory of CSV files')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to TaBERT checkpoint (model.bin)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file (default: auto-generated)')
    parser.add_argument('--context', type=str, default=None,
                        help='Context/question text for context-aware embeddings')
    parser.add_argument('--context_mode', type=str, default='column',
                        choices=['context', 'header', 'column'],
                        help='Context mode: context (with question), header (column names), column (empty)')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows to load from CSV (default: 100)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N tables (default: 100). Set to 0 to disable.')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()

    # Validate context mode
    if args.context_mode == 'context' and args.context is None:
        parser.error("--context is required when --context_mode=context")

    # Determine if input is file or directory
    is_directory = os.path.isdir(args.input)

    # Default output filename
    if args.output is None:
        if is_directory:
            args.output = 'column_embeddings.pkl'
        else:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}_embeddings.pkl"

    # Load model once
    embedder = TaBERTEmbedder(args.checkpoint, args.device)

    table_list = None
    if args.table_list:
        from trl_bench.utils.table_list import load_table_list
        table_list = load_table_list(args.table_list)

    # Process input
    if is_directory:
        print(f"Processing directory: {args.input}")

        # Load checkpoint/resume support
        existing_results, processed_tables, checkpoint_path = load_checkpoint_data(args.output)

        start_time = time.time()
        results = embedder.encode_directory(
            args.input,
            context_mode=args.context_mode,
            max_rows=args.max_rows,
            existing_results=existing_results,
            processed_tables=processed_tables,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            table_list=table_list,
        )
        inference_time = time.time() - start_time

        # Save final results
        with open(args.output, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        # Remove checkpoint file after successful completion
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"Checkpoint file removed (processing complete).")
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint file: {e}")

        # Summary
        new_tables = len(results) - len(existing_results)
        print("\n" + "=" * 60)
        print("BATCH EMBEDDING EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Tables processed: {len(results)} total ({new_tables} new)")
        print(f"Output saved to: {args.output}")
        print(f"Inference time: {inference_time:.2f} seconds")
        print("=" * 60)

    else:
        print(f"Processing file: {args.input}")
        result = embedder.encode_csv(
            args.input,
            context=args.context,
            context_mode=args.context_mode,
            max_rows=args.max_rows
        )

        with open(args.output, 'wb') as f:
            pickle.dump(result, f)

        print("\n" + "=" * 60)
        print("EMBEDDING EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Table: {result['table_name']}")
        print(f"Model type: {result['model_type']}")
        print(f"Columns: {len(result['column_embeddings'])}")
        print(f"Column names: {result['column_names']}")
        print(f"Embedding dimension: {result['embedding_dim']}")
        print(f"Context mode: {args.context_mode}")
        print(f"Output saved to: {args.output}")
        print("=" * 60)


if __name__ == '__main__':
    main()
