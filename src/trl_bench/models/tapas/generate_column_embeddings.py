#!/usr/bin/env python
"""
Generate column embeddings from CSV files using TAPAS (HuggingFace implementation).

TAPAS (Table Parser) is a BERT-based model for table understanding, introduced in:
"TAPAS: Weakly Supervised Table Parsing via Pre-training" (Herzig et al., ACL 2020)
https://arxiv.org/abs/2004.02349

Supports both single file and batch (directory) modes. Model is loaded once.
Supports two encoding modes:
  - 'table': Pure table embeddings (no question context)
  - 'question': Question-aware embeddings using provided question text

RESUME SUPPORT
==============
This script supports resuming from interruptions:
- Checkpoint file (.checkpoint.pkl) saved alongside output every N tables
- On restart, automatically detects and loads checkpoint
- Skips already-processed tables
- Checkpoint removed after successful completion

Usage:
    # Single file - table-only mode (default)
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --output embeddings.pkl

    # With question context
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --question "What is the total revenue?" \
        --output embeddings.pkl

    # Batch mode (directory of CSVs)
    python generate_column_embeddings.py \
        --input /path/to/csv_directory/ \
        --output all_embeddings.pkl

    # With checkpoint interval (for large datasets)
    python generate_column_embeddings.py \
        --input /path/to/csv_directory/ \
        --output all_embeddings.pkl \
        --checkpoint_interval 100

    # Use specific model variant
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --model google/tapas-large \
        --output embeddings.pkl

Note:
    - By default, uses google/tapas-base model (768-dim embeddings)
    - Available variants: tapas-tiny, tapas-mini, tapas-small, tapas-base, tapas-large
    - Fine-tuned variants available: tapas-base-finetuned-wtq, tapas-base-finetuned-sqa, etc.
"""

import os
import sys
import pickle
import argparse
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Suppress FutureWarnings from TAPAS tokenizer (pandas Series indexing deprecation)
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.models.tapas')

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path for potential shared utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

    # Prune stale entries missing token_mean so they get re-encoded.
    # Earlier code versions didn't produce token_mean; checkpoint resume
    # would otherwise carry those entries forward indefinitely.
    if existing_results:
        stale = [
            e['table_name'] for e in existing_results
            if isinstance(e.get('table_embedding'), dict)
            and 'token_mean' not in e['table_embedding']
        ]
        if stale:
            stale_set = set(stale)
            existing_results = [e for e in existing_results if e['table_name'] not in stale_set]
            processed_tables -= stale_set
            print(f"  Pruned {len(stale)} stale entries missing token_mean (will re-encode)")

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


def _extract_column_embeddings(
    hidden_states: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    column_ids: torch.Tensor,
    num_columns: int
) -> Dict[int, np.ndarray]:
    """
    Extract per-column embeddings from TAPAS hidden states.

    TAPAS uses column_ids in token_type_ids to identify which column each token belongs to.
    Column ID 0 typically means the token is not part of any column (e.g., question tokens).

    Args:
        hidden_states: Last hidden state [seq_len, hidden_dim]
        token_type_ids: Token type IDs [seq_len, 7] (TAPAS uses 7 token types)
        attention_mask: Attention mask [seq_len]
        column_ids: Column IDs for each token [seq_len]
        num_columns: Number of columns in the table

    Returns:
        Dict mapping column index (0-based) to mean-pooled embedding
    """
    col_embeddings = {}
    hidden_dim = hidden_states.shape[-1]

    # Column IDs in TAPAS are 1-indexed (0 = not a column token)
    for col_idx in range(1, num_columns + 1):
        # Find tokens belonging to this column
        col_mask = (column_ids == col_idx) & (attention_mask == 1)

        if col_mask.sum() > 0:
            col_tokens = hidden_states[col_mask]
            col_embedding = col_tokens.mean(dim=0).cpu().numpy()
            col_embeddings[col_idx - 1] = col_embedding.astype(np.float32)
        else:
            # No tokens for this column (might be truncated)
            col_embeddings[col_idx - 1] = np.zeros(hidden_dim, dtype=np.float32)

    return col_embeddings


class TAPASEmbedder:
    """
    TAPAS embedder using HuggingFace Transformers.

    Generates table and column embeddings from CSV files using the pre-trained
    TAPAS model. Supports question-aware embeddings for semantic parsing tasks.
    """

    # Available pre-trained models
    AVAILABLE_MODELS = {
        # Base models (pre-trained only)
        'google/tapas-base': 768,
        'google/tapas-large': 1024,
        'google/tapas-small': 512,
        'google/tapas-mini': 256,
        'google/tapas-tiny': 128,
        # Fine-tuned on WikiTableQuestions
        'google/tapas-base-finetuned-wtq': 768,
        'google/tapas-large-finetuned-wtq': 1024,
        'google/tapas-small-finetuned-wtq': 512,
        'google/tapas-mini-finetuned-wtq': 256,
        'google/tapas-tiny-finetuned-wtq': 128,
        # Fine-tuned on Sequential Question Answering
        'google/tapas-base-finetuned-sqa': 768,
        'google/tapas-large-finetuned-sqa': 1024,
        'google/tapas-small-finetuned-sqa': 512,
        'google/tapas-mini-finetuned-sqa': 256,
        'google/tapas-tiny-finetuned-sqa': 128,
        # Fine-tuned on WikiSQL (supervised)
        'google/tapas-base-finetuned-wikisql-supervised': 768,
        'google/tapas-large-finetuned-wikisql-supervised': 1024,
    }

    def __init__(
        self,
        model_name: str = 'google/tapas-base',
        device: str = None,
        max_length: int = 512,
        max_cell_chars: int = 200000
    ):
        """
        Initialize TAPAS embedder.

        Args:
            model_name: HuggingFace model identifier (default: google/tapas-base)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length (default: 512)
            max_cell_chars: Max chars per cell before truncation (default: 200000)
        """
        try:
            from transformers import TapasTokenizer, TapasModel, TapasConfig
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.max_length = max_length
        self.max_cell_chars = max_cell_chars
        self.model_name = model_name

        print(f"Loading TAPAS model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Get embedding dimension from config
        self.embedding_dim = self.model.config.hidden_size

        print(f"Model loaded successfully")
        print(f"Device: {device}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Max sequence length: {max_length}")

    def _prepare_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 100
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for TAPAS encoding.

        TAPAS requires all values to be strings and handles tables
        with a specific format.

        Args:
            df: Input DataFrame
            max_rows: Maximum rows to include

        Returns:
            Prepared DataFrame with string values
        """
        # Limit rows
        df = df.head(max_rows).copy()

        # Ensure stable integer index for TAPAS numeric parsing (uses iloc)
        df = df.reset_index(drop=True)

        # Convert all values to strings (TAPAS requirement)
        df = df.fillna('')
        df = df.astype(str)

        # Truncate extremely long cells to avoid pathological tokenization time.
        # Set to a high default to avoid impacting typical tables.
        if self.max_cell_chars is not None:
            for col in df.columns:
                series = df[col]
                try:
                    max_len = series.str.len().max()
                except Exception:
                    max_len = None
                if max_len is not None and max_len > self.max_cell_chars:
                    print(
                        f"Warning: Truncating cells in column '{col}' "
                        f"to {self.max_cell_chars} chars (max observed {max_len})."
                    )
                    df[col] = series.str.slice(0, self.max_cell_chars)

        # Ensure column names are strings
        df.columns = [str(c) for c in df.columns]

        return df

    def _is_header_too_long(self, err: Exception) -> bool:
        msg = str(err)
        return ("query and table header" in msg and "max_length" in msg) or (
            "table header results in a length" in msg
        ) or ("Too many columns" in msg)

    def _tokenize_table(self, df: pd.DataFrame, question: str):
        return self.tokenizer(
            table=df,
            queries=question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def _tokenize_with_column_drop(
        self,
        df: pd.DataFrame,
        question: str
    ):
        header_error = None
        try:
            inputs = self._tokenize_table(df, question)
            return inputs, list(range(len(df.columns))), []
        except Exception as e:
            if not self._is_header_too_long(e):
                raise
            header_error = e

        num_cols = len(df.columns)
        if num_cols <= 1:
            raise

        best = None
        best_inputs = None
        low, high = 1, num_cols
        while low <= high:
            mid = (low + high) // 2
            try:
                inputs = self._tokenize_table(df.iloc[:, :mid], question)
                best = mid
                best_inputs = inputs
                low = mid + 1
            except Exception as e:
                if self._is_header_too_long(e):
                    high = mid - 1
                else:
                    raise

        if best is None:
            if header_error is not None:
                raise header_error
            raise RuntimeError("Tokenization failed: header too long even after dropping columns.")

        kept = list(range(best))
        dropped = list(range(best, num_cols))
        return best_inputs, kept, dropped

    def encode_csv(
        self,
        csv_path: str,
        question: str = None,
        max_rows: int = 100,
        delimiter: str = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a CSV file.

        Args:
            csv_path: Path to CSV file
            question: Optional question text for question-aware embeddings
            max_rows: Maximum rows to load from CSV
            delimiter: CSV delimiter (default: auto-detect, falls back to ',')

        Returns:
            dict with:
                - 'table_embedding': numpy array (embedding_dim,) - mean-pooled
                - 'column_embeddings': dict {col_idx: numpy array (embedding_dim,), ...}
                - 'cls_embedding': numpy array (embedding_dim,) - CLS token
                - 'column_names': list of column names
                - 'table_name': table identifier (filename without extension)
                - 'model_name': model identifier used
                - 'embedding_dim': embedding dimension (768 for base)
                - 'question': question text if provided
        """
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load and prepare table
        if delimiter:
            df = pd.read_csv(csv_path, nrows=max_rows, delimiter=delimiter, dtype=str)
        else:
            # Try to auto-detect delimiter
            try:
                df = pd.read_csv(csv_path, nrows=max_rows, dtype=str)
            except Exception:
                # Try common delimiters
                for delim in [',', '#', '\t', ';']:
                    try:
                        df = pd.read_csv(csv_path, nrows=max_rows, delimiter=delim, dtype=str)
                        if len(df.columns) > 1:
                            break
                    except Exception:
                        continue

        # Fall back to python engine if C engine failed on all delimiters
        # (handles embedded \r in quoted fields that trigger C parser buffer overflow)
        if 'df' not in locals():
            df = pd.read_csv(csv_path, nrows=max_rows, dtype=str, engine='python')

        df = self._prepare_table(df, max_rows)

        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        original_columns = list(df.columns)

        # TAPAS column ID vocabulary is 256 (index 0 = non-table tokens),
        # so the model can represent at most 255 data columns. Pre-cap here
        # to avoid CUDA assertion failures in the embedding layer. The repair
        # stage will fill in dropped columns afterwards.
        max_col_ids = 255
        if len(df.columns) > max_col_ids:
            print(
                f"Warning: Pre-capping {len(df.columns)} columns to {max_col_ids} "
                f"(TAPAS column ID limit) for {csv_path}"
            )
            df = df.iloc[:, :max_col_ids]

        # Use empty question if none provided (for pure table embeddings)
        if question is None:
            question = ""

        # Tokenize table with question
        try:
            inputs, kept_indices, dropped_indices = self._tokenize_with_column_drop(df, question)
        except Exception as e:
            # Handle edge cases (empty tables, etc.)
            raise RuntimeError(f"Tokenization failed for {csv_path}: {e}")

        if dropped_indices:
            df = df.iloc[:, kept_indices]
            column_names = [original_columns[i] for i in kept_indices]
            dropped_names = [original_columns[i] for i in dropped_indices]
            print(
                f"Warning: Dropped {len(dropped_indices)} columns to fit TAPAS max_length "
                f"for {csv_path}"
            )
        else:
            column_names = original_columns
            dropped_names = []

        num_columns = len(column_names)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract hidden states (last layer)
        hidden_states = outputs.last_hidden_state[0]  # Remove batch dim
        attention_mask = inputs['attention_mask'][0]

        # Get token type IDs for column identification
        # TAPAS token_type_ids has shape [batch, seq_len, 7]:
        # [segment_ids, column_ids, row_ids, prev_label, column_ranks, inv_column_ranks, numeric_relations]
        token_type_ids = inputs['token_type_ids'][0]
        column_ids = token_type_ids[:, 1]  # Column IDs are at index 1

        # CLS embedding (first token) - in TAPAS, CLS represents the table
        cls_embedding = hidden_states[0].cpu().numpy().astype(np.float32)

        # Extract per-column embeddings
        col_embeddings = _extract_column_embeddings(
            hidden_states, token_type_ids, attention_mask, column_ids, num_columns
        )
        if dropped_indices:
            remapped: Dict[int, np.ndarray] = {}
            for local_idx, vec in col_embeddings.items():
                if local_idx < len(kept_indices):
                    remapped[int(kept_indices[local_idx])] = vec
            col_embeddings = remapped

        # Compute table-level embedding variants using aggregation module
        # TAPAS native support:
        # - cls_embedding: Yes (TAPAS CLS token represents table context)
        # - table_embedding: No (no native table-level output)
        # - column_mean: Computed via aggregation
        # - token_mean: Mean of all non-padding token hidden states
        mask = attention_mask.unsqueeze(-1).float()  # (seq_len, 1)
        token_mean = ((hidden_states * mask).sum(dim=0) / mask.sum()).cpu().numpy().astype(np.float32)

        table_embedding = {
            'cls_embedding': cls_embedding,
            'table_embedding': None,  # No native support
            'column_mean': aggregate_embeddings(col_embeddings, 'mean'),
            'token_mean': token_mean,
        }

        result = {
            'table_id': table_name,  # Canonical identifier for downstream lookup
            'table': csv_path,  # Full path for legacy compatibility
            'table_embedding': table_embedding,
            'column_embeddings': col_embeddings,
            'column_names': column_names,
            'table_name': table_name,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }
        if dropped_names:
            result['dropped_column_indices'] = dropped_indices
            result['dropped_column_names'] = dropped_names

        if question:
            result['question'] = question

        return result

    def encode_directory(
        self,
        csv_dir: str,
        question: str = None,
        max_rows: int = 100,
        show_progress: bool = True,
        existing_results: List[Dict] = None,
        processed_tables: set = None,
        checkpoint_path: Path = None,
        checkpoint_interval: int = 100,
        log_each_table: bool = False,
        table_list: set = None,
    ) -> List[Dict]:
        """
        Generate embeddings for all CSV files in a directory.

        Supports resume from checkpoint: provide existing_results and processed_tables
        to skip already-processed files.

        Args:
            csv_dir: Path to directory containing CSV files
            question: Optional question text for all tables
            max_rows: Maximum rows to load per table
            show_progress: Whether to show progress bar
            existing_results: List of already-processed results (for resume)
            processed_tables: Set of table names already processed (for resume)
            checkpoint_path: Path to save checkpoint file
            checkpoint_interval: Save checkpoint every N tables (0 to disable)
            log_each_table: Log each table before encoding (for debugging)
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

        total_tables = len(csv_files)
        for idx, csv_file in enumerate(iterator):
            csv_path = os.path.join(csv_dir, csv_file)
            if log_each_table:
                msg = f"[{idx + 1}/{total_tables}] Encoding {csv_file}"
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg, flush=True)
            result = self.encode_csv(csv_path, question=question, max_rows=max_rows)
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
        Generate question-aware embeddings for multiple questions on the same table.

        Efficient for QA tasks where multiple questions use the same table.

        Args:
            csv_path: Path to CSV file
            questions: List of question strings
            max_rows: Maximum rows to load

        Returns:
            List of embedding dicts, one per question
        """
        results = []

        for question in questions:
            result = self.encode_csv(csv_path, question=question, max_rows=max_rows)
            results.append(result)

        return results


# Backward-compatible function API
def get_column_embeddings(
    csv_path: str,
    model_name: str = 'google/tapas-base',
    question: str = None,
    max_cell_chars: int = 200000,
    device: str = None
) -> Dict:
    """
    Generate column embeddings from a single CSV file.

    For batch processing, use TAPASEmbedder class directly.

    Args:
        csv_path: Path to CSV file
        model_name: HuggingFace model identifier
        question: Optional question text
        device: Device to use

    Returns:
        dict with embeddings (see TAPASEmbedder.encode_csv)
    """
    embedder = TAPASEmbedder(model_name=model_name, device=device, max_cell_chars=max_cell_chars)
    return embedder.encode_csv(csv_path, question=question)


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings from CSV file(s) using TAPAS'
    )
    parser.add_argument('--input', '--csv', type=str, required=True, dest='input',
                        help='Path to CSV file or directory of CSV files')
    parser.add_argument('--model', type=str, default='google/tapas-base',
                        help='HuggingFace model name (default: google/tapas-base)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file (default: auto-generated)')
    parser.add_argument('--question', type=str, default=None,
                        help='Question text for question-aware embeddings')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows to load from CSV (default: 100)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--max_cell_chars', type=int, default=200000,
                        help='Max chars per cell before truncation (default: 200000).')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N tables (default: 100). Set to 0 to disable.')
    parser.add_argument('--log_each_table', action='store_true',
                        help='Log each table before encoding (for debugging stalls).')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()

    # Determine if input is file or directory
    is_directory = os.path.isdir(args.input)

    # Default output filename
    if args.output is None:
        if is_directory:
            args.output = 'tapas_embeddings.pkl'
        else:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}_tapas_embeddings.pkl"

    # Load model once
    embedder = TAPASEmbedder(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        max_cell_chars=args.max_cell_chars
    )

    table_list = None
    if args.table_list:
        from trl_bench.utils.table_list import load_table_list
        table_list = load_table_list(args.table_list)

    # Process input
    if is_directory:
        print(f"\nProcessing directory: {args.input}")

        # Load checkpoint/resume support
        existing_results, processed_tables, checkpoint_path = load_checkpoint_data(args.output)

        start_time = time.time()
        results = embedder.encode_directory(
            args.input,
            question=args.question,
            max_rows=args.max_rows,
            existing_results=existing_results,
            processed_tables=processed_tables,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            log_each_table=args.log_each_table,
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
        print(f"Model: {args.model}")
        print(f"Tables processed: {len(results)} total ({new_tables} new)")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        if args.question:
            print(f"Question: {args.question}")
        print(f"Output saved to: {args.output}")
        print(f"Inference time: {inference_time:.2f} seconds")
        print("=" * 60)

    else:
        print(f"\nProcessing file: {args.input}")
        result = embedder.encode_csv(
            args.input,
            question=args.question,
            max_rows=args.max_rows
        )

        with open(args.output, 'wb') as f:
            pickle.dump(result, f)

        print("\n" + "=" * 60)
        print("EMBEDDING EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Table: {result['table_name']}")
        print(f"Model: {result['model_name']}")
        print(f"Columns: {len(result['column_embeddings'])}")
        print(f"Column names: {result['column_names']}")
        print(f"Embedding dimension: {result['embedding_dim']}")
        if args.question:
            print(f"Question: {args.question}")
        print(f"Output saved to: {args.output}")
        print("=" * 60)


if __name__ == '__main__':
    main()
