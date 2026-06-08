#!/usr/bin/env python3
"""
Generate column embeddings for datasets using the repo's TURL extraction path.

This script processes CSV tables and generates embeddings using a frozen TURL
encoder in a cell-content-only, mode-4-style configuration. It writes the
repo's unified column-embedding format for downstream tasks.

RESUME SUPPORT
==============
This script supports resuming from interruptions:
- Checkpoint file (.checkpoint.pkl) saved alongside output every N tables
- On restart, automatically detects and loads checkpoint
- Skips already-processed tables
- Checkpoint removed after successful completion

Output format:
    [{
        'table_id': 'table_name',      # Canonical identifier
        'table': 'path/to/table.csv',  # Full path to source file
        'table_embedding': {
            'cls_embedding': None,         # Not produced by this TURL path
            'table_embedding': None,       # No native table embedding exposed
            'column_mean': np.array([312]) # Mean of column embeddings
        },
        'column_embeddings': {
            0: np.array([312]),
            1: np.array([312]),
            ...
        },
        'column_names': ['col1', 'col2', ...]
    }, ...]

Usage:
    # Process a directory of CSV files
    python generate_column_embeddings_dataset.py \\
        --mode table_directory \\
        --input_dir datasets/ecb_join/tables \\
        --output_file embeddings/column/turl/ecb_join.pkl

    # Process with custom model checkpoint
    python generate_column_embeddings_dataset.py \\
        --mode table_directory \\
        --input_dir datasets/santos/datalake \\
        --output_file embeddings/column/turl/santos.pkl \\
        --checkpoint checkpoints/turl/pretrained

    # With checkpoint interval (for large datasets)
    python generate_column_embeddings_dataset.py \\
        --mode table_directory \\
        --input_dir datasets/large_dataset \\
        --output_file embeddings/column/turl/large_dataset.pkl \\
        --checkpoint_interval 100
"""

import argparse
import os
import sys
import pickle
import csv
import glob
import time
from pathlib import Path

# Fix compatibility issues
import setuptools
import distutils.version

# Allow large CSV fields (some datasets have very long cells).
try:
    csv.field_size_limit(100_000_000)
except (OverflowError, ValueError):
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = max_size // 10
            if max_size < 1_000_000:
                break

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Fix torch._six compatibility issue
if not hasattr(torch, '_six'):
    import types
    torch._six = types.ModuleType('_six')
    torch._six.string_classes = (str, bytes)

# Add code directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(script_dir, 'code')
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, code_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.model.configuration import TableConfig
from code.model.model import HybridTableModel
from code.model.transformers import BertTokenizer
from trl_bench.utils.aggregation import aggregate_embeddings


def extract_table_id(table_path: str) -> str:
    """
    Extract table_id from a table path.

    Removes directory path and file extension to get canonical identifier.
    Handles compression suffixes (.gz, .bz2) and double extensions (.csv.gz).

    Args:
        table_path: Table filename or path (e.g., 'path/to/table.csv.gz')

    Returns:
        Clean table_id (e.g., 'table')
    """
    basename = os.path.basename(table_path)
    # Strip compression suffixes first
    for ext in ['.gz', '.bz2']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
    # Then strip data format extensions
    for ext in ['.csv', '.json', '.tsv', '.parquet']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
    return basename


def load_checkpoint_data(output_path: str):
    """
    Load existing checkpoint or output file.

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
            processed_tables = {e['table'] for e in existing_results}
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
            processed_tables = {e['table'] for e in existing_results}
            print(f"  Loaded {len(existing_results)} already-processed tables from output")
        except Exception as e:
            print(f"  Warning: Failed to load output file: {e}")
            existing_results = []
            processed_tables = set()

    return existing_results, processed_tables, checkpoint_path


def save_checkpoint(results: list, checkpoint_path: Path):
    """
    Save current progress to checkpoint file.

    Args:
        results: List of result entries
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


class TableDirectoryDataset(Dataset):
    """
    Dataset for processing a directory of CSV files.
    Each CSV file is treated as a separate table.
    """

    def __init__(
        self,
        csv_files,
        max_rows=100,
        max_cell_length=64,
        max_cell_chars=512,
        max_entities=None,
        tokenizer=None,
        fail_on_read_error=True,
    ):
        """
        Args:
            csv_files: List of CSV file paths
            max_rows: Maximum number of rows to process per table
            max_cell_length: Maximum tokens per cell
            max_entities: Maximum number of entities (cells) per table
            tokenizer: BERT tokenizer
            fail_on_read_error: Fail fast on CSV read errors instead of skipping
        """
        self.csv_files = csv_files
        self.max_rows = max_rows
        self.max_cell_length = max_cell_length
        # Character cap avoids tokenizing megabyte-long cells; this does not
        # change the first max_cell_length tokens for bert-base-uncased.
        self.max_cell_chars = max_cell_chars
        self.max_entities = max_entities
        self.fail_on_read_error = fail_on_read_error

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Lazy loading: defer preprocessing to __getitem__ to avoid
        # holding all tables in memory at once.

    def _read_csv(self, csv_path):
        """Read header + up to max_rows rows (streaming) to bound memory."""
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers is None:
                    return None, None, csv_path

                data_rows = []
                for row_idx, row in enumerate(reader):
                    if row_idx >= self.max_rows:
                        break
                    data_rows.append(row)

            return headers, data_rows, csv_path
        except Exception as e:
            if self.fail_on_read_error:
                raise RuntimeError(f"Failed to read {csv_path}: {e}") from e
            print(f"Warning: Failed to read {csv_path}: {e}")
            return None, None, csv_path

    def _preprocess_table(self, csv_path):
        """Preprocess a single CSV table for TURL Mode 4."""
        headers, data_rows, table_path = self._read_csv(csv_path)

        if headers is None or not data_rows:
            return None

        num_columns = len(headers)

        if self.max_entities is not None and num_columns > 0:
            max_rows_by_entities = self.max_entities // num_columns
            if max_rows_by_entities < 1:
                # Can't satisfy the cap without dropping columns.
                print(
                    f"Warning: {csv_path} has {num_columns} columns > "
                    f"max_entities {self.max_entities}; keeping 1 row to preserve columns."
                )
                max_rows_by_entities = 1
            if len(data_rows) > max_rows_by_entities:
                print(
                    f"Info: Capping rows for {csv_path}: {len(data_rows)} -> "
                    f"{max_rows_by_entities} (cols={num_columns}, max_entities={self.max_entities})"
                )
                data_rows = data_rows[:max_rows_by_entities]

        # Build entity (cell) inputs
        input_ent_text = []
        input_ent_type = []
        column_en_map = {}
        row_en_map = {}

        entity_idx = 0
        for row_idx, row in enumerate(data_rows):
            for col_idx, cell_value in enumerate(row):
                if col_idx >= num_columns:
                    continue

                cell_text = str(cell_value).strip()
                if not cell_text:
                    cell_text = "[EMPTY]"
                elif self.max_cell_chars is not None and len(cell_text) > self.max_cell_chars:
                    cell_text = cell_text[:self.max_cell_chars]

                tokenized = self.tokenizer.encode(
                    cell_text,
                    max_length=self.max_cell_length,
                    add_special_tokens=False
                )
                if not tokenized:
                    tokenized = [self.tokenizer.unk_token_id]

                input_ent_text.append(tokenized)
                input_ent_type.append(4)

                if col_idx not in column_en_map:
                    column_en_map[col_idx] = []
                column_en_map[col_idx].append(entity_idx)

                if row_idx not in row_en_map:
                    row_en_map[row_idx] = []
                row_en_map[row_idx].append(entity_idx)

                entity_idx += 1

        num_entities = len(input_ent_text)

        if num_entities == 0:
            # Return None to skip empty tables; _preprocess() handles this
            return None

        # Create column_entity_mask
        column_entity_mask = np.zeros([num_columns, num_entities], dtype=np.float32)
        for col_idx in range(num_columns):
            if col_idx in column_en_map:
                for ent_idx in column_en_map[col_idx]:
                    column_entity_mask[col_idx, ent_idx] = 1.0

        # Create entity-entity attention mask
        ent_ent_mask = np.eye(num_entities, dtype=np.int32)

        for col_idx, ent_indices in column_en_map.items():
            for i in ent_indices:
                for j in ent_indices:
                    ent_ent_mask[i, j] = 1

        for row_idx, ent_indices in row_en_map.items():
            for i in ent_indices:
                for j in ent_indices:
                    ent_ent_mask[i, j] = 1

        # Pad entity text
        input_ent_cell_length = [len(x) for x in input_ent_text]
        max_cell_len = max(input_ent_cell_length)
        input_ent_text_padded = np.zeros([num_entities, max_cell_len], dtype=np.int64)
        for i, tokens in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(tokens)] = tokens

        return {
            'table_path': table_path,
            'headers': headers,
            'num_columns': num_columns,
            'num_entities': num_entities,
            'input_ent_text': input_ent_text_padded,
            'input_ent_text_length': np.array(input_ent_cell_length, dtype=np.int64),
            'input_ent_type': np.array(input_ent_type, dtype=np.int64),
            'input_ent_mask': ent_ent_mask,
            'column_entity_mask': column_entity_mask,
        }

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, index):
        csv_path = self.csv_files[index]
        return self._preprocess_table(csv_path)


def collate_fn(batch):
    """Collate function for DataLoader"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    table_paths = [item['table_path'] for item in batch]
    headers_list = [item['headers'] for item in batch]
    num_columns_list = [item['num_columns'] for item in batch]

    max_entities = max(item['num_entities'] for item in batch)
    max_cell_len = max(item['input_ent_text'].shape[1] for item in batch)
    max_columns = max(item['num_columns'] for item in batch)
    batch_size = len(batch)

    padded_ent_text = np.zeros((batch_size, max_entities, max_cell_len), dtype=np.int64)
    padded_ent_length = np.ones((batch_size, max_entities), dtype=np.int64)
    padded_ent_type = np.zeros((batch_size, max_entities), dtype=np.int64)
    padded_ent_mask = np.zeros((batch_size, max_entities, max_entities), dtype=np.int64)
    padded_col_mask = np.zeros((batch_size, max_columns, max_entities), dtype=np.float32)

    for i, item in enumerate(batch):
        n_ent = item['num_entities']
        n_col = item['num_columns']
        cell_len = item['input_ent_text'].shape[1]

        padded_ent_text[i, :n_ent, :cell_len] = item['input_ent_text']
        padded_ent_length[i, :n_ent] = item['input_ent_text_length']
        padded_ent_type[i, :n_ent] = item['input_ent_type']
        padded_ent_mask[i, :n_ent, :n_ent] = item['input_ent_mask']
        padded_col_mask[i, :n_col, :n_ent] = item['column_entity_mask']
        if n_col < max_columns:
            padded_col_mask[i, n_col:, 0] = 1.0

    return {
        'table_paths': table_paths,
        'headers': headers_list,
        'num_columns': num_columns_list,
        'input_ent_text': torch.LongTensor(padded_ent_text),
        'input_ent_text_length': torch.LongTensor(padded_ent_length),
        'input_ent_type': torch.LongTensor(padded_ent_type),
        'input_ent_mask': torch.LongTensor(padded_ent_mask),
        'column_entity_mask': torch.FloatTensor(padded_col_mask),
    }


class TURLEmbeddingExtractor:
    """Extract column embeddings using TURL Mode 4"""

    def __init__(self, model_path, device='cuda'):
        self.device = device

        print(f"Loading TURL model from {model_path}")

        config = TableConfig.from_pretrained(model_path)
        config.output_attentions = False
        config.output_hidden_states = False

        self.model = HybridTableModel(config, is_simple=True)

        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        table_state_dict = {
            k.replace('table.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('table.')
        }
        self.model.load_state_dict(table_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = config.hidden_size
        print(f"Model loaded on {device} (hidden_size={self.hidden_size})")

    def extract_embeddings(self, dataloader, existing_results=None, checkpoint_path=None,
                            checkpoint_interval=100):
        """
        Extract column embeddings and format for downstream tasks.

        Args:
            dataloader: DataLoader containing table data
            existing_results: List of already-processed results (for resume)
            checkpoint_path: Path to checkpoint file (for saving progress)
            checkpoint_interval: Save checkpoint every N tables (0 to disable)

        Returns:
            List of dicts in unified format:
            [{
                'table': str,
                'cls_embedding': None,
                'table_embedding': np.array([hidden_size]),
                'column_embeddings': {col_idx: np.array([hidden_size]), ...}  # Unified format (plural)
            }, ...]
        """
        # Combine existing results with new ones
        results = list(existing_results) if existing_results else []
        tables_since_checkpoint = 0
        initial_count = len(results)

        print("\nExtracting TURL Mode 4 embeddings...")

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Processing tables")
            for batch in pbar:
                if batch is None:
                    continue
                table_paths = batch['table_paths']
                headers_list = batch['headers']
                num_columns_list = batch['num_columns']

                input_ent_text = batch['input_ent_text'].to(self.device)
                input_ent_text_length = batch['input_ent_text_length'].to(self.device)
                input_ent_type = batch['input_ent_type'].to(self.device)
                input_ent_mask = batch['input_ent_mask'].to(self.device)
                column_entity_mask = batch['column_entity_mask'].to(self.device)

                # TURL Mode 4: Only entity stream
                tok_outputs, ent_outputs, _ = self.model(
                    None, None, None, None,
                    input_ent_text,
                    input_ent_text_length,
                    None,
                    None,
                    input_ent_type,
                    input_ent_mask,
                    None
                )

                ent_sequence_output = ent_outputs[0]

                # Aggregate to column level
                ent_col_output = torch.matmul(column_entity_mask, ent_sequence_output)
                ent_col_output /= column_entity_mask.sum(dim=-1, keepdim=True).clamp(1.0, 9999.0)

                ent_col_output_cpu = ent_col_output.cpu().numpy()

                # Format for downstream tasks
                for i in range(len(table_paths)):
                    num_cols = num_columns_list[i]
                    col_embeddings = ent_col_output_cpu[i, :num_cols, :]

                    # Create column_embedding dict
                    column_embedding = {
                        col_idx: col_embeddings[col_idx].copy()
                        for col_idx in range(num_cols)
                    }

                    # Compute table-level embedding variants using aggregation module
                    # TURL native support:
                    # - cls_embedding: No (entity-based model, no CLS)
                    # - table_embedding: No (no native table-level output)
                    # - column_mean: Computed via aggregation
                    table_embedding = {
                        'cls_embedding': None,  # TURL doesn't use CLS
                        'table_embedding': None,  # No native support
                        'column_mean': aggregate_embeddings(column_embedding, 'mean'),
                    }

                    results.append({
                        'table_id': extract_table_id(table_paths[i]),
                        'table': table_paths[i],
                        'table_embedding': table_embedding,
                        'column_embeddings': column_embedding,  # Unified format (plural)
                        'column_names': headers_list[i],  # Column header names
                    })
                    tables_since_checkpoint += 1

                # Save checkpoint at regular intervals
                if checkpoint_interval > 0 and checkpoint_path and tables_since_checkpoint >= checkpoint_interval:
                    save_checkpoint(results, checkpoint_path)
                    pbar.set_postfix({'saved': len(results)})
                    tables_since_checkpoint = 0

            pbar.close()

        new_tables = len(results) - initial_count
        print(f"Extracted embeddings for {len(results)} tables ({new_tables} new)")
        return results


def find_csv_files(path):
    """Find CSV files from path (file or directory)"""
    if os.path.isfile(path):
        if path.endswith('.csv'):
            return [path]
        else:
            raise ValueError(f"Not a CSV file: {path}")
    elif os.path.isdir(path):
        csv_files = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.csv'):
                    csv_files.append(entry.path)
        if not csv_files:
            for root, _, files in os.walk(path):
                for name in files:
                    if name.endswith('.csv'):
                        csv_files.append(os.path.join(root, name))
        return sorted(csv_files)
    else:
        raise ValueError(f"Path not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings for dataset using TURL Mode 4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, default='table_directory',
                        choices=['table_directory'],
                        help='Processing mode (currently only table_directory supported)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing CSV files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output pickle file for embeddings')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to TURL model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows per table')
    parser.add_argument('--max_entities', type=int, default=None,
                        help='Maximum entities (cells) per table; caps rows to preserve columns')
    parser.add_argument('--max_cell_length', type=int, default=10,
                        help='Maximum tokens per cell (original TURL default: 10)')
    parser.add_argument('--max_cell_chars', type=int, default=512,
                        help='Maximum characters per cell before tokenization (default: 512)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N tables (default: 100). Set to 0 to disable.')
    parser.add_argument('--allow_read_errors', action='store_true',
                        help='Allow CSV read errors and skip affected tables (default: fail fast).')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()

    # Normalize max_entities
    if args.max_entities is not None and args.max_entities <= 0:
        args.max_entities = None

    # Find model checkpoint
    if args.checkpoint is None:
        default_paths = [
            'checkpoints/turl/pretrained',
            os.path.join(script_dir, '..', '..', 'checkpoints', 'turl', 'pretrained'),
        ]
        for path in default_paths:
            if os.path.exists(path):
                args.checkpoint = path
                break

    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        raise ValueError("Could not find TURL checkpoint. Please specify --checkpoint")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    # =========================================================================
    # RESUME SUPPORT: Load existing checkpoint if available
    # =========================================================================
    existing_results, processed_tables, resume_checkpoint_path = load_checkpoint_data(args.output_file)

    # Find all CSV files
    all_csv_files = find_csv_files(args.input_dir)
    if args.table_list:
        from trl_bench.utils.table_list import load_table_list
        _table_list = load_table_list(args.table_list)
        all_csv_files = [f for f in all_csv_files if os.path.basename(f) in _table_list]
    if not all_csv_files:
        raise ValueError(f"No CSV files found in: {args.input_dir}")

    total_files = len(all_csv_files)

    # Filter out already-processed files
    if processed_tables:
        csv_files = [f for f in all_csv_files if f not in processed_tables]
        skipped = total_files - len(csv_files)
        print(f"\nResume mode:")
        print(f"  Total files:     {total_files}")
        print(f"  Already done:    {skipped}")
        print(f"  Remaining:       {len(csv_files)}")

        if not csv_files:
            print("\nAll tables already processed. Nothing to do.")
            # Ensure final output exists
            if not Path(args.output_file).exists():
                print(f"Saving final output to {args.output_file}...")
                with open(args.output_file, 'wb') as f:
                    pickle.dump(existing_results, f, protocol=4)
            # Clean up checkpoint
            if resume_checkpoint_path.exists():
                resume_checkpoint_path.unlink()
                print(f"Checkpoint file removed.")
            return
    else:
        csv_files = all_csv_files

    print("=" * 70)
    print("TURL Column Embedding Generator (Mode 4 - Dataset)")
    print("=" * 70)
    print(f"Input:              {args.input_dir} ({len(csv_files)} files to process)")
    print(f"Output:             {args.output_file}")
    print(f"Model checkpoint:   {args.checkpoint}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint interval: {args.checkpoint_interval} tables")
    print("=" * 70)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create dataset and dataloader
    start_time = time.time()
    dataset = TableDirectoryDataset(
        csv_files=csv_files,
        max_rows=args.max_rows,
        max_cell_length=args.max_cell_length,
        max_cell_chars=args.max_cell_chars,
        max_entities=args.max_entities,
        tokenizer=tokenizer,
        fail_on_read_error=not args.allow_read_errors,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # Extract embeddings with checkpointing
    extractor = TURLEmbeddingExtractor(args.checkpoint, args.device)
    results = extractor.extract_embeddings(
        dataloader,
        existing_results=existing_results,
        checkpoint_path=resume_checkpoint_path,
        checkpoint_interval=args.checkpoint_interval
    )

    inference_time = time.time() - start_time

    # =========================================================================
    # Save final results and clean up
    # =========================================================================
    print(f"\nSaving embeddings to {args.output_file}")
    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    # Remove checkpoint file after successful completion
    if resume_checkpoint_path.exists():
        try:
            resume_checkpoint_path.unlink()
            print(f"Checkpoint file removed (processing complete).")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint file: {e}")

    # Summary
    new_tables = len(results) - len(existing_results)
    total_columns = sum(len(r['column_embeddings']) for r in results)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Tables processed:    {len(results)} total ({new_tables} new)")
    print(f"  Total columns:       {total_columns}")
    print(f"  Embedding dimension: {extractor.hidden_size}")
    print(f"  Output format:       Unified v2.0")
    print(f"  Output file:         {args.output_file}")
    print(f"  Processing time:     {inference_time:.2f} seconds")
    print("=" * 70)


if __name__ == '__main__':
    main()
