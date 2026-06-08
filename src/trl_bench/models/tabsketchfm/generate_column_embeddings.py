#!/usr/bin/env python
"""
Generate column embeddings from CSV files using TabSketchFM.

Supports both single file and batch (directory) modes. Model is loaded once.

RESUME SUPPORT
==============
This script supports resuming from interruptions:
- Checkpoint file (.checkpoint.pkl) saved alongside output every N tables
- On restart, automatically detects and loads checkpoint
- Skips already-processed tables
- Checkpoint removed after successful completion

Usage:
    # Single file
    python generate_column_embeddings.py \
        --input /path/to/table.csv \
        --checkpoint checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
        --output embeddings.pkl

    # Batch mode (directory of CSVs)
    python generate_column_embeddings.py \
        --input /path/to/csv_directory/ \
        --checkpoint checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
        --output all_embeddings.pkl

    # With checkpoint interval (for large datasets)
    python generate_column_embeddings.py \
        --input /path/to/csv_directory/ \
        --checkpoint checkpoints/tabsketchfm/epoch=10-step=27786.ckpt \
        --output all_embeddings.pkl \
        --checkpoint_interval 100

Note:
    - This script requires a pretrained TabSketchFM checkpoint (.ckpt file).
    - Finetuned classifier checkpoints are not supported for embedding extraction.
    - --csv is an alias for --input (for backward compatibility).
"""

import os
import sys
import tempfile
import shutil
import pickle
import argparse
import csv
import bz2
import json
import time
import faulthandler
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'tabsketchfm'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tabsketchfm.data_processing.data_prep import prep_data
from tabsketchfm.data_processing.tabular_tokenizer import TableSimilarityTokenizer, fake_tablename_metadata
from tabsketchfm.models.tabsketchfm import TabSketchFM
from transformers import AutoConfig, AutoTokenizer
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
    # The embedding repair adapter previously dropped token_mean when
    # rewriting table_embedding dicts; this ensures those tables are
    # re-processed on the next run.
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
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
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


def _extract_embeddings_from_hidden(bert_tokenizer, hidden_state, input_ids, seq_length):
    """
    Extract embeddings from hidden states.

    Returns:
        table_embedding: Mean-pooled table representation [768]
        col_embeddings: Per-column embeddings {col_idx: [768], ...}
        cls_embedding: CLS token representation [768]
    """
    cls_embedding = hidden_state[0].cpu().numpy()

    special_tokens = {bert_tokenizer.cls_token, bert_tokenizer.sep_token, bert_tokenizer.pad_token}
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    mask = []
    num_sep = 0
    col_states = {}

    for i in range(seq_length):
        if tokens[i] in special_tokens:
            mask.append(False)
            if tokens[i] == bert_tokenizer.sep_token and i != 0:
                num_sep += 1
        else:
            mask.append(True)
            if num_sep not in col_states:
                col_states[num_sep] = []
            col_states[num_sep].append(hidden_state[i])

    # Table embedding: mean of all non-special tokens
    all_states = []
    for states in col_states.values():
        all_states.extend(states)

    if all_states:
        table_embedding = torch.mean(torch.stack(all_states), dim=0).cpu().numpy()
    else:
        table_embedding = cls_embedding

    # Per-column embeddings
    col_embeddings = {}
    for col_idx, states in col_states.items():
        t = torch.stack(states, dim=0)
        col_embeddings[col_idx] = torch.mean(t, dim=0).cpu().numpy()

    return table_embedding, col_embeddings, cls_embedding


def _read_csv_header(csv_path: str) -> list[str] | None:
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                return header
    except Exception:
        return None
    return None


def _align_column_embeddings(col_embeddings, expected_len: int | None):
    """
    Align TabSketchFM column embeddings to CSV columns.

    TabSketchFM tokenization includes a leading "table name" segment that
    becomes col index 0. Real CSV columns start at index 1. When we detect
    that pattern, shift indices down by 1 and drop the metadata column.
    """
    if not isinstance(col_embeddings, dict):
        return col_embeddings
    try:
        int_map = {int(key): val for key, val in col_embeddings.items()}
    except Exception:
        return col_embeddings

    if not int_map:
        return int_map

    if expected_len is None:
        keys = set(int_map)
        if 0 in keys:
            return {k - 1: v for k, v in int_map.items() if k != 0}
        if min(keys) == 1:
            return {k - 1: v for k, v in int_map.items()}
        return int_map

    def score(mapping: dict[int, object]) -> tuple[int, int]:
        in_range = [k for k in mapping if 0 <= k < expected_len]
        out_range = len(mapping) - len(in_range)
        return (len(in_range), -out_range)

    as_is = dict(int_map)
    if 0 in int_map:
        shifted = {k - 1: v for k, v in int_map.items() if k != 0}
    else:
        shifted = {k - 1: v for k, v in int_map.items()}

    as_is_score = score(as_is)
    shifted_score = score(shifted)
    if shifted_score > as_is_score:
        aligned = shifted
    elif shifted_score < as_is_score:
        aligned = as_is
    else:
        keys = set(int_map)
        if 0 in keys and (max(keys) >= expected_len or len(keys) > expected_len):
            aligned = shifted
        elif 0 not in keys and min(keys) == 1 and max(keys) >= expected_len:
            aligned = shifted
        else:
            aligned = as_is

    aligned = {k: v for k, v in aligned.items() if 0 <= k < expected_len}
    return aligned


class TabSketchFMEmbedder:
    """
    TabSketchFM embedder that loads the model once for batch processing.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Load TabSketchFM model and tokenizer.

        Args:
            checkpoint_path: Path to pretrained TabSketchFM checkpoint (.ckpt)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Setup tokenizer
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.max_position_embeddings = 512
        config.task_specific_params = {'hash_input_size': config.hidden_size}
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = TableSimilarityTokenizer(
            tokenizer=self.bert_tokenizer,
            config=config,
            table_metadata_func=fake_tablename_metadata
        )

        # Load model
        print(f"Loading TabSketchFM model: {checkpoint_path}")
        self.model = TabSketchFM.load_from_checkpoint(checkpoint_path, map_location=device)
        self.model = self.model.to(device)
        self.model.eval()

    def encode_csv(self, csv_path: str) -> dict:
        """
        Generate embeddings for a single CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            dict with keys:
                - 'table_embedding': numpy array [768]
                - 'column_embeddings': dict {col_idx: numpy array [768], ...}
                - 'cls_embedding': numpy array [768]
                - 'column_names': list of column names
                - 'table_name': CSV filename (without extension)
        """
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        temp_dir = tempfile.mkdtemp(prefix='tabsketchfm_')

        try:
            # Preprocess CSV
            prep_data(csv_path, temp_dir, metadata=None, dataset_type=None, num_augs=1)

            bz2_files = [f for f in os.listdir(temp_dir) if f.endswith('.json.bz2')]
            if not bz2_files:
                raise RuntimeError(f"Preprocessing failed for {csv_path}")

            with bz2.open(os.path.join(temp_dir, bz2_files[0]), 'rt') as f:
                table_data = json.load(f)

            original_header = _read_csv_header(csv_path)
            if original_header:
                column_names = original_header
            else:
                column_names = list(table_data['columns'].keys())
            table_name = os.path.splitext(os.path.basename(csv_path))[0]

            # Tokenize and forward pass
            tokenized = self.tokenizer.tokenize_function(table_data)
            batch_data = {k: v.unsqueeze(0).to(self.device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = self.model.model.bert(**batch_data, return_dict=True, output_hidden_states=True)

            hidden_state = outputs.last_hidden_state[0]
            input_ids = batch_data['input_ids'][0]
            seq_length = int(batch_data['attention_mask'][0].sum().item())

            raw_table_emb, col_embs, cls_emb = _extract_embeddings_from_hidden(
                self.bert_tokenizer, hidden_state, input_ids, seq_length
            )
            expected_len = len(column_names) if column_names is not None else None
            col_embs = _align_column_embeddings(col_embs, expected_len)

            # Compute table-level embedding variants using aggregation module
            # TabSketchFM native support:
            # - cls_embedding: Yes (BERT CLS token represents table)
            # - table_embedding: No (no native table-level output)
            # - column_mean: Computed via aggregation
            # - token_mean: Mean of non-special content tokens (from _extract_embeddings_from_hidden)
            table_embedding = {
                'cls_embedding': cls_emb,
                'table_embedding': None,  # No native support
                'column_mean': aggregate_embeddings(col_embs, 'mean'),
                'token_mean': raw_table_emb,
            }

            return {
                'table_id': table_name,  # Canonical identifier for downstream lookup
                'table': csv_path,  # Full path for legacy compatibility
                'table_embedding': table_embedding,
                'column_embeddings': col_embs,
                'column_names': column_names,
                'table_name': table_name
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def encode_directory(
        self,
        csv_dir: str,
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
            print(f"Encoding table: {csv_path}", flush=True)
            result = self.encode_csv(csv_path)
            results.append(result)
            tables_since_checkpoint += 1

            # Save checkpoint at regular intervals
            if checkpoint_interval > 0 and checkpoint_path and tables_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, checkpoint_path)
                if show_progress:
                    iterator.set_postfix({'saved': len(results)})
                tables_since_checkpoint = 0

        return results


# Backward-compatible function API
def get_column_embeddings(csv_path: str, checkpoint_path: str, device: str = None) -> dict:
    """
    Generate column embeddings from a single CSV file.

    Note: For batch processing, use TabSketchFMEmbedder class directly.

    Args:
        csv_path: Path to the CSV file
        checkpoint_path: Path to pretrained TabSketchFM checkpoint (.ckpt)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)

    Returns:
        dict with embeddings (see TabSketchFMEmbedder.encode_csv)
    """
    embedder = TabSketchFMEmbedder(checkpoint_path, device)
    return embedder.encode_csv(csv_path)


# Keep old function name for compatibility
find_table_col = _extract_embeddings_from_hidden


def main():
    parser = argparse.ArgumentParser(
        description='Generate column embeddings from CSV file(s) using TabSketchFM'
    )
    parser.add_argument('--input', '--csv', type=str, required=True, dest='input',
                        help='Path to CSV file or directory of CSV files')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained TabSketchFM checkpoint (.ckpt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file (default: auto-generated)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='Save checkpoint every N tables (default: 100). Set to 0 to disable.')
    parser.add_argument('--table_list', type=str, default=None,
                        help='Path to file listing CSV basenames to process (for sharded runs)')

    args = parser.parse_args()
    faulthandler.enable()

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
    embedder = TabSketchFMEmbedder(args.checkpoint, args.device)

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
        print("\n" + "=" * 50)
        print("BATCH EMBEDDING EXTRACTION COMPLETE")
        print("=" * 50)
        print(f"Tables processed: {len(results)} total ({new_tables} new)")
        print(f"Output saved to: {args.output}")
        print(f"Inference time: {inference_time:.2f} seconds")
        print("=" * 50)

    else:
        print(f"Processing file: {args.input}")
        result = embedder.encode_csv(args.input)

        with open(args.output, 'wb') as f:
            pickle.dump(result, f)

        print("\n" + "=" * 50)
        print("EMBEDDING EXTRACTION COMPLETE")
        print("=" * 50)
        print(f"Table: {result['table_name']}")
        print(f"Columns: {len(result['column_embeddings'])}")
        print(f"Column names: {result['column_names']}")
        # Get embedding dimension from column_mean or cls_embedding
        table_emb = result['table_embedding']
        if table_emb.get('column_mean') is not None:
            emb_dim = len(table_emb['column_mean'])
        elif table_emb.get('cls_embedding') is not None:
            emb_dim = len(table_emb['cls_embedding'])
        else:
            emb_dim = 'unknown'
        print(f"Embedding dimension: {emb_dim}")
        print(f"Output saved to: {args.output}")
        print("=" * 50)


if __name__ == '__main__':
    main()
