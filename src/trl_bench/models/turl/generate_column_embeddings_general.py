#!/usr/bin/env python3
"""
Generate Column Embeddings for General CSV Files using TURL Mode 4

This script generates column embeddings from ANY CSV file using only cell content.
No Wikipedia entity linking, no special metadata required.

Mode 4: Uses ONLY cell content text (no headers in embedding, no entity embeddings)
- Input: Any CSV file with headers in the first row
- Output: Column embeddings [num_columns, 768]

Usage:
    # Single CSV file
    python generate_column_embeddings_general.py --csv_file data/my_table.csv

    # Directory of CSV files
    python generate_column_embeddings_general.py --csv_dir data/tables/

    # With custom output
    python generate_column_embeddings_general.py --csv_file data/my_table.csv --output_dir embeddings/
"""

import argparse
import os
import sys
import pickle
import csv
import glob

# Fix distutils issue before importing torch/transformers
import setuptools
import distutils.version

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler

# Fix torch._six compatibility issue
if not hasattr(torch, '_six'):
    import types
    torch._six = types.ModuleType('_six')
    torch._six.string_classes = (str, bytes)

# Add code directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(script_dir, 'code')
sys.path.insert(0, code_dir)

from code.model.configuration import TableConfig
from code.model.model import HybridTableModel
from code.model.transformers import BertTokenizer


class GeneralCSVDataset(Dataset):
    """
    Dataset for general CSV files - no Wikipedia dependencies.

    Simply reads CSV files where:
    - First row = column headers
    - Remaining rows = data cells

    All cells are treated as text to be embedded.
    """

    def __init__(self, csv_files, max_rows=100, max_cell_length=64, tokenizer=None):
        """
        Args:
            csv_files: List of CSV file paths, or single path
            max_rows: Maximum number of rows to process per table (excluding header)
            max_cell_length: Maximum tokens per cell
            tokenizer: BERT tokenizer (will load default if None)
        """
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        self.csv_files = csv_files
        self.max_rows = max_rows
        self.max_cell_length = max_cell_length

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Preprocess all tables
        self.data = []
        self._preprocess()

    def _read_csv(self, csv_path):
        """Read CSV file and return headers and rows"""
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return None, None

        headers = rows[0]
        data_rows = rows[1:self.max_rows + 1]  # Limit rows

        return headers, data_rows

    def _preprocess_table(self, csv_path):
        """
        Preprocess a single CSV table for Mode 4.

        Mode 4 only uses:
        - input_ent_text: Tokenized cell content
        - input_ent_text_length: Length of each cell
        - input_ent_type: Type markers (all set to 4 for regular cells)
        - input_ent_mask: Attention mask (row/column structure)
        - column_entity_mask: Maps cells to columns
        """
        headers, data_rows = self._read_csv(csv_path)

        if headers is None or not data_rows:
            return None

        table_id = os.path.basename(csv_path)
        num_columns = len(headers)

        # Build entity (cell) inputs
        # Each cell becomes an "entity" with just its text
        input_ent_text = []
        input_ent_type = []
        column_en_map = {}  # column_idx -> list of entity indices
        row_en_map = {}     # row_idx -> list of entity indices

        entity_idx = 0
        for row_idx, row in enumerate(data_rows):
            for col_idx, cell_value in enumerate(row):
                if col_idx >= num_columns:
                    continue  # Skip if row has more columns than header

                # Tokenize cell text
                cell_text = str(cell_value).strip()
                if not cell_text:
                    cell_text = "[EMPTY]"  # Handle empty cells

                tokenized = self.tokenizer.encode(
                    cell_text,
                    max_length=self.max_cell_length,
                    add_special_tokens=False
                )
                if not tokenized:
                    tokenized = [self.tokenizer.unk_token_id]

                input_ent_text.append(tokenized)
                input_ent_type.append(4)  # Type 4 = regular cell entity

                # Track column mapping
                if col_idx not in column_en_map:
                    column_en_map[col_idx] = []
                column_en_map[col_idx].append(entity_idx)

                # Track row mapping
                if row_idx not in row_en_map:
                    row_en_map[row_idx] = []
                row_en_map[row_idx].append(entity_idx)

                entity_idx += 1

        num_entities = len(input_ent_text)

        if num_entities == 0:
            return None

        # Create column_entity_mask: [num_columns, num_entities]
        # Maps which entities belong to which column
        column_entity_mask = np.zeros([num_columns, num_entities], dtype=np.float32)
        for col_idx in range(num_columns):
            if col_idx in column_en_map:
                for ent_idx in column_en_map[col_idx]:
                    column_entity_mask[col_idx, ent_idx] = 1.0

        # Create entity-entity attention mask based on row/column co-occurrence
        # Cells in the same row or same column can attend to each other
        ent_ent_mask = np.eye(num_entities, dtype=np.int32)

        # Same column attention
        for col_idx, ent_indices in column_en_map.items():
            for i in ent_indices:
                for j in ent_indices:
                    ent_ent_mask[i, j] = 1

        # Same row attention
        for row_idx, ent_indices in row_en_map.items():
            for i in ent_indices:
                for j in ent_indices:
                    ent_ent_mask[i, j] = 1

        # Pad entity text to same length
        input_ent_cell_length = [len(x) for x in input_ent_text]
        max_cell_len = max(input_ent_cell_length)
        input_ent_text_padded = np.zeros([num_entities, max_cell_len], dtype=np.int64)
        for i, tokens in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(tokens)] = tokens

        return {
            'table_id': table_id,
            'headers': headers,
            'num_columns': num_columns,
            'num_entities': num_entities,
            'input_ent_text': input_ent_text_padded,
            'input_ent_text_length': np.array(input_ent_cell_length, dtype=np.int64),
            'input_ent_type': np.array(input_ent_type, dtype=np.int64),
            'input_ent_mask': ent_ent_mask,
            'column_entity_mask': column_entity_mask,
        }

    def _preprocess(self):
        """Preprocess all CSV files"""
        print(f"Preprocessing {len(self.csv_files)} CSV files...")

        for csv_path in tqdm(self.csv_files, desc="Processing CSV files"):
            processed = self._preprocess_table(csv_path)
            if processed is not None:
                self.data.append(processed)

        print(f"Successfully preprocessed {len(self.data)} tables")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class GeneralCSVLoader(DataLoader):
    """DataLoader for general CSV dataset"""

    def __init__(self, dataset, batch_size=1):

        def collate_fn(batch):
            """Collate function that handles variable-sized inputs"""

            table_ids = [item['table_id'] for item in batch]
            headers_list = [item['headers'] for item in batch]
            num_columns_list = [item['num_columns'] for item in batch]

            # Find max dimensions for padding
            max_entities = max(item['num_entities'] for item in batch)
            max_cell_len = max(item['input_ent_text'].shape[1] for item in batch)
            max_columns = max(item['num_columns'] for item in batch)
            batch_size = len(batch)

            # Pad input_ent_text: [batch, max_entities, max_cell_len]
            padded_ent_text = np.zeros((batch_size, max_entities, max_cell_len), dtype=np.int64)
            for i, item in enumerate(batch):
                n_ent = item['num_entities']
                cell_len = item['input_ent_text'].shape[1]
                padded_ent_text[i, :n_ent, :cell_len] = item['input_ent_text']

            # Pad input_ent_text_length: [batch, max_entities]
            padded_ent_length = np.ones((batch_size, max_entities), dtype=np.int64)  # Default 1 to avoid div by 0
            for i, item in enumerate(batch):
                n_ent = item['num_entities']
                padded_ent_length[i, :n_ent] = item['input_ent_text_length']

            # Pad input_ent_type: [batch, max_entities]
            padded_ent_type = np.zeros((batch_size, max_entities), dtype=np.int64)
            for i, item in enumerate(batch):
                n_ent = item['num_entities']
                padded_ent_type[i, :n_ent] = item['input_ent_type']

            # Pad input_ent_mask: [batch, max_entities, max_entities]
            padded_ent_mask = np.zeros((batch_size, max_entities, max_entities), dtype=np.int64)
            for i, item in enumerate(batch):
                n_ent = item['num_entities']
                padded_ent_mask[i, :n_ent, :n_ent] = item['input_ent_mask']

            # Pad column_entity_mask: [batch, max_columns, max_entities]
            padded_col_mask = np.zeros((batch_size, max_columns, max_entities), dtype=np.float32)
            for i, item in enumerate(batch):
                n_col = item['num_columns']
                n_ent = item['num_entities']
                padded_col_mask[i, :n_col, :n_ent] = item['column_entity_mask']
                # For padded columns, point to first entity to avoid NaN (will be masked out later)
                if n_col < max_columns:
                    padded_col_mask[i, n_col:, 0] = 1.0

            return {
                'table_ids': table_ids,
                'headers': headers_list,
                'num_columns': num_columns_list,
                'input_ent_text': torch.LongTensor(padded_ent_text),
                'input_ent_text_length': torch.LongTensor(padded_ent_length),
                'input_ent_type': torch.LongTensor(padded_ent_type),
                'input_ent_mask': torch.LongTensor(padded_ent_mask),
                'column_entity_mask': torch.FloatTensor(padded_col_mask),
            }

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False
        )


class Mode4EmbeddingExtractor:
    """Extract column embeddings from cell content using frozen TURL model"""

    def __init__(self, model_path, device='cuda'):
        self.device = device

        print(f"Loading pretrained model from {model_path}")

        # Load configuration
        config = TableConfig.from_pretrained(model_path)
        config.output_attentions = False
        config.output_hidden_states = False

        # Load model (just the table encoder)
        self.model = HybridTableModel(config, is_simple=True)

        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load only the table encoder weights
        table_state_dict = {
            k.replace('table.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('table.')
        }
        self.model.load_state_dict(table_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = config.hidden_size
        print(f"Model loaded on {device} (hidden_size={self.hidden_size})")

    def extract_embeddings(self, dataloader):
        """
        Extract column embeddings from cell content for all tables.

        Mode 4: Uses ONLY input_ent_text (cell content)
        Ignores: headers, metadata, pre-trained entity embeddings

        Returns:
            List of dicts, each containing:
                - table_id: str
                - headers: list of column names
                - embeddings: np.array [num_columns, hidden_size]
        """
        results = []

        print("\nExtracting Mode 4 embeddings (cell content only)...")
        print("  Using: Cell text content")
        print("  Ignoring: Headers (for embedding), metadata, entity embeddings")
        print()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing tables"):
                table_ids = batch['table_ids']
                headers_list = batch['headers']
                num_columns_list = batch['num_columns']

                input_ent_text = batch['input_ent_text'].to(self.device)
                input_ent_text_length = batch['input_ent_text_length'].to(self.device)
                input_ent_type = batch['input_ent_type'].to(self.device)
                input_ent_mask = batch['input_ent_mask'].to(self.device)
                column_entity_mask = batch['column_entity_mask'].to(self.device)

                # MODE 4: Only use entity (cell) stream
                # All token inputs = None, entity embedding = None
                tok_outputs, ent_outputs, _ = self.model(
                    None, None, None, None,           # No token inputs
                    input_ent_text,                   # Cell text tokens
                    input_ent_text_length,            # Cell text lengths
                    None,                             # No mask type
                    None,                             # No entity embeddings!
                    input_ent_type,                   # Entity types
                    input_ent_mask,                   # Attention mask
                    None                              # No candidates
                )

                # Extract entity sequence output
                ent_sequence_output = ent_outputs[0]  # [batch, num_entities, hidden_size]

                # Aggregate entities to column level
                # column_entity_mask: [batch, num_columns, num_entities]
                # ent_sequence_output: [batch, num_entities, hidden_size]
                ent_col_output = torch.matmul(column_entity_mask, ent_sequence_output)

                # Normalize by number of entities per column
                ent_col_output /= column_entity_mask.sum(dim=-1, keepdim=True).clamp(1.0, 9999.0)

                # Move to CPU
                ent_col_output_cpu = ent_col_output.cpu().numpy()

                # Split batch and store results
                for i in range(len(table_ids)):
                    num_cols = num_columns_list[i]
                    results.append({
                        'table_id': table_ids[i],
                        'headers': headers_list[i],
                        'embeddings': ent_col_output_cpu[i, :num_cols, :].copy()  # [num_columns, hidden_size]
                    })

        print(f"Extracted embeddings for {len(results)} tables")
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
        description='Generate column embeddings for general CSV files using TURL Mode 4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single CSV file
    python generate_column_embeddings_general.py --csv_file data/my_table.csv

    # Directory of CSV files
    python generate_column_embeddings_general.py --csv_dir data/tables/

    # Custom model and output
    python generate_column_embeddings_general.py --csv_file data/table.csv \\
        --pretrained_model /path/to/model --output_dir /path/to/output
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv_file', type=str,
                            help='Path to a single CSV file')
    input_group.add_argument('--csv_dir', type=str,
                            help='Path to directory containing CSV files')

    # Model and output options
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='Path to pretrained TURL model checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='column_embeddings',
                        help='Output directory for embeddings')
    parser.add_argument('--output_file', type=str,
                        default=None,
                        help='Output filename (default: embeddings.pkl)')

    # Processing options
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows per table (excluding header)')
    parser.add_argument('--max_cell_length', type=int, default=10,
                        help='Maximum tokens per cell (original TURL default: 10)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Resolve model path
    if args.pretrained_model is None:
        # Try default locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_paths = [
            os.path.join(parent_dir, 'checkpoint', 'pretrained'),
            os.path.join(script_dir, 'checkpoint', 'pretrained'),
            os.path.join(parent_dir, 'turl', 'checkpoint', 'pretrained'),
        ]
        for path in default_paths:
            if os.path.exists(path):
                args.pretrained_model = path
                break
        if args.pretrained_model is None:
            raise ValueError("Could not find pretrained model. Please specify --pretrained_model")

    # Find CSV files
    csv_path = args.csv_file or args.csv_dir
    csv_files = find_csv_files(csv_path)

    if not csv_files:
        raise ValueError(f"No CSV files found in: {csv_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("TURL Column Embedding Generator (General CSV - Mode 4)")
    print("=" * 80)
    print(f"Mode: Cell content only (no headers in embedding, no entity linking)")
    print()
    print(f"Input: {len(csv_files)} CSV file(s)")
    print(f"Model: {args.pretrained_model}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Max rows per table: {args.max_rows}")
    print(f"Max tokens per cell: {args.max_cell_length}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create dataset
    print("Loading and preprocessing CSV files...")
    dataset = GeneralCSVDataset(
        csv_files=csv_files,
        max_rows=args.max_rows,
        max_cell_length=args.max_cell_length,
        tokenizer=tokenizer
    )

    if len(dataset) == 0:
        raise ValueError("No valid tables found in CSV files")

    # Create dataloader
    dataloader = GeneralCSVLoader(dataset, batch_size=args.batch_size)

    # Initialize extractor and extract embeddings
    extractor = Mode4EmbeddingExtractor(args.pretrained_model, args.device)
    results = extractor.extract_embeddings(dataloader)

    # Save results
    output_filename = args.output_file or 'embeddings.pkl'
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"\nSaving embeddings to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    # Also save a summary
    summary = {
        'num_tables': len(results),
        'hidden_size': extractor.hidden_size,
        'mode': 4,
        'description': 'Cell content only embeddings (Mode 4)',
        'tables': [
            {
                'table_id': r['table_id'],
                'num_columns': len(r['headers']),
                'headers': r['headers']
            }
            for r in results
        ]
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"  Tables processed: {len(results)}")
    print(f"  Embedding dimension: {extractor.hidden_size}")
    print(f"  Output file: {output_path}")
    print(f"  Summary file: {summary_path}")
    print()
    print("Output format (embeddings.pkl):")
    print("  List of dicts, each containing:")
    print("    - 'table_id': str (filename)")
    print("    - 'headers': list of column names")
    print("    - 'embeddings': np.array [num_columns, 768]")
    print()

    # Print first table as example
    if results:
        print("Example (first table):")
        print(f"  Table: {results[0]['table_id']}")
        print(f"  Headers: {results[0]['headers']}")
        print(f"  Embeddings shape: {results[0]['embeddings'].shape}")


if __name__ == '__main__':
    main()
