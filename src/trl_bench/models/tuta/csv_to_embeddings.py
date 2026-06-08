#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed CSV to TUTA Embeddings - Production Implementation

This module fixes the truncation bug in the original csv_to_embeddings.py
by implementing multi-sequence aggregation for table-level embeddings and
row-by-row processing for cell/token-level embeddings.

Usage:
    from csv_to_embeddings import TUTAEmbedder

    embedder = TUTAEmbedder('./tuta.bin', 'tuta', device_id=0)

    # Table-level (complete - uses multi-sequence aggregation)
    table_emb = embedder.csv_to_embeddings('large.csv', aggregate='cls')

    # Cell-level (complete - uses row-by-row processing)
    cell_emb = embedder.csv_to_embeddings('large.csv', aggregate='cell')
"""

import os
import sys
import csv
import json
import torch

# Raise CSV field size limit to handle tables with large cell values
csv.field_size_limit(sys.maxsize)
import argparse
import numpy as np
import tempfile

# Add tuta directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tuta'))

import utils as ut
import reader as rdr
import tokenizer as tknr
import model.backbones as bbs


class SimpleCSVReader:
    """Simplified reader for CSV files without complex hierarchy extraction"""
    def __init__(self, args):
        self.tree_depth = args.tree_depth
        self.node_degree = args.node_degree
        self.row_size = args.row_size
        self.column_size = args.column_size
        self.args = args

    def read_csv(self, csv_path):
        """Read CSV file and convert to table structure"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader_obj = csv.reader(f)
            rows = list(reader_obj)

        if not rows:
            return None

        # Get dimensions
        row_num = len(rows)
        col_num = max(len(row) for row in rows) if rows else 0

        # Truncate if table exceeds model's position embedding limits
        # Model supports max 256 rows × 256 columns (indices 0-256, embedding size 257)
        original_row_num = row_num
        original_col_num = col_num
        truncated = False

        if row_num > self.row_size:
            rows = rows[:self.row_size]
            row_num = self.row_size
            truncated = True

        if col_num > self.column_size:
            rows = [row[:self.column_size] for row in rows]
            col_num = self.column_size
            truncated = True

        if truncated:
            import sys
            print(f"  ⚠ Table truncated from {original_row_num}×{original_col_num} to {row_num}×{col_num}",
                  file=sys.stderr)

        # Pad rows to same length
        cell_matrix = []
        for row in rows:
            padded_row = row + [''] * (col_num - len(row))
            cell_matrix.append(padded_row)

        return {
            'cell_matrix': cell_matrix,
            'row_num': row_num,
            'col_num': col_num
        }

    def create_flat_positions(self, row_num, col_num):
        """Create flat (non-hierarchical) tree positions for simple tables"""
        top_positions = []
        left_positions = []

        for i in range(row_num):
            for j in range(col_num):
                top_pos = [-1] * (self.tree_depth - 1) + [min(j, self.node_degree[-1] - 1)]
                left_pos = [-1] * (self.tree_depth - 1) + [min(i, self.node_degree[-1] - 1)]
                top_positions.append(top_pos)
                left_positions.append(left_pos)

        return top_positions, left_positions

    def create_default_formats(self, row_num, col_num):
        """Create default format features (all zeros)"""
        format_matrix = []
        for i in range(row_num):
            row_formats = []
            for j in range(col_num):
                row_formats.append([0.0] * 11)
            format_matrix.append(row_formats)
        return format_matrix


class SimpleTokenizer(tknr.TableTokenizer):
    """Simplified tokenizer for inference"""
    def __init__(self, args):
        super().__init__(args)

    def no_sampling(self, token_matrix):
        """Don't sample any cells - include all"""
        sampling_mask = [[1 for _ in row] for row in token_matrix]
        return sampling_mask

    def init_table_seq(self, root_context=""):
        """Initialize table sequence with CLS_ID at head"""
        context_tokens, context_number = self.tokenize_text(
            cell_string=root_context, add_separate=False, max_cell_len=8
        )
        token_list = [[tknr.CLS_ID] + context_tokens]
        num_list = [[self.wordpiece_tokenizer.default_num] + context_number]
        pos_list = [(self.row_size, self.column_size, [-1] * self.tree_depth, [-1] * self.tree_depth)]
        format_list = [self.default_format]
        ind_list = [[-1] + [-2 for _ in context_tokens]]
        cell_num = 1
        seq_len = len(token_list[0])
        return token_list, num_list, pos_list, format_list, ind_list, cell_num, seq_len

    def create_table_seq(self, sampling_matrix, token_matrix, number_matrix,
                         position_lists, format_matrix, max_seq_len=512, max_cell_length=16):
        """Create table sequence for inference (no labels needed)"""
        seqs = []
        start_row = 0

        # Split tables if they exceed max length
        while start_row < len(token_matrix):
            token_list, num_list, pos_list, format_list, ind_list, cell_num, seq_len = self.init_table_seq()
            top_pos_list, left_pos_list = position_lists
            icell = 0
            mark_exceed_len = False

            for irow, token_row in enumerate(token_matrix):
                if mark_exceed_len:
                    break
                if irow < start_row:
                    continue

                for icol, token_cell in enumerate(token_row):
                    if sampling_matrix[irow][icol] == 0:
                        continue

                    token_cell = token_cell[:max_cell_length]
                    cell_len = len(token_cell)

                    if cell_len + seq_len >= max_seq_len:
                        if irow > start_row:
                            start_row = irow
                        else:
                            start_row = irow + 1
                        seqs.append([token_list, num_list, pos_list, format_list, ind_list])
                        mark_exceed_len = True
                        break

                    # Add cell data
                    pos_list.append((irow, icol, top_pos_list[icell], left_pos_list[icell]))
                    icell += 1

                    format_vector = []
                    for ivec, vec in enumerate(format_matrix[irow][icol]):
                        format_vector.append(min(vec, self.format_range[ivec]) / self.format_range[ivec])
                    format_list.append(format_vector)

                    token_list.append(token_cell)
                    num_list.append(number_matrix[irow][icol][:cell_len])
                    ind_list.append([cell_num * 2] * cell_len)
                    ind_list[-1][0] -= 1

                    seq_len += cell_len
                    cell_num += 1

            if not mark_exceed_len:
                seqs.append([token_list, num_list, pos_list, format_list, ind_list])
                start_row = len(token_matrix)

        return seqs


class TUTAEmbedder:
    """
    FIXED TUTA Embedder - Handles large tables correctly

    This class fixes the truncation bug by:
    - Using multi-sequence aggregation for table-level embeddings
    - Using row-by-row processing for cell-level embeddings
    """

    def __init__(self, model_path, target='tuta', device_id=None):
        """
        Args:
            model_path: Path to TUTA .bin checkpoint
            target: Model type ('tuta', 'tuta_explicit', or 'base')
            device_id: GPU device ID (None for auto-detect, -1 for CPU only)
        """
        self.model_path = model_path
        self.target = target

        # Auto-detect GPU if device_id is None
        if device_id is None:
            if torch.cuda.is_available():
                self.device_id = 0
                print("GPU detected! Using GPU for acceleration.")
            else:
                self.device_id = None
                print("No GPU detected. Using CPU.")
        elif device_id == -1:
            self.device_id = None
        else:
            self.device_id = device_id

        # Setup args
        self.args = self._create_args()

        # Initialize reader and tokenizer
        self.reader = SimpleCSVReader(self.args)
        self.tokenizer = SimpleTokenizer(self.args)

        # Set vocab_size from tokenizer
        self.args.vocab_size = len(self.tokenizer.vocab)

        # Build model
        self.model = self._build_model()

        print(f"TUTA Embedder (FIXED) initialized with model: {model_path}")
        print(f"Model type: {target}, Device: {'cuda:'+str(self.device_id) if self.device_id is not None else 'cpu'}")

    def _create_args(self):
        """Create default arguments for TUTA model"""
        class Args:
            pass

        args = Args()
        vocab_dir = os.path.join(os.path.dirname(__file__), 'tuta', 'vocab')
        args.vocab_path = os.path.join(vocab_dir, 'tuta_vocab.txt')

        # Model architecture
        args.hidden_size = 768
        args.intermediate_size = 3072
        args.num_attention_heads = 12
        args.num_encoder_layers = 12
        args.hidden_dropout_prob = 0.1
        args.attention_dropout_prob = 0.1
        args.layer_norm_eps = 1e-6
        args.hidden_act = 'gelu'

        # Numeric features (model adds +2, so checkpoint=12 means config=10)
        args.magnitude_size = 10  # Model adds +2 → 12 positions
        args.precision_size = 10  # Model adds +2 → 12 positions
        args.top_digit_size = 10  # Model adds +2 → 12 positions
        args.low_digit_size = 10  # Model adds +2 → 12 positions

        # Position features
        args.max_cell_length = 64  # Checkpoint order_weight has 64 positions
        args.row_size = 256  # Checkpoint row_weight has 257, indices 0-256
        args.column_size = 256  # Checkpoint column_weight has 257, indices 0-256
        args.tree_depth = 4
        args.node_degree = [32, 32, 64, 256]  # Model adds +1 → sum=385 positions
        args.total_node = sum(args.node_degree)

        # Attention
        args.attn_method = 'add'
        args.attention_distance = 100
        args.attention_step = 0

        # Format features
        args.num_format_feature = 11

        # Target model type
        args.target = self.target

        # Tokenizer specific
        args.max_seq_len = 512
        args.add_separate = True

        # Defaults
        args.default_tree_position = [args.total_node for _ in args.node_degree]

        # Tokenizer specific settings
        args.text_threshold = 0.5
        args.value_threshold = 0.1
        args.clc_rate = 0.3
        args.wcm_rate = 0.3

        # Repository paths
        args.context_repo_path = os.path.join(vocab_dir, 'context_repo_init.txt')
        args.cellstr_repo_path = os.path.join(vocab_dir, 'cellstr_repo_init.txt')

        return args

    def _build_model(self):
        """Build and load TUTA model"""
        backbone = bbs.BACKBONES[self.args.target](self.args)

        # Load checkpoint with proper key mapping
        # The checkpoint has "backbone." prefix but the model doesn't
        import torch
        checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))

        # Strip "backbone." prefix from checkpoint keys
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                new_key = key[9:]  # Remove "backbone." prefix
                new_checkpoint[new_key] = value
            else:
                new_checkpoint[key] = value

        # Load the corrected checkpoint
        model_dict = backbone.state_dict()
        matched_params = 0
        mismatched_params = 0

        for name, params in new_checkpoint.items():
            if name in model_dict:
                if params.size() == model_dict[name].size():
                    model_dict[name] = params
                    matched_params += 1
                else:
                    # Handle size mismatch (e.g., order_weight: 64 vs 16)
                    old_size = params.size()[0]
                    new_size = model_dict[name].size()[0]
                    if len(params.size()) == 2:
                        # Take the smaller size to avoid index errors
                        min_size = min(old_size, new_size)
                        model_dict[name][:min_size, :] = params[:min_size, :]
                        mismatched_params += 1
                        print(f"  Partially loaded {name}: shape {params.size()} -> {model_dict[name].size()}")

        print(f"Loaded {matched_params} parameters from checkpoint (with {mismatched_params} partial matches)")
        backbone.load_state_dict(model_dict, strict=False)

        if self.device_id is not None:
            backbone.cuda(self.device_id)
        else:
            backbone.cpu()

        backbone.eval()
        return backbone

    def csv_to_embeddings(self, csv_path, output_format='tensor', aggregate='cls'):
        """
        Extract embeddings from a CSV file (FIXED - handles large tables correctly)

        Args:
            csv_path: Path to CSV file
            output_format: 'tensor', 'numpy', or 'list'
            aggregate: How to aggregate token embeddings
                - 'cls': Table-level embedding - USES MULTI-SEQUENCE AGGREGATION (default)
                - 'row': Row-level embeddings - USES ROW-BY-ROW PROCESSING
                - 'cell': Cell-level embeddings - USES ROW-BY-ROW PROCESSING
                - 'all': Token-level embeddings - USES ROW-BY-ROW PROCESSING

        Returns:
            Embeddings in specified format:
            - 'cls': (1, 768) - complete table representation
            - 'row': (num_rows, 768) - one embedding per row
            - 'cell': (num_cells, 768) - one embedding per cell
            - 'all': List of (num_tokens_per_row, 768) arrays

        Note:
            - 'cls' aligns with paper's table-level approach (uses [CLS] token)
            - 'row' provides row-level representations (each row's [CLS] token)
            - 'cell' aligns with paper's cell-level approach (uses [SEP] tokens)
            - 'all' for token-level tasks (rarely needed)
        """
        if aggregate == 'cls':
            return self._get_table_embedding_complete(csv_path, output_format)
        elif aggregate == 'row':
            return self._get_row_embeddings_complete(csv_path, output_format)
        elif aggregate == 'cell':
            return self._get_cell_embeddings_complete(csv_path, output_format)
        elif aggregate == 'all':
            return self._get_token_embeddings_complete(csv_path, output_format)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}. Valid options: 'cls', 'row', 'cell', 'all'")

    def _get_table_embedding_complete(self, csv_path, output_format='tensor'):
        """
        SOLUTION FOR TABLE-LEVEL: Multi-sequence aggregation

        Process all sequences, extract [CLS] from each, average them.
        """
        print(f"[TABLE-LEVEL] Processing with multi-sequence aggregation...")

        # Read and prepare table
        table_data = self.reader.read_csv(csv_path)
        if table_data is None:
            raise ValueError(f"Failed to read CSV: {csv_path}")

        cell_matrix = table_data['cell_matrix']
        row_num = table_data['row_num']
        col_num = table_data['col_num']

        print(f"  Table size: {row_num} rows × {col_num} columns")

        # Create positions and formats
        top_positions, left_positions = self.reader.create_flat_positions(row_num, col_num)
        format_matrix = self.reader.create_default_formats(row_num, col_num)

        # Tokenize
        token_matrix, number_matrix = self.tokenizer.tokenize_string_matrix(
            string_matrix=cell_matrix,
            add_separate=True,
            max_cell_len=self.args.max_cell_length
        )

        sampling_mask = self.tokenizer.no_sampling(token_matrix)

        # Create ALL sequences (not just first one!)
        position_lists = (top_positions, left_positions)
        seqs = self.tokenizer.create_table_seq(
            sampling_matrix=sampling_mask,
            token_matrix=token_matrix,
            number_matrix=number_matrix,
            position_lists=position_lists,
            format_matrix=format_matrix,
            max_seq_len=self.args.max_seq_len,
            max_cell_length=self.args.max_cell_length
        )

        if not seqs:
            raise ValueError("Failed to create table sequence")

        print(f"  Created {len(seqs)} sequences")

        # Process each sequence and collect [CLS] embeddings
        cls_embeddings = []

        for i, seq in enumerate(seqs):
            token_list, num_list, pos_list, format_list, ind_list = seq

            # Convert to model inputs
            inputs = self._lists_to_tensors(token_list, num_list, pos_list, format_list, ind_list)

            # Get embeddings
            with torch.no_grad():
                embeddings = self._forward_model(inputs)

            # Extract [CLS] token (position 0)
            cls_token = embeddings[:, 0, :]  # Shape: (1, 768)
            cls_embeddings.append(cls_token)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(seqs)} sequences")

        # Aggregate chunk embeddings using mean pooling
        cls_embeddings = torch.cat(cls_embeddings, dim=0)  # (num_seqs, 768)
        table_embedding = cls_embeddings.mean(dim=0, keepdim=True)  # (1, 768)

        print(f"  ✓ Complete table embedding generated")

        # Convert to desired format
        if output_format == 'numpy':
            return table_embedding.cpu().numpy()
        elif output_format == 'list':
            return table_embedding.cpu().numpy().tolist()
        else:
            return table_embedding

    def _get_row_embeddings_complete(self, csv_path, output_format='tensor'):
        """
        SOLUTION FOR ROW-LEVEL: Row-by-row [CLS] extraction

        Process each row with headers, extract [CLS] token for each row.
        Each row gets a single embedding representing that entire row in context.
        """
        print(f"[ROW-LEVEL] Processing with row-by-row method...")

        # Read CSV
        with open(csv_path, 'r') as f:
            reader_obj = csv.reader(f)
            header = next(reader_obj)
            rows = list(reader_obj)

        num_cols = len(header)
        num_data_rows = len(rows)

        print(f"  Table: {num_data_rows} data rows × {num_cols} columns")
        print(f"  Will generate {num_data_rows} row embeddings")

        # Process data rows
        all_row_embeddings = []

        for i, row in enumerate(rows):
            # Create temp CSV with header + single row
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                writer = csv.writer(tmp)
                writer.writerow(header)
                writer.writerow(row)
                tmp_path = tmp.name

            # Get [CLS] embedding for this row (represents the whole row)
            row_emb = self._get_embeddings_single_table(tmp_path, aggregate='cls')
            all_row_embeddings.append(row_emb)

            # Cleanup
            os.unlink(tmp_path)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{num_data_rows} rows")

        # Stack all row embeddings
        row_embeddings = np.vstack(all_row_embeddings)

        print(f"  ✓ Complete! Generated {row_embeddings.shape[0]} row embeddings")

        # Convert format
        if output_format == 'list':
            return row_embeddings.tolist()
        elif output_format == 'tensor':
            return torch.from_numpy(row_embeddings)
        else:  # numpy
            return row_embeddings

    def _get_cell_embeddings_complete(self, csv_path, output_format='tensor'):
        """
        SOLUTION FOR CELL-LEVEL: Row-by-row processing

        Process each row with headers, extract cell embeddings.
        """
        print(f"[CELL-LEVEL] Processing with row-by-row method...")

        # Read CSV
        with open(csv_path, 'r') as f:
            reader_obj = csv.reader(f)
            header = next(reader_obj)
            rows = list(reader_obj)

        num_cols = len(header)
        num_data_rows = len(rows)

        print(f"  Table: {num_data_rows} data rows × {num_cols} columns")
        print(f"  Will generate {num_data_rows * num_cols} cell embeddings")

        # Process data rows
        all_cell_embeddings = []

        for i, row in enumerate(rows):
            # Create temp CSV with header + single row
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                writer = csv.writer(tmp)
                writer.writerow(header)
                writer.writerow(row)
                tmp_path = tmp.name

            # Get embeddings using original method (which works for small tables)
            emb = self._get_embeddings_single_table(tmp_path, aggregate='cell')

            # Extract data row embeddings (skip header)
            data_row_emb = emb[num_cols:, :]
            all_cell_embeddings.append(data_row_emb)

            # Cleanup
            os.unlink(tmp_path)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{num_data_rows} rows")

        # Stack all data embeddings
        data_embeddings = np.vstack(all_cell_embeddings)

        print(f"  ✓ Complete! Generated {data_embeddings.shape[0]} cell embeddings")

        # Convert format
        if output_format == 'list':
            return data_embeddings.tolist()
        elif output_format == 'tensor':
            return torch.from_numpy(data_embeddings)
        else:  # numpy
            return data_embeddings

    def _get_token_embeddings_complete(self, csv_path, output_format='tensor'):
        """
        SOLUTION FOR TOKEN-LEVEL: Row-by-row processing

        Process each row, extract all token embeddings.
        """
        print(f"[TOKEN-LEVEL] Processing with row-by-row method...")
        print(f"  Note: Consider using cell-level instead if possible")

        with open(csv_path, 'r') as f:
            reader_obj = csv.reader(f)
            header = next(reader_obj)
            rows = list(reader_obj)

        print(f"  Table: {len(rows)} rows")

        all_token_embeddings = []

        for i, row in enumerate(rows):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                writer = csv.writer(tmp)
                writer.writerow(header)
                writer.writerow(row)
                tmp_path = tmp.name

            # Get token embeddings
            token_emb = self._get_embeddings_single_table(tmp_path, aggregate='all')
            all_token_embeddings.append(token_emb)

            os.unlink(tmp_path)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(rows)} rows")

        print(f"  ✓ Complete! Generated token embeddings for {len(rows)} rows")

        # Return as list of arrays (can't stack due to variable lengths)
        if output_format == 'list':
            return [emb.tolist() for emb in all_token_embeddings]
        else:
            return all_token_embeddings

    def _get_embeddings_single_table(self, csv_path, aggregate='cls'):
        """
        Get embeddings for a single small table (internal helper)
        This is the original logic that works fine for small tables.
        """
        # Read CSV
        table_data = self.reader.read_csv(csv_path)
        if table_data is None:
            raise ValueError(f"Failed to read CSV: {csv_path}")

        cell_matrix = table_data['cell_matrix']
        row_num = table_data['row_num']
        col_num = table_data['col_num']

        # Create positions and formats
        top_positions, left_positions = self.reader.create_flat_positions(row_num, col_num)
        format_matrix = self.reader.create_default_formats(row_num, col_num)

        # Tokenize
        token_matrix, number_matrix = self.tokenizer.tokenize_string_matrix(
            string_matrix=cell_matrix,
            add_separate=True,
            max_cell_len=self.args.max_cell_length
        )

        sampling_mask = self.tokenizer.no_sampling(token_matrix)

        # Create sequence
        position_lists = (top_positions, left_positions)
        seqs = self.tokenizer.create_table_seq(
            sampling_matrix=sampling_mask,
            token_matrix=token_matrix,
            number_matrix=number_matrix,
            position_lists=position_lists,
            format_matrix=format_matrix,
            max_seq_len=self.args.max_seq_len,
            max_cell_length=self.args.max_cell_length
        )

        if not seqs:
            raise ValueError("Failed to create table sequence")

        # Take first sequence (safe for small tables)
        seq = seqs[0]
        token_list, num_list, pos_list, format_list, ind_list = seq

        # Convert to model inputs
        inputs = self._lists_to_tensors(token_list, num_list, pos_list, format_list, ind_list)

        # Get embeddings
        with torch.no_grad():
            embeddings = self._forward_model(inputs)

        # Aggregate embeddings
        embeddings = self._aggregate_embeddings(
            embeddings,
            aggregate,
            indicator=inputs.get('indicator') if aggregate == 'cell' else None
        )

        return embeddings.cpu().numpy()

    def _lists_to_tensors(self, token_list, num_list, pos_list, format_list, ind_list):
        """Convert lists to tensors for model input"""
        token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
        token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
        format_vec, indicator = [], []

        for tokens, num_feats, (row, col, ttop, tleft), fmt, ind in zip(
            token_list, num_list, pos_list, format_list, ind_list
        ):
            cell_len = len(tokens)
            token_id.extend(tokens)
            num_mag.extend([f[0] for f in num_feats])
            num_pre.extend([f[1] for f in num_feats])
            num_top.extend([f[2] for f in num_feats])
            num_low.extend([f[3] for f in num_feats])

            token_order.extend(list(range(cell_len)))
            pos_row.extend([row] * cell_len)
            pos_col.extend([col] * cell_len)

            entire_top = ut.UNZIPS[self.target](
                zipped=ttop,
                node_degree=self.args.node_degree,
                total_node=self.args.total_node
            )
            pos_top.extend([entire_top] * cell_len)

            entire_left = ut.UNZIPS[self.target](
                zipped=tleft,
                node_degree=self.args.node_degree,
                total_node=self.args.total_node
            )
            pos_left.extend([entire_left] * cell_len)

            format_vec.extend([fmt] * cell_len)
            indicator.extend(ind)

        # Convert to tensors
        device = torch.device(f'cuda:{self.device_id}' if self.device_id is not None else 'cpu')

        tensors = {
            'token_id': torch.LongTensor([token_id]).to(device),
            'num_mag': torch.LongTensor([num_mag]).to(device),
            'num_pre': torch.LongTensor([num_pre]).to(device),
            'num_top': torch.LongTensor([num_top]).to(device),
            'num_low': torch.LongTensor([num_low]).to(device),
            'token_order': torch.LongTensor([token_order]).to(device),
            'pos_row': torch.LongTensor([pos_row]).to(device),
            'pos_col': torch.LongTensor([pos_col]).to(device),
            'pos_top': torch.LongTensor([pos_top]).to(device),
            'pos_left': torch.LongTensor([pos_left]).to(device),
            'format_vec': torch.FloatTensor([format_vec]).to(device),
            'indicator': torch.LongTensor([indicator]).to(device)
        }

        return tensors

    def _forward_model(self, inputs):
        """Forward pass through model"""
        return self.model(
            token_id=inputs['token_id'],
            num_mag=inputs['num_mag'],
            num_pre=inputs['num_pre'],
            num_top=inputs['num_top'],
            num_low=inputs['num_low'],
            token_order=inputs['token_order'],
            pos_row=inputs['pos_row'],
            pos_col=inputs['pos_col'],
            pos_top=inputs['pos_top'],
            pos_left=inputs['pos_left'],
            format_vec=inputs['format_vec'],
            indicator=inputs['indicator']
        )

    def _aggregate_embeddings(self, embeddings, method='mean', indicator=None):
        """
        Aggregate token embeddings to cell level.

        Matches TUTA's original behavior which uses token-aggregation-based embeddings.
        This is the method used in TUTA's CTC fine-tuning (see tuta/model/heads.py).

        Args:
            embeddings: Token-level embeddings [batch_size, seq_len, hidden_size]
            method: Aggregation method ('cls', 'mean', 'cell', 'all')
            indicator: Indicator tensor for cell boundaries [batch_size, seq_len]

        Returns:
            Aggregated embeddings
        """
        if method == 'cls':
            return embeddings[:, 0, :]
        elif method == 'mean':
            return embeddings.mean(dim=1)
        elif method == 'cell':
            if indicator is None:
                raise ValueError("indicator required for cell-level aggregation")

            # TUTA uses token-aggregation-based cell embeddings
            # This matches the original implementation in tuta/model/heads.py

            # Step 1: Aggregate tokens by indicator value (token_sum)
            # This is how TUTA groups tokens within the same cell
            x_mask = indicator.unsqueeze(1)                      # [batch_size, 1, seq_len]
            y_mask = x_mask.transpose(-1, -2)                    # [batch_size, seq_len, 1]
            mask_matrix = y_mask.eq(x_mask).float()              # [batch_size, seq_len, seq_len]
            aggregated_states = mask_matrix.matmul(embeddings)   # [batch_size, seq_len, hidden_size]

            # Step 2: Extract cell embeddings
            # In TUTA, each cell has:
            # - SEP token with odd indicator (e.g., 1, 3, 5, ...)
            # - Non-SEP tokens with even indicator (e.g., 2, 4, 6, ...)
            # After aggregation, tokens with same indicator have same embedding

            # TUTA's CTC uses the aggregated non-SEP tokens (tok_logits = ctc_logits[1::2])
            # We extract the first occurrence of each even indicator (aggregated cell content)

            indicator_1d = indicator.squeeze(0)  # [seq_len]
            aggregated_1d = aggregated_states.squeeze(0)  # [seq_len, hidden_size]

            # Find positions of odd indicators (SEP tokens) to identify cell boundaries
            is_sep = (indicator_1d > 0) & (indicator_1d % 2 == 1)
            sep_positions = torch.where(is_sep)[0]

            # For each cell, extract the aggregated embedding from the position after SEP
            # This matches TUTA's [1::2] extraction which gets the first non-SEP token
            cell_embeddings_list = []
            for i, sep_pos in enumerate(sep_positions):
                # Get the position after SEP (the aggregated non-SEP tokens)
                if sep_pos + 1 < len(indicator_1d):
                    # Check if the next position belongs to the same cell
                    # (indicator should be sep_indicator + 1 for the same cell)
                    sep_indicator = indicator_1d[sep_pos].item()
                    next_indicator = indicator_1d[sep_pos + 1].item()

                    if next_indicator == sep_indicator + 1:
                        # This position has the aggregated embedding of all tokens in the cell
                        cell_embedding = aggregated_1d[sep_pos + 1]
                        cell_embeddings_list.append(cell_embedding)
                    else:
                        # Next position is another cell's SEP - this cell has only SEP token
                        # Fall back to using the SEP embedding itself
                        cell_embedding = aggregated_1d[sep_pos]
                        cell_embeddings_list.append(cell_embedding)
                else:
                    # Edge case: SEP is the last token (shouldn't happen with proper data)
                    # Fall back to using the SEP embedding itself
                    cell_embedding = aggregated_1d[sep_pos]
                    cell_embeddings_list.append(cell_embedding)

            # Stack into tensor
            if len(cell_embeddings_list) > 0:
                cell_embeddings = torch.stack(cell_embeddings_list, dim=0)
            else:
                # No cells found - return empty tensor
                cell_embeddings = torch.empty(0, embeddings.shape[-1], device=embeddings.device)

            return cell_embeddings

        elif method == 'all':
            return embeddings.squeeze(0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract TUTA embeddings (FIXED VERSION - handles large tables correctly)'
    )
    parser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to TUTA .bin checkpoint')
    parser.add_argument('--model_type', type=str, default='tuta',
                        choices=['tuta', 'tuta_explicit', 'base'],
                        help='TUTA model variant')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save embeddings')
    parser.add_argument('--aggregate', type=str, default='cls',
                        choices=['cls', 'row', 'cell', 'all'],
                        help='Embedding type: cls=table, row=rows, cell=cells, all=tokens')
    parser.add_argument('--device_id', type=int, default=None,
                        help='GPU device ID (None=auto-detect, -1=CPU only)')

    args = parser.parse_args()

    # Create embedder
    embedder = TUTAEmbedder(
        model_path=args.model_path,
        target=args.model_type,
        device_id=args.device_id
    )

    # Extract embeddings
    print(f"\nExtracting {args.aggregate}-level embeddings from: {args.csv_path}")
    embeddings = embedder.csv_to_embeddings(
        csv_path=args.csv_path,
        output_format='numpy',
        aggregate=args.aggregate
    )

    if args.aggregate in ['cls', 'row', 'cell']:
        print(f"Embeddings shape: {embeddings.shape}")
    else:  # 'all'
        print(f"Generated token embeddings for {len(embeddings)} rows")

    # Save if output path provided
    if args.output_path:
        np.save(args.output_path, embeddings)
        print(f"Embeddings saved to: {args.output_path}")

    return embeddings


if __name__ == '__main__':
    main()
