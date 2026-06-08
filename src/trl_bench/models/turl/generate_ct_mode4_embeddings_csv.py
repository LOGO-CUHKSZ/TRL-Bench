#!/usr/bin/env python3
"""
Generate Column Embeddings for CT Mode 4 (Cell Content Only) using CSV/JSON dataset

This is an alternative version that reads from CSV files and JSON metadata
instead of the original JSON format.

Mode 4: Predict column types from CELL CONTENT only (no headers, no metadata, no pre-trained entity embeddings)

Usage:
    python generate_ct_mode4_embeddings_csv.py --data_split train
    python generate_ct_mode4_embeddings_csv.py --data_split test
"""

import argparse
import os
import sys
import pickle
import json
import csv

# Fix distutils issue before importing torch/transformers
import setuptools
import distutils.version

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import itertools

# Fix torch._six compatibility issue
if not hasattr(torch, '_six'):
    import types
    torch._six = types.ModuleType('_six')
    torch._six.string_classes = (str, bytes)

# Add code directory to path so that imports like "from model.transformers" work
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(script_dir, 'code')
sys.path.insert(0, code_dir)

from code.model.configuration import TableConfig
from code.model.model import HybridTableModel
from code.model.transformers import BertTokenizer


class CSVWikiCTDataset(Dataset):
    """Dataset that loads from CSV files and JSON metadata"""

    def __init__(self, csv_dataset_dir, data_split, entity_vocab, type_vocab,
                 max_column=10, max_input_tok=500, max_length=[50, 10, 10], tokenizer=None):
        """
        Args:
            csv_dataset_dir: Root directory of CSV dataset
            data_split: 'train' or 'test'
            entity_vocab: Entity vocabulary
            type_vocab: Type vocabulary
            max_column: Maximum number of cells per column
            max_input_tok: Maximum input tokens
            max_length: [max_title_length, max_header_length, max_cell_length]
            tokenizer: BERT tokenizer
        """
        self.data_split = data_split
        self.csv_dataset_dir = csv_dataset_dir
        self.csv_dir = os.path.join(csv_dataset_dir, data_split, "csv_tables")
        self.max_column = max_column
        self.max_input_tok = max_input_tok
        self.max_title_length = max_length[0]
        self.max_header_length = max_length[1]
        self.max_cell_length = max_length[2]

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('data/pre-trained_models/bert-base-uncased')

        # Setup vocabularies
        self.entity_vocab = entity_vocab
        self.entity_wikid2id = {self.entity_vocab[x]['wiki_id']: x for x in self.entity_vocab}
        self.type_vocab = type_vocab
        self.type_num = len(self.type_vocab)

        # Load metadata
        metadata_path = os.path.join(csv_dataset_dir, data_split, f"{data_split}_metadata.json")
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded metadata for {len(self.metadata)} tables")

        # Preprocess all tables
        self._preprocess()

    def _preprocess_table(self, table_meta):
        """Preprocess a single table - matches the original process_single_CT function"""

        # Extract metadata
        table_id = table_meta['table_id']
        pgTitle = table_meta['page_title']
        pgEnt = table_meta['page_entity_id']
        secTitle = table_meta['section_title']
        caption = table_meta['caption']
        headers = table_meta['headers']
        type_annotations = table_meta['type_annotations']

        # Read CSV file
        csv_path = os.path.join(self.csv_dir, table_meta['csv_filename'])
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            csv_rows = list(reader)
            # Skip header row which we already have
            if csv_rows and csv_rows[0] == headers:
                csv_rows = csv_rows[1:]

        # Build entities list in the original format
        # Original format: list of columns, each column is list of [[row,col], [entity_id, text]]
        entities = [[] for _ in range(len(headers))]

        # Process entity information from metadata
        for ent_info in table_meta['entities']:
            row_idx = ent_info['row']
            col_idx = ent_info['col']
            entity_id = ent_info['entity_id']
            entity_text = ent_info['entity_text']

            # Add to column list (limit by max_column)
            if len(entities[col_idx]) < self.max_column:
                entities[col_idx].append([[row_idx, col_idx], [entity_id, entity_text]])

        # Flatten entities to match original format
        entities_flat = [z for column in entities for z in column]

        # Map page entity to vocabulary
        pgEnt = self.entity_wikid2id.get(pgEnt, -1)

        # Tokenize text components
        tokenized_pgTitle = self.tokenizer.encode(pgTitle, max_length=self.max_title_length, add_special_tokens=False)
        tokenized_meta = tokenized_pgTitle + \
                        self.tokenizer.encode(secTitle, max_length=self.max_title_length, add_special_tokens=False)
        if caption != secTitle:
            tokenized_meta += self.tokenizer.encode(caption, max_length=self.max_title_length, add_special_tokens=False)

        tokenized_headers = [self.tokenizer.encode(z, max_length=self.max_header_length, add_special_tokens=False)
                            for z in headers]

        # Build token inputs
        input_tok = []
        input_tok_pos = []
        input_tok_type = []

        tokenized_meta_length = len(tokenized_meta)
        input_tok += tokenized_meta
        input_tok_pos += list(range(tokenized_meta_length))
        input_tok_type += [0] * tokenized_meta_length

        tokenized_headers_length = [len(z) for z in tokenized_headers]
        input_tok += list(itertools.chain(*tokenized_headers))
        input_tok_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
        input_tok_type += [1] * sum(tokenized_headers_length)

        # Build entity inputs
        input_ent = []
        input_ent_text = []
        input_ent_type = []
        column_en_map = {}
        row_en_map = {}

        for e_i, (index, cell) in enumerate(entities_flat):
            entity, entity_text = cell
            entity = self.entity_wikid2id.get(entity, 0)
            tokenized_ent_text = self.tokenizer.encode(entity_text, max_length=self.max_cell_length, add_special_tokens=False)
            input_ent.append(entity)
            input_ent_text.append(tokenized_ent_text)
            input_ent_type.append(4)

            if index[1] not in column_en_map:
                column_en_map[index[1]] = [e_i]
            else:
                column_en_map[index[1]].append(e_i)

            if index[0] not in row_en_map:
                row_en_map[index[0]] = [e_i]
            else:
                row_en_map[index[0]].append(e_i)

        input_ent_length = len(input_ent)

        # Create column entity mask
        column_entity_mask = np.zeros([len(type_annotations), len(input_ent)], dtype=int)
        for j in range(len(type_annotations)):
            if j in column_en_map:
                for e_i_1 in column_en_map[j]:
                    column_entity_mask[j, e_i_1] = 1

        # Create column header mask
        start_i = 0
        header_span = {}
        column_header_mask = np.zeros([len(type_annotations), len(input_tok)], dtype=int)
        for j in range(len(type_annotations)):
            header_span[j] = (start_i, start_i + tokenized_headers_length[j])
            column_header_mask[j, tokenized_meta_length + header_span[j][0]:tokenized_meta_length + header_span[j][1]] = 1
            start_i += tokenized_headers_length[j]

        # Create input masks
        tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
        meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
        header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)

        for e_i, (index, _) in enumerate(entities_flat):
            header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1

        ent_header_mask = np.transpose(header_ent_mask)

        input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
        ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)

        ent_ent_mask = np.eye(len(input_ent), dtype=int)
        for _, e_is in column_en_map.items():
            for e_i_1 in e_is:
                for e_i_2 in e_is:
                    ent_ent_mask[e_i_1, e_i_2] = 1
        for _, e_is in row_en_map.items():
            for e_i_1 in e_is:
                for e_i_2 in e_is:
                    ent_ent_mask[e_i_1, e_i_2] = 1

        input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]

        # Prepend pgEnt to input_ent (as in original)
        if pgEnt != -1:
            input_tok_mask[1] = np.concatenate([np.ones([len(input_tok), 1], dtype=int), input_tok_mask[1]], axis=1)
        else:
            input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int), input_tok_mask[1]], axis=1)

        input_ent = [pgEnt if pgEnt != -1 else 0] + input_ent
        input_ent_text = [tokenized_pgTitle[:self.max_cell_length]] + input_ent_text
        input_ent_type = [2] + input_ent_type

        new_input_ent_mask = [np.ones([len(input_ent), len(input_tok)], dtype=int),
                             np.ones([len(input_ent), len(input_ent)], dtype=int)]
        new_input_ent_mask[0][1:, :] = input_ent_mask[0]
        new_input_ent_mask[1][1:, 1:] = input_ent_mask[1]
        if pgEnt == -1:
            new_input_ent_mask[1][:, 0] = 0
            new_input_ent_mask[1][0, :] = 0

        column_entity_mask = np.concatenate([np.zeros([len(type_annotations), 1], dtype=int), column_entity_mask], axis=1)
        input_ent_mask = new_input_ent_mask

        # Create labels
        labels = np.zeros([len(type_annotations), self.type_num], dtype=int)
        for j, types in enumerate(type_annotations):
            for t in types:
                labels[j, self.type_vocab[t]] = 1

        # Pad entity text
        input_ent_cell_length = [len(x) if len(x) != 0 else 1 for x in input_ent_text]
        max_cell_length = max(input_ent_cell_length)
        input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
        for i, x in enumerate(input_ent_text):
            input_ent_text_padded[i, :len(x)] = x

        return [table_id, np.array(input_tok), np.array(input_tok_type), np.array(input_tok_pos),
                (np.array(input_tok_mask[0]), np.array(input_tok_mask[1])), len(input_tok),
                np.array(input_ent), input_ent_text_padded, input_ent_cell_length, np.array(input_ent_type),
                (np.array(input_ent_mask[0]), np.array(input_ent_mask[1])), len(input_ent),
                column_header_mask, column_entity_mask, labels, len(labels)]

    def _preprocess(self):
        """Preprocess all tables"""
        print("Preprocessing tables...")
        self.data = []
        for table_meta in tqdm(self.metadata, desc="Processing tables"):
            processed = self._preprocess_table(table_meta)
            self.data.append(processed)
        print(f"Preprocessed {len(self.data)} tables")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class CSVCTLoader(DataLoader):
    """DataLoader for CSV-based CT dataset - mirrors the original CTLoader"""

    def __init__(self, dataset, sampler, batch_size=1, is_train=False):
        self.is_train = is_train

        def collate_fn(data):
            # Transpose list of tuples
            transposed_data = list(zip(*data))

            table_id = transposed_data[0]

            # Pad input_tok arrays to max length in batch
            input_tok_arrays = transposed_data[1]
            max_tok_len = max(arr.shape[0] for arr in input_tok_arrays)
            padded_input_tok = np.zeros((len(input_tok_arrays), max_tok_len), dtype=np.int64)
            for i, arr in enumerate(input_tok_arrays):
                padded_input_tok[i, :arr.shape[0]] = arr
            input_tok = torch.LongTensor(padded_input_tok)

            # Pad input_tok_type arrays
            input_tok_type_arrays = transposed_data[2]
            padded_input_tok_type = np.zeros((len(input_tok_type_arrays), max_tok_len), dtype=np.int64)
            for i, arr in enumerate(input_tok_type_arrays):
                padded_input_tok_type[i, :arr.shape[0]] = arr
            input_tok_type = torch.LongTensor(padded_input_tok_type)

            # Pad input_tok_pos arrays
            input_tok_pos_arrays = transposed_data[3]
            padded_input_tok_pos = np.zeros((len(input_tok_pos_arrays), max_tok_len), dtype=np.int64)
            for i, arr in enumerate(input_tok_pos_arrays):
                padded_input_tok_pos[i, :arr.shape[0]] = arr
            input_tok_pos = torch.LongTensor(padded_input_tok_pos)

            # Pad input_tok_mask arrays - combined format
            max_ent_len = max(d[1].shape[1] for d in transposed_data[4])  # mask1 has entity dimension

            # Create combined mask as in original
            padded_input_tok_mask = np.zeros((len(transposed_data[4]), max_tok_len, max_tok_len + max_ent_len), dtype=np.int64)
            for i, (mask0, mask1) in enumerate(transposed_data[4]):
                # tok-to-tok part
                padded_input_tok_mask[i, :mask0.shape[0], :mask0.shape[1]] = mask0
                # tok-to-ent part
                padded_input_tok_mask[i, :mask1.shape[0], max_tok_len:max_tok_len + mask1.shape[1]] = mask1
            input_tok_mask = torch.LongTensor(padded_input_tok_mask)

            # Pad input_ent arrays
            input_ent_arrays = transposed_data[6]
            padded_input_ent = np.zeros((len(input_ent_arrays), max_ent_len), dtype=np.int64)
            for i, arr in enumerate(input_ent_arrays):
                padded_input_ent[i, :arr.shape[0]] = arr
            input_ent = torch.LongTensor(padded_input_ent)

            # Process input_ent_text - pad to max length in batch
            input_ent_text_arrays = transposed_data[7]
            max_ent_text_len = max(arr.shape[1] for arr in input_ent_text_arrays)
            padded_input_ent_text = np.zeros((len(input_ent_text_arrays), max_ent_len, max_ent_text_len), dtype=np.int64)
            for i, arr in enumerate(input_ent_text_arrays):
                padded_input_ent_text[i, :arr.shape[0], :arr.shape[1]] = arr
            input_ent_text = torch.LongTensor(padded_input_ent_text)

            # Process input_ent_text_length - pad with 1s for empty entities
            input_ent_text_length_lists = transposed_data[8]
            padded_input_ent_text_length = np.ones((len(input_ent_text_length_lists), max_ent_len), dtype=np.int64)
            for i, length_list in enumerate(input_ent_text_length_lists):
                padded_input_ent_text_length[i, :len(length_list)] = length_list
            input_ent_text_length = torch.LongTensor(padded_input_ent_text_length)

            # Pad input_ent_type arrays
            input_ent_type_arrays = transposed_data[9]
            padded_input_ent_type = np.zeros((len(input_ent_type_arrays), max_ent_len), dtype=np.int64)
            for i, arr in enumerate(input_ent_type_arrays):
                padded_input_ent_type[i, :arr.shape[0]] = arr
            input_ent_type = torch.LongTensor(padded_input_ent_type)

            # Pad input_ent_mask arrays - combined format
            padded_input_ent_mask = np.zeros((len(transposed_data[10]), max_ent_len, max_tok_len + max_ent_len), dtype=np.int64)
            for i, (mask0, mask1) in enumerate(transposed_data[10]):
                # ent-to-tok part
                padded_input_ent_mask[i, :mask0.shape[0], :mask0.shape[1]] = mask0
                # ent-to-ent part
                padded_input_ent_mask[i, :mask1.shape[0], max_tok_len:max_tok_len + mask1.shape[1]] = mask1
            input_ent_mask = torch.LongTensor(padded_input_ent_mask)

            # Pad column masks
            max_col_len = max(arr.shape[0] for arr in transposed_data[12])

            padded_column_header_mask = np.zeros((len(transposed_data[12]), max_col_len, max_tok_len), dtype=np.float32)
            for i, arr in enumerate(transposed_data[12]):
                col_num = arr.shape[0]
                padded_column_header_mask[i, :col_num, :arr.shape[1]] = arr
                # Set padded columns to point to first token position (matching original)
                if col_num < max_col_len:
                    padded_column_header_mask[i, col_num:, 0] = 1
            column_header_mask = torch.FloatTensor(padded_column_header_mask)

            padded_column_entity_mask = np.zeros((len(transposed_data[13]), max_col_len, max_ent_len), dtype=np.float32)
            for i, arr in enumerate(transposed_data[13]):
                col_num = arr.shape[0]
                padded_column_entity_mask[i, :col_num, :arr.shape[1]] = arr
                # Set padded columns to point to first entity position (matching original)
                if col_num < max_col_len:
                    padded_column_entity_mask[i, col_num:, 0] = 1
            column_entity_mask = torch.FloatTensor(padded_column_entity_mask)

            # Pad labels
            num_types = transposed_data[14][0].shape[-1]
            padded_labels = np.zeros((len(transposed_data[14]), max_col_len, num_types), dtype=np.float32)
            for i, arr in enumerate(transposed_data[14]):
                padded_labels[i, :arr.shape[0], :] = arr
            labels = torch.FloatTensor(padded_labels)
            labels_mask = torch.FloatTensor((labels.sum(dim=2) > 0).float())

            return (table_id, input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                   input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask,
                   column_entity_mask, column_header_mask, labels_mask, labels)

        super().__init__(dataset, sampler=sampler, batch_size=batch_size,
                        collate_fn=collate_fn, pin_memory=False)


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
        checkpoint = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location=device
        )

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

        print(f"Model loaded on {device}")

    def extract_embeddings(self, dataloader):
        """
        Extract column embeddings from cell content for all samples

        Mode 4: Uses ONLY input_ent_text (cell content)
        Ignores: metadata (caption+headers), pre-trained entity embeddings

        Returns:
            dict with:
                embeddings: List of arrays, each with shape [num_columns, hidden_size]
                labels: List of arrays, each with shape [num_columns, num_types]
                table_ids: List of table IDs
                labels_masks: List of labels masks
        """
        all_embeddings = []
        all_labels = []
        all_table_ids = []
        all_labels_masks = []

        print("Extracting Mode 4 embeddings (cell content only)...")
        print("  Using: input_ent_text (cell content)")
        print("  Ignoring: metadata (caption+headers), entity embeddings")
        print()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                table_id, input_tok, input_tok_type, input_tok_pos, input_tok_mask, \
                    input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask, \
                    column_entity_mask, column_header_mask, labels_mask, labels = batch

                # Move entity inputs to device (we need these for Mode 4)
                input_ent_text = input_ent_text.to(self.device)
                input_ent_text_length = input_ent_text_length.to(self.device)
                input_ent_type = input_ent_type.to(self.device)
                input_ent_mask = input_ent_mask.to(self.device)
                column_entity_mask = column_entity_mask.to(self.device)

                # MODE 4: Set metadata and pre-trained embeddings to None
                input_tok = None
                input_tok_type = None
                input_tok_pos = None
                input_tok_mask = None
                input_ent = None  # No pre-trained entity embeddings!

                # Extract only ent-to-ent part of the mask for mode 4
                # The mask is [batch, ent_len, tok_len+ent_len]
                # We need just [batch, ent_len, ent_len] for mode 4
                ent_mask_size = input_ent_text.shape[1]
                # input_ent_mask shape: [batch, ent_len, tok_len + ent_len]
                # We want: [batch, ent_len, ent_len] (just the ent-ent part)
                # The ent-ent part starts at position tok_len in the last dimension
                # Since we're setting input_tok to None, we extract from the end
                input_ent_mask = input_ent_mask[:, :, -ent_mask_size:]

                # Forward pass through table model
                # tok_outputs will be None since input_tok is None
                tok_outputs, ent_outputs, _ = self.model(
                    None, None, None, None,  # All token inputs = None
                    input_ent_text, input_ent_text_length, None,
                    None, input_ent_type, input_ent_mask, None
                )

                # Extract entity sequence output (from BERT encoding of cell text)
                ent_sequence_output = ent_outputs[0]  # [batch, num_entities, hidden_size]

                # Aggregate entities to column level using column_entity_mask
                # column_entity_mask: [batch, num_columns, num_entities]
                # ent_sequence_output: [batch, num_entities, hidden_size]
                # Result: [batch, num_columns, hidden_size]
                ent_col_output = torch.matmul(column_entity_mask, ent_sequence_output)

                # Normalize by number of entities per column
                ent_col_output /= column_entity_mask.sum(dim=-1, keepdim=True).clamp(1.0, 9999.0)

                # Move to CPU and store
                ent_col_output_cpu = ent_col_output.cpu().numpy()  # [batch, num_columns, hidden_size]
                labels_cpu = labels.cpu().numpy()  # [batch, num_columns, num_types]
                labels_mask_cpu = labels_mask.cpu().numpy()  # [batch, num_columns]

                # Split batch into individual samples
                for i in range(ent_col_output_cpu.shape[0]):
                    all_embeddings.append(ent_col_output_cpu[i])
                    all_labels.append(labels_cpu[i])
                    all_table_ids.append(table_id[i] if isinstance(table_id, (list, tuple)) else table_id[i].item())
                    all_labels_masks.append(labels_mask_cpu[i])

        print(f"Extracted embeddings for {len(all_embeddings)} tables")

        return {
            'embeddings': all_embeddings,
            'labels': all_labels,
            'table_ids': all_table_ids,
            'labels_masks': all_labels_masks
        }


def main():
    parser = argparse.ArgumentParser(description='Generate CT Mode 4 embeddings from CSV dataset (cell content only)')
    parser.add_argument('--pretrained_model', type=str,
                        default='checkpoint/pretrained',
                        help='Path to pretrained model checkpoint (relative to parent dir)')
    parser.add_argument('--csv_dataset_dir', type=str,
                        default='../ct_mode2_pipeline/csv_dataset',
                        help='Path to CSV dataset directory (can reuse mode2 CSV dataset)')
    parser.add_argument('--data_split', type=str,
                        required=True,
                        choices=['train', 'test'],
                        help='Which data split to process')
    parser.add_argument('--output_dir', type=str,
                        default='embeddings_mode4_csv',
                        help='Output directory for embeddings (relative to pipeline dir)')
    parser.add_argument('--batch_size', type=int,
                        default=20,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Make paths absolute
    if not os.path.isabs(args.pretrained_model):
        args.pretrained_model = os.path.join(parent_dir, args.pretrained_model)
    if not os.path.isabs(args.csv_dataset_dir):
        # Try relative to script_dir first, then parent_dir
        if os.path.exists(os.path.join(script_dir, args.csv_dataset_dir)):
            args.csv_dataset_dir = os.path.join(script_dir, args.csv_dataset_dir)
        else:
            args.csv_dataset_dir = os.path.join(parent_dir, args.csv_dataset_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("CT Mode 4 Embedding Generation (CSV Dataset - Cell Content Only)")
    print("="*80)
    print(f"Mode 4: Uses ONLY cell content (input_ent_text)")
    print(f"  Ignores: metadata (caption+headers)")
    print(f"  Ignores: pre-trained entity embeddings")
    print()
    print(f"Pretrained model: {args.pretrained_model}")
    print(f"CSV Dataset directory: {args.csv_dataset_dir}")
    print(f"Data split: {args.data_split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Load vocabularies from saved pickle files
    print("Loading vocabularies...")
    vocab_dir = os.path.join(args.csv_dataset_dir, "vocabularies")

    with open(os.path.join(vocab_dir, "entity_vocab.pkl"), 'rb') as f:
        entity_vocab = pickle.load(f)
    with open(os.path.join(vocab_dir, "type_vocab.pkl"), 'rb') as f:
        type_vocab = pickle.load(f)

    print(f"  Entity vocab size: {len(entity_vocab)}")
    print(f"  Type vocab size: {len(type_vocab)}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"  Tokenizer loaded from HuggingFace: bert-base-uncased")
    print()

    # Load dataset
    print(f"Loading {args.data_split} dataset from CSV files...")
    dataset = CSVWikiCTDataset(
        args.csv_dataset_dir,
        args.data_split,
        entity_vocab,
        type_vocab,
        max_input_tok=500,
        max_length=[50, 10, 10],
        tokenizer=tokenizer
    )
    print(f"  Dataset size: {len(dataset)}")
    print()

    # Create dataloader
    sampler = SequentialSampler(dataset)
    dataloader = CSVCTLoader(dataset, sampler=sampler, batch_size=args.batch_size, is_train=False)

    # Initialize extractor
    extractor = Mode4EmbeddingExtractor(args.pretrained_model, args.device)

    # Extract embeddings
    results = extractor.extract_embeddings(dataloader)

    # Save results
    output_file = os.path.join(args.output_dir, f'{args.data_split}_embeddings.pkl')
    print(f"\nSaving embeddings to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f, protocol=4)

    # Save metadata
    metadata = {
        'num_samples': len(results['embeddings']),
        'hidden_size': results['embeddings'][0].shape[-1],
        'num_types': results['labels'][0].shape[-1],
        'data_split': args.data_split,
        'mode': 4,
        'description': 'Cell content only (no headers, no metadata, no entity embeddings)',
        'pretrained_model': args.pretrained_model,
        'csv_dataset_dir': args.csv_dataset_dir
    }
    metadata_file = os.path.join(args.output_dir, f'{args.data_split}_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print("\nDone!")
    print(f"  Mode: 4 (cell content only)")
    print(f"  Samples: {metadata['num_samples']}")
    print(f"  Hidden size: {metadata['hidden_size']}")
    print(f"  Num types: {metadata['num_types']}")
    print()


if __name__ == '__main__':
    main()
