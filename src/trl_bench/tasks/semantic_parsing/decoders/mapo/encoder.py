"""
Lightweight encoder that loads pre-computed embeddings and applies trainable projections.

This encoder is designed for the embedding-based evaluation framework where
embeddings are pre-computed and cached to pkl files.
"""

import pickle
from typing import Dict, List

import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence

from .encoder_base import EncoderBase, ContextEncoding, COLUMN_TYPES


class _BertModelStub:
    """Stub to provide bert_model interface for BertDecoder compatibility.

    BertDecoder expects encoder.bert_model.bert_config.hidden_size.
    This stub provides that interface using the embedding dimension from the cache.
    """
    def __init__(self, hidden_size: int):
        self.bert_config = type('BertConfig', (), {'hidden_size': hidden_size})()


class EmbeddingEncoder(EncoderBase):
    """
    Lightweight encoder that loads pre-computed embeddings
    and applies trainable projection layers.

    Dimension-agnostic: works with any embedding size D.

    This produces the same output format as BertEncoder.encode()
    so the decoder works unchanged.
    """

    def __init__(
        self,
        column_pkl_path: str,
        question_pkl_paths: List[str],
        output_size: int,
        config: Dict,
        question_feat_size: int,
        builtin_func_num: int,
        memory_size: int,
        column_feature_num: int,
        dropout: float = 0.
    ):
        EncoderBase.__init__(self, output_size, builtin_func_num, memory_size)

        self.config = config
        self.question_feat_size = question_feat_size
        self.dropout = nn.Dropout(dropout)
        self.max_variable_num_on_memory = memory_size - builtin_func_num
        self.column_feature_num = column_feature_num

        # Load column embeddings: {table_id: (num_cols, dim)}
        print(f'Loading column embeddings from {column_pkl_path}...', flush=True)
        with open(column_pkl_path, 'rb') as f:
            col_data = pickle.load(f)
        self.column_cache: Dict[str, np.ndarray] = {}
        for entry in col_data:
            table_id = entry['table_id']
            col_embs = entry['column_embeddings']  # {0: (dim,), 1: (dim,), ...}
            stacked = np.stack([col_embs[i] for i in range(len(col_embs))], axis=0)
            self.column_cache[table_id] = stacked

        # Load question embeddings: {text_id: (seq_len, dim)}
        self.question_cache: Dict[str, np.ndarray] = {}
        for qpath in question_pkl_paths:
            print(f'Loading question embeddings from {qpath}...', flush=True)
            with open(qpath, 'rb') as f:
                q_data = pickle.load(f)
            for entry in q_data:
                self.question_cache[entry['text_id']] = entry['embedding']

        # Infer embedding dimensions from caches
        sample_col = next(iter(self.column_cache.values()))
        self.embedding_dim = sample_col.shape[-1]  # column embedding dim
        sample_q = next(iter(self.question_cache.values()))
        self.question_embedding_dim = sample_q.shape[-1]  # question embedding dim
        print(f'Column embedding dimension: {self.embedding_dim}', flush=True)
        print(f'Question embedding dimension: {self.question_embedding_dim}', flush=True)
        print(f'Loaded {len(self.column_cache)} tables, {len(self.question_cache)} questions', flush=True)

        # Add stub bert_model for BertDecoder compatibility
        # Uses question dim because cls_encoding (mean-pooled questions) feeds into decoder_cell_init_linear
        self.bert_model = _BertModelStub(self.question_embedding_dim)

        # Trainable projection layers
        # bert_output_project is applied to question embeddings, so uses question_embedding_dim
        self.bert_output_project = nn.Linear(
            self.question_embedding_dim + question_feat_size,
            output_size,
            bias=False
        )

        self.question_encoding_att_value_to_key = nn.Linear(
            output_size,
            output_size,
            bias=False
        )

        # Column type embedding and projection
        if self.config['table_representation'] == 'canonical':
            self.column_type_to_id = {t: i for i, t in enumerate(COLUMN_TYPES)}
            self.column_type_embedding = nn.Embedding(len(self.column_type_to_id), self.config['value_embedding_size'])

        if self.config.get('use_column_type_embedding', False):
            self.bert_table_output_project = nn.Linear(
                self.embedding_dim + self.column_type_embedding.embedding_dim,
                output_size,
                bias=False
            )
        else:
            self.bert_table_output_project = nn.Linear(
                self.embedding_dim,
                output_size,
                bias=False
            )

        self.constant_value_embedding_linear = lambda x: x

        self.init_weights()

    def init_weights(self):
        """Initialize weights with normal distribution."""
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        modules = [
            module
            for name, module
            in self._modules.items()
            if module
        ]

        for module in modules:
            module.apply(_init_weights)

    @classmethod
    def build(cls, config, column_pkl_path: str, question_pkl_paths: List[str], master=None):
        """Build the encoder from config."""
        return cls(
            column_pkl_path=column_pkl_path,
            question_pkl_paths=question_pkl_paths,
            output_size=config['hidden_size'],
            question_feat_size=config['n_en_input_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            column_feature_num=config['n_de_output_features'],
            dropout=config['dropout'],
            config=config
        )

    def get_embeddings(self, example_id: str, table_id: str) -> Dict[str, torch.Tensor]:
        """Load embeddings for a single example from cache."""
        device = next(self.parameters()).device
        return {
            'question': torch.from_numpy(self.question_cache[example_id]).to(device),
            'column': torch.from_numpy(self.column_cache[table_id]).to(device),
        }

    def example_list_to_batch(self, env_context: List[Dict]) -> Dict:
        """Convert list of env contexts to batched tensors."""
        batch_dict = dict()
        for key in ('constant_spans', 'question_features'):
            val_list = [x[key] for x in env_context]

            if key == 'question_features':
                max_entry_num = max(len(val) for val in val_list)
                dtype = np.float32
            else:
                max_entry_num = self.max_variable_num_on_memory
                dtype = np.int64

            entry_dim = len(val_list[0][0])
            batch_size = len(env_context)

            batch_value_tensor = np.zeros((batch_size, max_entry_num, entry_dim), dtype=dtype)

            if key == 'constant_spans':
                batch_value_tensor.fill(-1.)

            for i, val in enumerate(val_list):
                entry_num = len(val)
                batch_value_tensor[i, :entry_num] = val

            batch_dict[key] = torch.from_numpy(batch_value_tensor).to(next(self.parameters()).device)

        return batch_dict

    def encode(self, env_context: List[Dict]) -> ContextEncoding:
        """
        Load cached embeddings and apply projections.

        Args:
            env_context: List of dicts containing example metadata
                         (question_features, constant_spans, table, id, etc.)

        Returns:
            ContextEncoding dict compatible with decoder (same format as BertEncoder.encode())
        """
        batch_size = len(env_context)
        device = next(self.parameters()).device

        batched_context = self.example_list_to_batch(env_context)

        # 1. Load embeddings from cache
        example_ids = [ctx['id'] for ctx in env_context]
        table_ids = [ctx['table'].id for ctx in env_context]
        question_embs = []
        column_embs = []
        question_lengths = []
        column_lengths = []

        for eid, tid in zip(example_ids, table_ids):
            emb = self.get_embeddings(eid, tid)
            question_embs.append(emb['question'])
            column_embs.append(emb['column'])
            question_lengths.append(emb['question'].shape[0])
            column_lengths.append(emb['column'].shape[0])

        # 2. Pad to batch
        # (batch_size, max_seq_len, embedding_dim)
        question_encoding = pad_sequence(question_embs, batch_first=True)
        # (batch_size, max_col_num, embedding_dim)
        canonical_column_encoding = pad_sequence(column_embs, batch_first=True)

        # Create masks
        max_question_len = question_encoding.size(1)
        max_column_num = canonical_column_encoding.size(1)

        question_mask = torch.zeros(batch_size, max_question_len, device=device)
        for i, length in enumerate(question_lengths):
            question_mask[i, :length] = 1.

        canonical_column_mask = torch.zeros(batch_size, max_column_num, device=device)
        for i, length in enumerate(column_lengths):
            canonical_column_mask[i, :length] = 1.

        # Compute cls_encoding from RAW embeddings BEFORE projection
        # BertEncoder uses the [CLS] token; we approximate with mean pooling of raw embeddings
        cls_encoding = (question_encoding * question_mask.unsqueeze(-1)).sum(dim=1) / question_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # 3. Apply question projection
        # Pad question_features to match question_encoding length
        question_features = batched_context['question_features']
        if question_features.size(1) < max_question_len:
            padding = torch.zeros(
                batch_size, max_question_len - question_features.size(1), question_features.size(2),
                device=device
            )
            question_features = torch.cat([question_features, padding], dim=1)
        else:
            question_features = question_features[:, :max_question_len, :]

        if self.question_feat_size > 0:
            question_encoding_with_feat = torch.cat([question_encoding, question_features], dim=-1)
        else:
            question_encoding_with_feat = question_encoding

        question_encoding = self.bert_output_project(question_encoding_with_feat)
        question_encoding_att_linear = self.question_encoding_att_value_to_key(question_encoding)

        context_encoding = {
            'batch_size': batch_size,
            'question_encoding': question_encoding,
            'question_mask': question_mask,
            'question_encoding_att_linear': question_encoding_att_linear,
        }

        # 4. Apply column projection with canonical table handling
        table_column_encoding = canonical_column_encoding
        table_column_mask = canonical_column_mask
        constant_value_num = batched_context['constant_spans'].size(1)

        if self.config['table_representation'] == 'canonical':
            new_tensor = table_column_encoding.new_tensor

            # Map raw columns to canonical columns
            raw_column_canonical_ids = np.zeros((batch_size, constant_value_num), dtype=np.int64)
            raw_column_mask = np.zeros((batch_size, constant_value_num), dtype=np.float32)
            raw_column_type_ids = np.zeros((batch_size, constant_value_num), dtype=np.int64)

            for e_id, context in enumerate(env_context):
                column_info = context['table'].column_info
                raw_columns = column_info['raw_columns']
                valid_column_num = min(constant_value_num, len(raw_columns))
                raw_column_canonical_ids[e_id, :valid_column_num] = column_info['raw_column_canonical_ids'][:valid_column_num]

                raw_column_type_ids[e_id, :valid_column_num] = [
                    self.column_type_to_id[col.type]
                    for col
                    in raw_columns
                ][:valid_column_num]

                raw_column_mask[e_id, :valid_column_num] = 1.

            raw_column_canonical_ids = new_tensor(raw_column_canonical_ids, dtype=torch.long)

            # Gather column encodings based on canonical IDs
            table_column_encoding = torch.gather(
                canonical_column_encoding,
                dim=1,
                index=raw_column_canonical_ids.unsqueeze(-1).expand(-1, -1, canonical_column_encoding.size(-1))
            )

            if self.config.get('use_column_type_embedding', False):
                type_fused_column_encoding = torch.cat(
                    [
                        table_column_encoding,
                        self.column_type_embedding(new_tensor(raw_column_type_ids, dtype=torch.long))
                    ],
                    dim=-1
                )
                table_column_encoding = type_fused_column_encoding

            table_column_mask = new_tensor(raw_column_mask)
            table_column_encoding = table_column_encoding * table_column_mask.unsqueeze(-1)
            max_column_num = table_column_encoding.size(1)

        # Apply column projection
        table_column_encoding = self.bert_table_output_project(table_column_encoding)

        if max_column_num < constant_value_num:
            constant_value_embedding = torch.cat([
                table_column_encoding,
                table_column_encoding.new_zeros(
                    batch_size, constant_value_num - max_column_num, table_column_encoding.size(-1))],
                dim=1)
        else:
            constant_value_embedding = table_column_encoding[:, :constant_value_num, :]

        # Get constant encoding (combines columns and entity spans)
        constant_encoding, constant_mask = self.get_constant_encoding(
            question_encoding, batched_context['constant_spans'], constant_value_embedding, table_column_mask)

        # Create a fake table_bert_encoding dict for compatibility
        # This is used by some downstream code for logging
        table_bert_encoding = {
            'question_encoding': question_encoding,
            'column_encoding': table_column_encoding,
            'context_token_mask': question_mask,
            'column_mask': table_column_mask,
            'input_tables': [ctx['table'] for ctx in env_context],
        }

        context_encoding.update({
            'column_encoding': table_column_encoding,
            'column_mask': table_column_mask,
            'canonical_column_encoding': canonical_column_encoding,
            'canonical_column_mask': canonical_column_mask,
            'cls_encoding': cls_encoding,
            'table_bert_encoding': table_bert_encoding,
            'constant_encoding': constant_encoding,
            'constant_mask': constant_mask
        })

        return context_encoding

    def get_constant_encoding(self, question_token_encoding, constant_span, constant_value_embedding, column_mask):
        """
        Compute constant encoding from question spans and column embeddings.

        This matches the implementation in BertEncoder.get_constant_encoding().
        """
        # (batch_size, mem_size)
        constant_span_mask = torch.ge(constant_span, 0)[:, :, 0].float()

        # mask out entries <= 0
        constant_span = constant_span * constant_span_mask.unsqueeze(-1).long()

        constant_span_size = constant_span.size()
        mem_size = constant_span_size[1]
        batch_size = question_token_encoding.size(0)

        # (batch_size, mem_size, 2, embed_size)
        constant_span_embedding = torch.gather(
            question_token_encoding.unsqueeze(1).expand(-1, mem_size, -1, -1),
            index=constant_span.unsqueeze(-1).expand(-1, -1, -1, question_token_encoding.size(-1)),
            dim=2
        )

        # (batch_size, mem_size, embed_size)
        constant_span_embedding = torch.mean(constant_span_embedding, dim=-2)
        constant_span_embedding = constant_span_embedding * constant_span_mask.unsqueeze(-1)

        constant_value_embedding = self.constant_value_embedding_linear(constant_value_embedding)

        constant_encoding = constant_value_embedding + constant_span_embedding
        constant_mask = (constant_span_mask.byte() | column_mask.byte()).float()

        return constant_encoding, constant_mask
