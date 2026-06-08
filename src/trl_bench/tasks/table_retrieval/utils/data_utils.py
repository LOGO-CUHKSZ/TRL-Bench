"""
Data utilities for Table Retrieval downstream task.

Provides functions for:
- Loading pre-computed embeddings (pickle formats)
- Creating PyTorch datasets for training
- Loading question/table data from JSON
- Building and managing id2table mappings
"""

import json
import logging
import os
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trl_bench.utils.unified_embedding_format import get_table_level_embedding
from trl_bench.utils.pickle_compat import load_pickle

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Classes
# =============================================================================

class EmbeddingRetrievalDataset(Dataset):
    """
    Dataset for retrieval training with pre-computed embeddings.

    Each sample consists of:
    - Query embedding
    - Positive table embedding
    - Hard negative table embeddings (optional)
    - Random negative table embeddings (from in-batch)
    """

    def __init__(
        self,
        table_embeddings: Dict[str, np.ndarray],  # table_id -> embedding
        query_embeddings: Dict[str, np.ndarray],  # question_id -> embedding
        training_data: List[Dict],  # [{question_id, table_id, ...}]
        num_hard_negatives: int = 5,
        hard_negative_ids: Optional[Dict[str, List[str]]] = None,  # question_id -> [table_ids]
        embedding_type: str = 'table_embedding',  # For pickle files with multiple types
    ):
        """
        Initialize dataset.

        Args:
            table_embeddings: Dict mapping table_id to embedding array
            query_embeddings: Dict mapping question_id to embedding array
            training_data: List of training samples with question_id and table_id
            num_hard_negatives: Number of hard negatives per sample
            hard_negative_ids: Pre-computed hard negatives per question
            embedding_type: Which embedding type to use from pickle files
        """
        self.table_embeddings = table_embeddings
        self.query_embeddings = query_embeddings
        self.training_data = training_data
        self.num_hard_negatives = num_hard_negatives
        self.hard_negative_ids = hard_negative_ids or {}
        self.embedding_type = embedding_type

        # Build list of all table IDs for random negative sampling
        self.all_table_ids = list(table_embeddings.keys())

        # Filter samples where we have both query and table embeddings
        self.valid_samples = []
        for sample in training_data:
            q_id = sample.get('question_id')
            t_id = sample.get('table_id')
            if q_id in query_embeddings and t_id in table_embeddings:
                self.valid_samples.append(sample)

        print(f"Loaded {len(self.valid_samples)}/{len(training_data)} valid samples")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.valid_samples[idx]
        q_id = sample['question_id']
        pos_table_id = sample['table_id']

        # Get embeddings
        query_emb = self.query_embeddings[q_id]
        positive_emb = self.table_embeddings[pos_table_id]

        # Get hard negatives if available
        hard_neg_embs = []
        hard_neg_ids = self.hard_negative_ids.get(q_id, [])
        for neg_id in hard_neg_ids[:self.num_hard_negatives]:
            if neg_id in self.table_embeddings and neg_id != pos_table_id:
                hard_neg_embs.append(self.table_embeddings[neg_id])

        return {
            'query_emb': torch.tensor(query_emb, dtype=torch.float32),
            'positive_emb': torch.tensor(positive_emb, dtype=torch.float32),
            'hard_neg_embs': [torch.tensor(e, dtype=torch.float32) for e in hard_neg_embs],
            'question_id': q_id,
            'table_id': pos_table_id,
        }


# =============================================================================
# Collate Functions
# =============================================================================

def collate_with_hard_negatives(batch: List[Dict]) -> Dict:
    """
    Collate function for batches with hard negatives.

    Creates a batch where each query has:
    - Its positive table at index i (diagonal)
    - Hard negatives following the positive
    - Other batch samples as in-batch negatives

    Returns:
        Dict with:
        - query_embs: [batch_size, dim]
        - context_embs: [batch_size * (1 + num_hard_neg), dim]
        - positive_idx: [batch_size] - index of positive for each query
        - hard_neg_idx: [batch_size, num_hard_neg] - indices of hard negatives
    """
    batch_size = len(batch)

    # Stack query embeddings
    query_embs = torch.stack([item['query_emb'] for item in batch])

    # Collect all context embeddings (positives + hard negatives)
    context_embs = []
    positive_idx = []
    hard_neg_idx = []

    for i, item in enumerate(batch):
        # Add positive
        pos_start_idx = len(context_embs)
        context_embs.append(item['positive_emb'])
        positive_idx.append(pos_start_idx)

        # Add hard negatives
        item_hard_neg_idx = []
        for neg_emb in item['hard_neg_embs']:
            item_hard_neg_idx.append(len(context_embs))
            context_embs.append(neg_emb)
        hard_neg_idx.append(item_hard_neg_idx)

    context_embs = torch.stack(context_embs)

    return {
        'query_embs': query_embs,
        'context_embs': context_embs,
        'positive_idx': positive_idx,
        'hard_neg_idx': hard_neg_idx,
        'question_ids': [item['question_id'] for item in batch],
        'table_ids': [item['table_id'] for item in batch],
    }


def collate_in_batch_only(batch: List[Dict]) -> Dict:
    """
    Collate function using only in-batch negatives (no hard negatives).

    Each query's positive is at the same index (diagonal of score matrix).

    Returns:
        Dict with:
        - query_embs: [batch_size, dim]
        - context_embs: [batch_size, dim] - positives only
        - positive_idx: [0, 1, 2, ...] - diagonal indices
    """
    query_embs = torch.stack([item['query_emb'] for item in batch])
    context_embs = torch.stack([item['positive_emb'] for item in batch])
    positive_idx = list(range(len(batch)))

    return {
        'query_embs': query_embs,
        'context_embs': context_embs,
        'positive_idx': positive_idx,
        'question_ids': [item['question_id'] for item in batch],
        'table_ids': [item['table_id'] for item in batch],
    }


# =============================================================================
# Embedding Loading Functions
# =============================================================================

def load_table_embeddings(
    path: str,
    embedding_type: str = 'column_mean',
    table_id_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load table embeddings from pickle file.

    Supports unified list format (v2.0):
        [{'table_id': ..., 'table_embedding': {dict}, ...}, ...]

    Args:
        path: Path to embeddings file (.pkl)
        embedding_type: Which embedding variant to extract:
            - 'column_mean': Mean-pooled column embedding (default)
            - 'cls_embedding': CLS token embedding
            - 'table_embedding': Native table embedding
            - 'token_mean': Mean of all non-padding token hidden states
        table_id_mapping: Optional dict mapping raw table_id (e.g. CSV basename)
            to canonical table_id (e.g. with spaces). If provided, remaps keys.
            Can be built from table_id_to_csv.json: {csv_basename: original_table_id}.

    Returns:
        Dict mapping table_id to embedding array
    """
    variant = embedding_type

    if path.endswith('.pkl'):
        data = load_pickle(path)

        embeddings = {}
        skipped = 0

        if isinstance(data, list):
            for item in data:
                table_id = item.get('table_id')
                if not table_id:
                    table_path = item.get('table', '')
                    table_basename = os.path.basename(table_path)
                    table_id = table_basename[:-4] if table_basename.endswith('.csv') else table_basename

                if not table_id:
                    continue

                emb = get_table_level_embedding(item, variant=variant)
                if emb is not None:
                    embeddings[table_id] = emb
                else:
                    skipped += 1

            if skipped:
                logger.warning(
                    "Skipped %d/%d tables missing variant '%s' in %s",
                    skipped, skipped + len(embeddings), variant, path,
                )

        else:
            raise ValueError(f"Unknown pickle format: expected list, got {type(data)}")

        # Remap table_ids if mapping provided
        if table_id_mapping:
            remapped = {}
            for tid, emb in embeddings.items():
                canonical = table_id_mapping.get(tid, tid)
                remapped[canonical] = emb
            embeddings = remapped

        return embeddings

    else:
        raise ValueError(f"Unsupported file format: {path}. Use .pkl files.")


def build_csv_to_table_id_mapping(table_id_to_csv_path: str) -> Dict[str, str]:
    """
    Build mapping from CSV basename (used in pkl table_id) to canonical table_id.

    The unified pkl files use CSV filenames as table_id (e.g. 'One_Tree_Hill_(TV_series)_HASH'),
    while ground truth uses the original table_id with spaces ('One Tree Hill (TV series)_HASH').
    This function builds the reverse mapping from table_id_to_csv.json.

    Args:
        table_id_to_csv_path: Path to table_id_to_csv.json
            Format: {"canonical_table_id": "csv_filename.csv", ...}

    Returns:
        Dict mapping csv_basename -> canonical_table_id
    """
    with open(table_id_to_csv_path, 'r') as f:
        tid_to_csv = json.load(f)

    csv_to_tid = {}
    for tid, csv_name in tid_to_csv.items():
        basename = csv_name[:-4] if csv_name.endswith('.csv') else csv_name
        csv_to_tid[basename] = tid

    return csv_to_tid


def load_query_embeddings(pkl_path: str) -> Dict[str, np.ndarray]:
    """
    Load query embeddings from text embedding pkl.

    Expected format: list of {"text_id": question_id, "text": ..., "embedding": array}

    Args:
        pkl_path: Path to query embeddings .pkl file

    Returns:
        Dict mapping question_id to embedding array
    """
    with open(pkl_path, 'rb') as f:
        data = load_pickle(f)

    return {entry['text_id']: np.array(entry['embedding'], dtype=np.float32) for entry in data}


def load_training_data(questions_path: str) -> List[Dict]:
    """
    Load training data from questions JSON file.

    Args:
        questions_path: Path to questions.json

    Returns:
        List of training samples with question_id and table_id
    """
    with open(questions_path, 'r') as f:
        return json.load(f)


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloader(
    table_embeddings: Dict[str, np.ndarray],
    query_embeddings: Dict[str, np.ndarray],
    training_data: List[Dict],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    use_hard_negatives: bool = False,
    hard_negative_ids: Optional[Dict[str, List[str]]] = None,
    num_hard_negatives: int = 5,
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        table_embeddings: Dict mapping table_id to embedding
        query_embeddings: Dict mapping question_id to embedding
        training_data: List of training samples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        use_hard_negatives: Whether to use hard negatives
        hard_negative_ids: Pre-computed hard negatives per question
        num_hard_negatives: Number of hard negatives per sample

    Returns:
        DataLoader instance
    """
    dataset = EmbeddingRetrievalDataset(
        table_embeddings=table_embeddings,
        query_embeddings=query_embeddings,
        training_data=training_data,
        num_hard_negatives=num_hard_negatives if use_hard_negatives else 0,
        hard_negative_ids=hard_negative_ids,
    )

    collate_fn = collate_with_hard_negatives if use_hard_negatives else collate_in_batch_only

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# =============================================================================
# Table/Question Loading Functions (from dataset.py)
# =============================================================================

def load_tables(path: str) -> List[Dict]:
    """
    Load tables from JSON file.

    Expected format:
    [
        {
            "table_id": "table_123",
            "title": "Table Title",
            "header": ["Col1", "Col2", ...],
            "rows": [["val1", "val2", ...], ...]
        },
        ...
    ]

    Args:
        path: Path to tables JSON file

    Returns:
        List of table dicts
    """
    with open(path, 'r', encoding='utf-8') as f:
        tables = json.load(f)

    print(f"Loaded {len(tables)} tables from {path}")

    # Validate format
    if tables:
        sample = tables[0]
        required_keys = ['table_id', 'header', 'rows']
        for key in required_keys:
            if key not in sample:
                print(f"Warning: Table missing key '{key}'")

    return tables


def load_questions(path: str) -> Tuple[List[Dict], List[str]]:
    """
    Load questions from JSON file.

    Expected format:
    [
        {
            "question_id": "q_123",
            "question": "What is...?",
            "table_id": "table_456"
        },
        ...
    ]

    Args:
        path: Path to questions JSON file

    Returns:
        Tuple of (questions list, ground truth table IDs)
    """
    with open(path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {path}")

    # Extract ground truth
    ground_truth = [q['table_id'] for q in questions]

    return questions, ground_truth


# =============================================================================
# ID2Table Mapping Functions
# =============================================================================

def build_id2table_mapping(tables: List[Dict]) -> Dict[int, str]:
    """
    Build mapping from index to table_id.

    Args:
        tables: List of table dicts

    Returns:
        Dict mapping index to table_id
    """
    return {i: t['table_id'] for i, t in enumerate(tables)}


def save_id2table_mapping(mapping: Dict[int, str], path: str):
    """Save id2table mapping to JSON."""
    # Convert int keys to strings for JSON serialization
    str_mapping = {str(k): v for k, v in mapping.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(str_mapping, f, indent=2)
    print(f"Saved id2table mapping to {path}")


def load_id2table_mapping(path: str) -> Dict[int, str]:
    """Load id2table mapping from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        str_mapping = json.load(f)
    # Convert string keys back to int
    return {int(k): v for k, v in str_mapping.items()}


# =============================================================================
# Curated Dataset (NQT-Retrieval Style)
# =============================================================================

class CuratedEmbeddingDataset(Dataset):
    """
    Dataset for training with curated hard negatives (NQT-Retrieval style).

    Each sample contains:
    - Query embedding
    - Positive table embeddings (from positive_ctxs)
    - Hard negative table embeddings (from hard_negative_ctxs)
    - Other negative table embeddings (from negative_ctxs)

    This matches the NQT-Retrieval data format where hard negatives are
    pre-mined from full corpus retrieval.
    """

    def __init__(
        self,
        table_embeddings: Dict[str, np.ndarray],  # table_id -> embedding
        query_embeddings: Dict[str, np.ndarray],  # question_id -> embedding
        curated_data: List[Dict],  # Curated format with positive_ctxs, hard_negative_ctxs
        num_hard_negatives: int = 1,
        num_other_negatives: int = 0,
        shuffle_positives: bool = False,
        shuffle_negatives: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            table_embeddings: Dict mapping table_id to embedding array
            query_embeddings: Dict mapping question_id to embedding array
            curated_data: List of curated samples with positive_ctxs, hard_negative_ctxs
            num_hard_negatives: Number of hard negatives per sample (default 1, matching NQT)
            num_other_negatives: Number of other negatives per sample (default 0)
            shuffle_positives: Whether to shuffle positive contexts
            shuffle_negatives: Whether to shuffle negative contexts
        """
        self.table_embeddings = table_embeddings
        self.query_embeddings = query_embeddings
        self.num_hard_negatives = num_hard_negatives
        self.num_other_negatives = num_other_negatives
        self.shuffle_positives = shuffle_positives
        self.shuffle_negatives = shuffle_negatives

        # Filter and validate samples
        self.valid_samples = []
        skipped = {'no_query': 0, 'no_positive': 0, 'no_hard_neg': 0}

        for sample in curated_data:
            q_id = sample.get('question_id')

            # Check query embedding exists
            if q_id not in query_embeddings:
                skipped['no_query'] += 1
                continue

            # Check at least one positive exists with embedding
            positive_ctxs = sample.get('positive_ctxs', [])
            valid_positives = [
                ctx for ctx in positive_ctxs
                if ctx.get('table_id') in table_embeddings
            ]
            if not valid_positives:
                skipped['no_positive'] += 1
                continue

            # Check hard negatives exist (required for this dataset)
            hard_neg_ctxs = sample.get('hard_negative_ctxs', [])
            valid_hard_negs = [
                ctx for ctx in hard_neg_ctxs
                if ctx.get('table_id') in table_embeddings
            ]
            if not valid_hard_negs:
                skipped['no_hard_neg'] += 1
                continue

            # Store validated sample with filtered contexts
            validated_sample = {
                'question_id': q_id,
                'question': sample.get('question', ''),
                'positive_ctxs': valid_positives,
                'hard_negative_ctxs': valid_hard_negs,
                'negative_ctxs': [
                    ctx for ctx in sample.get('negative_ctxs', [])
                    if ctx.get('table_id') in table_embeddings
                ],
            }
            self.valid_samples.append(validated_sample)

        print(f"CuratedEmbeddingDataset: {len(self.valid_samples)}/{len(curated_data)} valid samples")
        if any(skipped.values()):
            print(f"  Skipped: {skipped}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        import random

        sample = self.valid_samples[idx]
        q_id = sample['question_id']

        # Get query embedding
        query_emb = self.query_embeddings[q_id]

        # Get positive context (use first or shuffle)
        positive_ctxs = sample['positive_ctxs']
        if self.shuffle_positives and len(positive_ctxs) > 1:
            positive_ctx = random.choice(positive_ctxs)
        else:
            positive_ctx = positive_ctxs[0]
        positive_table_id = positive_ctx['table_id']
        positive_emb = self.table_embeddings[positive_table_id]

        # Get hard negatives (copy to avoid mutating stored data)
        hard_neg_ctxs = list(sample['hard_negative_ctxs'])
        if self.shuffle_negatives:
            random.shuffle(hard_neg_ctxs)
        hard_neg_ctxs = hard_neg_ctxs[:self.num_hard_negatives]

        hard_neg_embs = []
        hard_neg_ids = []
        for ctx in hard_neg_ctxs:
            table_id = ctx['table_id']
            if table_id != positive_table_id:
                hard_neg_embs.append(self.table_embeddings[table_id])
                hard_neg_ids.append(table_id)

        # Get other negatives (copy to avoid mutating stored data)
        other_neg_ctxs = list(sample['negative_ctxs'])
        if self.shuffle_negatives:
            random.shuffle(other_neg_ctxs)
        other_neg_ctxs = other_neg_ctxs[:self.num_other_negatives]

        other_neg_embs = []
        other_neg_ids = []
        for ctx in other_neg_ctxs:
            table_id = ctx['table_id']
            if table_id != positive_table_id and table_id not in hard_neg_ids:
                other_neg_embs.append(self.table_embeddings[table_id])
                other_neg_ids.append(table_id)

        return {
            'query_emb': torch.tensor(query_emb, dtype=torch.float32),
            'positive_emb': torch.tensor(positive_emb, dtype=torch.float32),
            'hard_neg_embs': [torch.tensor(e, dtype=torch.float32) for e in hard_neg_embs],
            'other_neg_embs': [torch.tensor(e, dtype=torch.float32) for e in other_neg_embs],
            'question_id': q_id,
            'table_id': positive_table_id,
            'hard_neg_ids': hard_neg_ids,
            'other_neg_ids': other_neg_ids,
        }


def collate_nqt_style(batch: List[Dict]) -> Dict:
    """
    Collate function matching NQT-Retrieval style.

    For each sample in batch:
    - 1 positive table
    - N hard negatives (from curated data)
    - M other negatives (from curated data)
    - In-batch negatives (other samples' contexts)

    The context tensor contains all contexts in order:
    [pos_0, hard_neg_0_0, ..., pos_1, hard_neg_1_0, ..., ...]

    Returns:
        Dict with:
        - query_embs: [batch_size, dim]
        - context_embs: [total_contexts, dim]
        - positive_idx: [batch_size] - index of positive for each query
        - hard_neg_idx: [[indices], ...] - indices of hard negatives for each query
    """
    batch_size = len(batch)

    # Stack query embeddings
    query_embs = torch.stack([item['query_emb'] for item in batch])

    # Collect all context embeddings
    context_embs = []
    positive_idx = []
    hard_neg_idx = []

    for i, item in enumerate(batch):
        current_pos_idx = len(context_embs)

        # Add positive
        context_embs.append(item['positive_emb'])
        positive_idx.append(current_pos_idx)

        # Add hard negatives
        item_hard_neg_idx = []
        for neg_emb in item['hard_neg_embs']:
            item_hard_neg_idx.append(len(context_embs))
            context_embs.append(neg_emb)

        # Add other negatives
        for neg_emb in item['other_neg_embs']:
            context_embs.append(neg_emb)

        hard_neg_idx.append(item_hard_neg_idx)

    context_embs = torch.stack(context_embs)

    return {
        'query_embs': query_embs,
        'context_embs': context_embs,
        'positive_idx': positive_idx,
        'hard_neg_idx': hard_neg_idx,
        'question_ids': [item['question_id'] for item in batch],
        'table_ids': [item['table_id'] for item in batch],
    }


def create_curated_dataloader(
    table_embeddings: Dict[str, np.ndarray],
    query_embeddings: Dict[str, np.ndarray],
    curated_data: List[Dict],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    num_hard_negatives: int = 1,
    num_other_negatives: int = 0,
    shuffle_positives: bool = False,
    shuffle_negatives: bool = True,
) -> DataLoader:
    """
    Create DataLoader for curated training data (NQT-Retrieval style).

    Args:
        table_embeddings: Dict mapping table_id to embedding
        query_embeddings: Dict mapping question_id to embedding
        curated_data: Curated training data with hard negatives
        batch_size: Batch size (NQT uses 8)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        num_hard_negatives: Hard negatives per sample (NQT uses 1)
        num_other_negatives: Other negatives per sample (NQT uses 0)
        shuffle_positives: Shuffle positive contexts
        shuffle_negatives: Shuffle negative contexts

    Returns:
        DataLoader instance
    """
    dataset = CuratedEmbeddingDataset(
        table_embeddings=table_embeddings,
        query_embeddings=query_embeddings,
        curated_data=curated_data,
        num_hard_negatives=num_hard_negatives,
        num_other_negatives=num_other_negatives,
        shuffle_positives=shuffle_positives,
        shuffle_negatives=shuffle_negatives,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_nqt_style,
    )


def load_curated_data(path: str) -> List[Dict]:
    """
    Load curated training data from JSON file.

    Expected format (NQT-Retrieval style):
    [
        {
            "question_id": "q_123",
            "question": "What is...?",
            "answers": ["answer1", ...],
            "table_id": "table_456",
            "positive_ctxs": [{"table_id": ..., "score": ...}, ...],
            "hard_negative_ctxs": [{"table_id": ..., "score": ...}, ...],
            "negative_ctxs": [{"table_id": ..., "score": ...}, ...]
        },
        ...
    ]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} curated samples from {path}")

    # Validate format
    if data:
        sample = data[0]
        required = ['question_id', 'positive_ctxs', 'hard_negative_ctxs']
        for key in required:
            if key not in sample:
                print(f"Warning: Curated data missing key '{key}'")

    return data
