"""
Data utilities for TabFact downstream task.

Provides functions for:
- Loading TabFact examples and tables
- Loading pre-computed embeddings
- Creating PyTorch datasets for training
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def load_tabfact_examples(data_dir: str, split: str = 'train') -> List[Dict]:
    """
    Load TabFact examples from JSONL file.

    Args:
        data_dir: Directory containing TabFact dataset
        split: One of 'train', 'validation', 'test'

    Returns:
        List of example dicts with keys: id, table_id, statement, label
    """
    filepath = Path(data_dir) / f"{split}.jsonl"
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def load_table(data_dir: str, table_id: str) -> pd.DataFrame:
    """
    Load a table by its ID.

    Args:
        data_dir: Directory containing TabFact dataset
        table_id: Table identifier

    Returns:
        pandas DataFrame of the table
    """
    table_path = Path(data_dir) / "tables" / f"{table_id}.csv"
    return pd.read_csv(table_path)


def load_embeddings(embeddings_file: str) -> Dict[str, np.ndarray]:
    """
    Load pre-computed embeddings from pickle file.

    Args:
        embeddings_file: Path to pickle file with embeddings

    Returns:
        Dict mapping example_id to embedding dict
    """
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


class TabFactDataset(Dataset):
    """
    PyTorch Dataset for TabFact classification.

    Loads pre-computed embeddings and labels for training.
    """

    def __init__(
        self,
        embeddings_file: str,
        data_dir: str,
        split: str = 'train',
        embedding_key: str = 'cls_embedding',
    ):
        """
        Initialize dataset.

        Args:
            embeddings_file: Path to pickle file with embeddings
            data_dir: Directory containing TabFact dataset
            split: One of 'train', 'validation', 'test'
            embedding_key: Key in embedding dict to use (default: 'cls_embedding')
        """
        self.embedding_key = embedding_key

        # Load examples
        self.examples = load_tabfact_examples(data_dir, split)
        print(f"Loaded {len(self.examples)} {split} examples")

        # Load embeddings
        self.embeddings = load_embeddings(embeddings_file)
        print(f"Loaded embeddings for {len(self.embeddings)} examples")

        # Filter to examples with embeddings
        self.valid_examples = []
        missing_count = 0
        for ex in self.examples:
            if ex['id'] in self.embeddings:
                self.valid_examples.append(ex)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count} examples missing embeddings")

        print(f"Using {len(self.valid_examples)} examples with embeddings")

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.valid_examples[idx]
        emb_dict = self.embeddings[example['id']]

        # Get embedding
        embedding = emb_dict[self.embedding_key]
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        # Get label
        label = torch.tensor(example['label'], dtype=torch.long)

        return {
            'embedding': embedding,
            'label': label,
            'id': example['id'],
        }


class TabFactEmbeddingDataset(Dataset):
    """
    Dataset for generating embeddings (no labels needed).

    Used for batch embedding generation.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_examples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing TabFact dataset
            split: One of 'train', 'validation', 'test'
            max_examples: Maximum number of examples to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.examples = load_tabfact_examples(data_dir, split)

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

        print(f"Loaded {len(self.examples)} {split} examples for embedding generation")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Load table
        table_path = self.data_dir / "tables" / f"{example['table_id']}.csv"

        return {
            'id': example['id'],
            'table_id': example['table_id'],
            'table_path': str(table_path),
            'statement': example['statement'],
            'label': example['label'],
        }


def collate_embeddings(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for TabFactDataset.

    Args:
        batch: List of example dicts from dataset

    Returns:
        Batched tensors
    """
    embeddings = torch.stack([item['embedding'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    ids = [item['id'] for item in batch]

    return {
        'embeddings': embeddings,
        'labels': labels,
        'ids': ids,
    }


def get_label_distribution(data_dir: str, split: str = 'train') -> Dict[str, int]:
    """
    Get label distribution for a split.

    Args:
        data_dir: Directory containing TabFact dataset
        split: One of 'train', 'validation', 'test'

    Returns:
        Dict with counts of each label
    """
    examples = load_tabfact_examples(data_dir, split)
    entailed = sum(1 for ex in examples if ex['label'] == 1)
    refuted = len(examples) - entailed

    return {
        'total': len(examples),
        'entailed': entailed,
        'refuted': refuted,
        'entailed_pct': 100 * entailed / len(examples),
        'refuted_pct': 100 * refuted / len(examples),
    }
