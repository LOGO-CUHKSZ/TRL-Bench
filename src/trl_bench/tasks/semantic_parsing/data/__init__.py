"""Data utilities for semantic parsing."""

from .data_utils import (
    load_jsonl,
    load_json,
    save_json,
    load_examples_from_shard,
)
from .schema import (
    Example,
    Table,
    Column,
)


def load_embeddings(column_pkl_path: str, question_pkl_paths: list):
    """Load pre-computed embeddings from pkl files.

    Args:
        column_pkl_path: Path to column embeddings pkl file
        question_pkl_paths: List of paths to question embeddings pkl files

    Returns:
        Tuple of (column_cache, question_cache) dicts
    """
    import pickle
    import numpy as np

    with open(column_pkl_path, 'rb') as f:
        col_data = pickle.load(f)
    column_cache = {}
    for entry in col_data:
        table_id = entry['table_id']
        col_embs = entry['column_embeddings']
        column_cache[table_id] = np.stack([col_embs[i] for i in range(len(col_embs))], axis=0)

    question_cache = {}
    for qpath in question_pkl_paths:
        with open(qpath, 'rb') as f:
            q_data = pickle.load(f)
        for entry in q_data:
            question_cache[entry['text_id']] = entry['embedding']

    return column_cache, question_cache


__all__ = [
    'load_jsonl',
    'load_json',
    'save_json',
    'load_examples_from_shard',
    'load_embeddings',
    'Example',
    'Table',
    'Column',
]
