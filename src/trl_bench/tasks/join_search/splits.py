"""Canonical query-role-disjoint split generation for join search.

The learned-projection join search probe (``join_search_learned``) trains on a
fixed train/test split of the OpenData join queries. The split is a
deterministic function of the ground truth, a curated query list, and
``(seed=42, train_ratio=0.2)``:

  1. ``preprocess_gt`` — basename table ids, drop self-pairs, dedup, and
     intersect with the curated query list (this is what reduces e.g. OpenData
     from 3350 raw query keys to the canonical 3103).
  2. ``split_by_query`` — partition the unique query keys with
     ``random.Random(seed)`` so no query key appears in more than one split.

``preprocess_gt`` and ``split_by_query`` are kept identical to the copies in
``run_learned_search.py`` so the regenerated split stays consistent (OpenData ->
620 train / 2483 test queries). ``run_learned_search.py`` should import
them from here to avoid drift.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Union

import pandas as pd

_PathLike = Union[str, "os.PathLike[str]"]


def preprocess_gt(gt_path: _PathLike, query_list_path: _PathLike) -> pd.DataFrame:
    """Load and filter GT to match evaluation semantics in run_search_and_evaluate.py."""
    gt_df = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    gt_df['query_table'] = gt_df['query_table'].apply(os.path.basename)
    gt_df['candidate_table'] = gt_df['candidate_table'].apply(os.path.basename)

    # Remove self-table pairs
    gt_df = gt_df[gt_df['query_table'] != gt_df['candidate_table']].reset_index(drop=True)

    # Deduplicate
    gt_df = gt_df.drop_duplicates().reset_index(drop=True)

    # Intersect with query list
    query_list = pd.read_csv(query_list_path, dtype=str, keep_default_na=False)
    query_list['query_table'] = query_list['query_table'].apply(os.path.basename)
    valid_query_keys = set(zip(query_list['query_table'], query_list['query_column']))
    gt_df = gt_df[
        gt_df.apply(lambda r: (r['query_table'], r['query_column']) in valid_query_keys, axis=1)
    ].reset_index(drop=True)
    return gt_df


def split_by_query(gt_df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    """Split GT rows by query key so no query appears in multiple splits."""
    query_keys = sorted(set(zip(gt_df['query_table'], gt_df['query_column'])))
    rng = random.Random(seed)
    rng.shuffle(query_keys)

    n = len(query_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_keys = set(query_keys[:n_train])
    val_keys = set(query_keys[n_train:n_train + n_val])
    test_keys = set(query_keys[n_train + n_val:])

    def _filter(keys):
        mask = gt_df.apply(lambda r: (r['query_table'], r['query_column']) in keys, axis=1)
        return gt_df[mask].reset_index(drop=True)

    return (
        _filter(train_keys), _filter(val_keys), _filter(test_keys),
        sorted(train_keys), sorted(val_keys), sorted(test_keys),
    )


def generate_canonical_split(
    *,
    ground_truth_path: _PathLike,
    query_list_path: _PathLike,
    out_dir: _PathLike,
    seed: int = 42,
    train_ratio: float = 0.2,
    val_ratio: float = 0.0,
) -> dict:
    """Reproduce the canonical fixed join-search split into ``out_dir``.

    Writes ``train_queries.csv`` / ``test_queries.csv`` (``query_table,
    query_column``), ``test_gt.csv`` (the preprocessed GT rows for test
    queries), and ``split_info.json``. Returns the split_info dict.

    ``query_list_path`` is the *curated* query list (the artifact that pins the
    canonical query set); the produced split is byte-stable for a fixed
    ``(ground_truth, query_list, seed, train_ratio)``.
    """
    gt_df = preprocess_gt(ground_truth_path, query_list_path)
    train_df, _val_df, test_df, train_keys, _val_keys, test_keys = split_by_query(
        gt_df, train_ratio, val_ratio, seed,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_keys, columns=['query_table', 'query_column']).to_csv(
        out / 'train_queries.csv', index=False,
    )
    pd.DataFrame(test_keys, columns=['query_table', 'query_column']).to_csv(
        out / 'test_queries.csv', index=False,
    )
    test_df.to_csv(out / 'test_gt.csv', index=False)

    info = {
        'description': 'Fixed 20/80 query-role-disjoint split for join search probe evaluation',
        'train_queries': len(train_keys),
        'test_queries': len(test_keys),
        'train_pairs': len(train_df),
        'test_pairs': len(test_df),
        'seed': seed,
        'train_ratio': train_ratio,
    }
    with open(out / 'split_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    return info
