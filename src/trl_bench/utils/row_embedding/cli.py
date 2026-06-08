"""
Shared CLI utilities for row-level embedding generation.

Provides standardized argument parsing and data loading intended to work
with row-level models (SCARF, SubTab, VIME, DAE, TabPFN, TabICL).
Note: TabPFN and TabICL currently use their own CLI parsing.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Container for train/test data split."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_labels: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    label_columns: Optional[List[str]] = None
    feature_columns: List[str] = None
    data_source: str = 'pre-split'  # 'pre-split', 'single-csv', or 'folder'
    split_ratio: Optional[float] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = []


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard arguments that all row-level models should support.

    This provides a consistent CLI interface across all embedding generation scripts.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        The modified parser
    """
    # Data input (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data_dir', type=str, default=None,
        help='Directory with train.csv and test.csv')
    input_group.add_argument('--input', type=str, default=None,
        help='Single CSV file (will be split internally)')
    input_group.add_argument('--input_folder', type=str, default=None,
        help='Folder of CSV files (batch mode)')

    # Label handling
    parser.add_argument('--label_columns', type=str, nargs='*', default=None,
        help='Label columns to exclude from features (None = pure unsupervised)')

    # Split configuration (only for single CSV mode)
    parser.add_argument('--split_ratio', type=float, default=0.8,
        help='Train/test split ratio (default: 0.8, only for --input mode)')
    parser.add_argument('--random_seed', type=int, default=42,
        help='Random seed for splitting (default: 42)')

    return parser


def load_data_from_args(args: argparse.Namespace) -> DataSplit:
    """
    Load data based on CLI arguments.

    Supports three modes:
    1. Pre-split: --data_dir with train.csv and test.csv
    2. Single CSV: --input with a single file to be split
    3. Batch mode: --input_folder with multiple CSVs (not implemented yet)

    Args:
        args: Parsed command line arguments

    Returns:
        DataSplit containing train/test DataFrames and optional labels
    """
    if args.data_dir is not None:
        return _load_presplit_data(args)
    elif args.input is not None:
        return _load_single_csv(args)
    elif args.input_folder is not None:
        raise NotImplementedError("Batch mode (--input_folder) not yet implemented")
    else:
        raise ValueError("Must specify --data_dir, --input, or --input_folder")


def _load_presplit_data(args: argparse.Namespace) -> DataSplit:
    """Load pre-split train.csv and test.csv from data_dir."""
    train_file = os.path.join(args.data_dir, 'train.csv')
    test_file = os.path.join(args.data_dir, 'test.csv')

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    label_columns = getattr(args, 'label_columns', None) or []
    train_labels = None
    test_labels = None
    feature_columns = list(train_df.columns)

    if label_columns:
        missing = [c for c in label_columns if c not in train_df.columns]
        if missing:
            raise ValueError(
                f"Label columns {missing} not found in train.csv. "
                f"Available columns: {list(train_df.columns)}"
            )
        label_set = set(label_columns)
        feature_columns = [c for c in feature_columns if c not in label_set]

    return DataSplit(
        train_df=train_df,
        test_df=test_df,
        train_labels=train_labels,
        test_labels=test_labels,
        label_columns=label_columns or None,
        feature_columns=feature_columns,
        data_source='pre-split',
    )


def _load_single_csv(args: argparse.Namespace) -> DataSplit:
    """Load single CSV and split into train/test."""
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    label_columns = getattr(args, 'label_columns', None) or []
    split_ratio = getattr(args, 'split_ratio', 0.8)
    random_seed = getattr(args, 'random_seed', 42)

    feature_columns = list(df.columns)
    if label_columns:
        missing = [c for c in label_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Label columns {missing} not found in {args.input}. "
                f"Available columns: {list(df.columns)}"
            )
        label_set = set(label_columns)
        feature_columns = [c for c in feature_columns if c not in label_set]

    # Stratified split when exactly one label column, otherwise random split
    if len(label_columns) == 1:
        train_df, test_df = train_test_split(
            df,
            train_size=split_ratio,
            random_state=random_seed,
            stratify=df[label_columns[0]]
        )
    else:
        train_df, test_df = train_test_split(
            df,
            train_size=split_ratio,
            random_state=random_seed,
        )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return DataSplit(
        train_df=train_df,
        test_df=test_df,
        train_labels=None,
        test_labels=None,
        label_columns=label_columns or None,
        feature_columns=feature_columns,
        data_source='single-csv',
        split_ratio=split_ratio,
        random_seed=random_seed,
    )


def get_generation_config(args: argparse.Namespace, data_split: DataSplit) -> dict:
    """
    Create generation config dict from args for metadata.

    Args:
        args: Parsed CLI arguments
        data_split: Data split information

    Returns:
        Dict with generation configuration
    """
    config = {
        'data_source': data_split.data_source,
        'batch_size': getattr(args, 'batch_size', None),
    }

    if data_split.data_source == 'pre-split':
        config['data_dir'] = getattr(args, 'data_dir', None)
    elif data_split.data_source == 'single-csv':
        config['input_file'] = getattr(args, 'input', None)
        config['split_ratio'] = data_split.split_ratio
        config['random_seed'] = data_split.random_seed

    return config
