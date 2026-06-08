"""
Row-level embedding utilities.

Provides shared CLI argument parsing and data loading utilities
for row-level embedding generation across all models.
"""

from .cli import (
    add_common_args,
    load_data_from_args,
    DataSplit,
)

from .directory import (
    discover_csv_files,
    check_checkpoint_complete,
    clean_partial_checkpoint,
    build_table_result,
    save_aggregate_pickle,
    register_save_on_signal,
    load_existing_results,
    get_completed_table_ids,
    cleanup_checkpoints,
    preprocess_table,
    train_raw_loop,
    save_model_checkpoint,
    load_model_from_checkpoint,
)

__all__ = [
    'add_common_args',
    'load_data_from_args',
    'DataSplit',
    'discover_csv_files',
    'check_checkpoint_complete',
    'clean_partial_checkpoint',
    'build_table_result',
    'save_aggregate_pickle',
    'register_save_on_signal',
    'load_existing_results',
    'get_completed_table_ids',
    'cleanup_checkpoints',
    'preprocess_table',
    'train_raw_loop',
    'save_model_checkpoint',
    'load_model_from_checkpoint',
]
