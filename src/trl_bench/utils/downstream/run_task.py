"""
Task-agnostic classifier/regressor training and evaluation.

This script takes pre-extracted embeddings and trains a model for any task
defined by a labels file. Supports both classification and regression tasks.

Usage:
    # Classification task
    python utils/downstream/run_task.py \
        --embeddings datalake_embeddings.pkl \
        --labels spider_join/labels.json \
        --task_name spider_join \
        --task_type classification \
        --num_labels 2 \
        --output_dir results/spider_join

    # Regression task
    python utils/downstream/run_task.py \
        --embeddings datalake_embeddings.pkl \
        --labels wiki_containment/labels.json \
        --task_name wiki_containment \
        --task_type regression \
        --num_labels 1 \
        --output_dir results/wiki_containment

    # Compare different embedding sources on same task
    python utils/downstream/run_task.py --embeddings embeddings_pretrained.pkl --labels task.json
    python utils/downstream/run_task.py --embeddings embeddings_raw_bert.pkl --labels task.json
"""

import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path

# Ensure repo root is on sys.path so `from utils.downstream.X` resolves
# when this script is invoked directly (python utils/downstream/run_task.py).
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from trl_bench.utils.pickle_compat import load_pickle


def combine_embeddings(emb1, emb2, method='concat'):
    """Combine two embeddings."""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    if method == 'concat':
        return np.concatenate([emb1, emb2])
    elif method == 'add':
        return emb1 + emb2
    elif method == 'multiply':
        return emb1 * emb2
    elif method == 'diff':
        return np.abs(emb1 - emb2)
    else:
        raise ValueError(f"Unknown combination method: {method}")


def _find_column_key(column_embedding_dict, col_id):
    """
    Find the column key in the embedding dict, handling int/string mismatches.

    Args:
        column_embedding_dict: Dict mapping column IDs to embeddings
        col_id: Column ID from labels (could be int or string)

    Returns:
        The matching key in the dict, or None if not found
    """
    # Try direct lookup
    if col_id in column_embedding_dict:
        return col_id

    # Try as integer
    try:
        int_key = int(col_id)
        if int_key in column_embedding_dict:
            return int_key
    except (ValueError, TypeError):
        pass

    # Try as string
    str_key = str(col_id)
    if str_key in column_embedding_dict:
        return str_key

    return None


def prepare_task_data(table_embeddings, labels, embedding_type='cls', combination_method='concat'):
    """
    Prepare training data for a task from individual table embeddings and labels.

    Args:
        table_embeddings: List of dicts with table embeddings
        labels: Labels dict with train/valid/test splits
        embedding_type: Which embedding to use:
            - 'cls': CLS token embedding (table-level)
            - 'table': Mean-pooled table embedding (table-level)
            - 'column_mean': Mean of all column embeddings (table-level)
            - 'token_mean': Mean of all non-padding token hidden states (table-level)
            - 'column': Specific column embeddings from labels (column-level)
        combination_method: How to combine pairs ('concat', 'add', 'multiply', 'diff')

    Returns:
        Dict with train/valid/test datasets ready for training
    """
    # For column-level embeddings, we need to store the full column_embedding dict
    use_column_embeddings = (embedding_type == 'column')

    # Create lookup: table filename -> embeddings
    table_to_emb = {}
    first_none_warning_shown = False

    for item in table_embeddings:
        # Prefer 'table_id' if available, otherwise extract from 'table' path
        table_key = item.get('table_id')
        if not table_key:
            table_name = item.get('table', '')
            table_key = table_name.split('/')[-1]

        # Normalize: strip extension to ensure consistent stem format
        # (some models include .csv in table_id, others don't)
        if table_key.endswith('.csv'):
            table_key = table_key[:-4]

        # Handle both 'column_embeddings' (unified) and 'column_embedding' (legacy)
        col_emb_data = item.get('column_embeddings') or item.get('column_embedding')

        table_emb_block = item.get('table_embedding')
        table_emb_dict = table_emb_block if isinstance(table_emb_block, dict) else None
        cls_embedding = item.get('cls_embedding')
        if cls_embedding is None and table_emb_dict is not None:
            cls_embedding = table_emb_dict.get('cls_embedding')
        if table_emb_dict is not None:
            table_embedding = table_emb_dict.get('table_embedding')
            column_mean = table_emb_dict.get('column_mean')
        elif table_emb_block is None:
            table_embedding = None
            column_mean = None
        else:
            raise TypeError(
                f"table_embedding must be a dict (v2.0 format) or None, "
                f"got {type(table_emb_block).__name__} for table '{table_key}'. "
                f"v1.0 format (raw array) is no longer supported — regenerate embeddings."
            )

        if use_column_embeddings:
            # Store the entire column_embedding dict for column-level lookups
            if col_emb_data is None:
                raise ValueError(
                    f"embedding_type='column' requested but column embeddings are None for table '{table_key}'.\n"
                    f"This model may not support column-level embeddings."
                )
            # Build a name-keyed dict so labels can look up by column name.
            # Embeddings use integer keys (0, 1, 2, ...) while labels may
            # reference columns by name ("Lineal", "Population", ...).
            # column_names maps index -> name, so we add name-keyed entries.
            col_names = item.get('column_names')
            enriched = dict(col_emb_data)  # keep original int keys
            if col_names:
                for idx, emb in col_emb_data.items():
                    try:
                        int_idx = int(idx)
                    except (ValueError, TypeError):
                        continue
                    if int_idx < len(col_names):
                        enriched[col_names[int_idx]] = emb
            table_to_emb[table_key] = {
                'column_embedding': enriched,
            }
        elif embedding_type == 'cls':
            if cls_embedding is None:
                if not first_none_warning_shown:
                    raise ValueError(
                        f"embedding_type='cls' requested but 'cls_embedding' is None for table '{table_key}'.\n"
                        f"This model may not support CLS embeddings.\n"
                        f"Try using '--embedding_type column_mean' or '--embedding_type table' instead."
                    )
            table_to_emb[table_key] = cls_embedding
        elif embedding_type == 'table':
            if table_embedding is None:
                if not first_none_warning_shown:
                    raise ValueError(
                        f"embedding_type='table' requested but 'table_embedding' is None for table '{table_key}'.\n"
                        f"This model may not support table-level embeddings.\n"
                        f"Try using '--embedding_type column_mean' instead."
                    )
            table_to_emb[table_key] = table_embedding
        elif embedding_type == 'column_mean':
            if col_emb_data is not None:
                col_embs = list(col_emb_data.values())
                table_to_emb[table_key] = np.mean(col_embs, axis=0).tolist()
            elif column_mean is not None:
                table_to_emb[table_key] = column_mean
            else:
                raise ValueError(
                    f"embedding_type='column_mean' requested but no usable embeddings found for table '{table_key}'.\n"
                    f"This model may not support column embeddings."
                )
        elif embedding_type == 'token_mean':
            token_mean = table_emb_dict.get('token_mean') if table_emb_dict is not None else None
            if token_mean is None:
                raise ValueError(
                    f"embedding_type='token_mean' requested but 'token_mean' is None for table '{table_key}'.\n"
                    f"This model may not support token_mean embeddings.\n"
                    f"Try using '--embedding_type column_mean' or '--embedding_type cls' instead."
                )
            table_to_emb[table_key] = token_mean
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    print(f"\n📊 Loaded embeddings for {len(table_to_emb)} unique tables")
    if use_column_embeddings:
        print(f"   Using COLUMN-LEVEL embeddings (from join_col_table1/join_col_table2 in labels)")

    # Validate column embedding requirements upfront
    if use_column_embeddings:
        # Check first item in any split to see if column specs exist
        sample_item = None
        for split_name in ['train', 'valid', 'test']:
            if split_name in labels and len(labels[split_name]) > 0:
                sample_item = labels[split_name][0]
                break

        if sample_item is None:
            raise ValueError("No samples found in labels file")

        has_col_spec = 'join_col_table1' in sample_item or 'col1' in sample_item
        if not has_col_spec:
            raise ValueError(
                f"embedding_type='column' requires column specifications in labels file.\n"
                f"Labels must contain 'join_col_table1'/'join_col_table2' or 'col1'/'col2' fields.\n"
                f"Found keys: {list(sample_item.keys())}\n"
                f"This task only supports table-level embeddings: 'cls', 'table', or 'column_mean'."
            )

    # Create paired datasets
    task_data = {}
    stats = {'train': {'total': 0, 'skipped': 0},
             'valid': {'total': 0, 'skipped': 0},
             'test': {'total': 0, 'skipped': 0}}

    for split_name in ['train', 'valid', 'test']:
        if split_name not in labels:
            continue

        task_data[split_name] = []

        for item in labels[split_name]:
            stats[split_name]['total'] += 1

            table1_name = item['table1']['filename'].split('/')[-1]
            table2_name = item['table2']['filename'].split('/')[-1]
            # Normalize: strip file extensions to match table_id (which is stem without extension)
            for suffix in ['.gz', '.bz2', '.csv']:
                if table1_name.endswith(suffix):
                    table1_name = table1_name[:-len(suffix)]
                if table2_name.endswith(suffix):
                    table2_name = table2_name[:-len(suffix)]
            label = item['label']

            if table1_name not in table_to_emb or table2_name not in table_to_emb:
                stats[split_name]['skipped'] += 1
                continue

            if use_column_embeddings:
                # Extract specific column embeddings based on labels
                col1_id = item.get('join_col_table1', item.get('col1'))
                col2_id = item.get('join_col_table2', item.get('col2'))

                # Column IDs in embeddings are integers, labels may have strings
                # Try both string and int versions
                col1_key = _find_column_key(table_to_emb[table1_name]['column_embedding'], col1_id)
                col2_key = _find_column_key(table_to_emb[table2_name]['column_embedding'], col2_id)

                if col1_key is None:
                    raise ValueError(
                        f"Column {col1_id} not found in embeddings for table '{table1_name}'.\n"
                        f"Available columns: {list(table_to_emb[table1_name]['column_embedding'].keys())}"
                    )
                if col2_key is None:
                    raise ValueError(
                        f"Column {col2_id} not found in embeddings for table '{table2_name}'.\n"
                        f"Available columns: {list(table_to_emb[table2_name]['column_embedding'].keys())}"
                    )

                emb1 = table_to_emb[table1_name]['column_embedding'][col1_key]
                emb2 = table_to_emb[table2_name]['column_embedding'][col2_key]
            else:
                # Table-level embeddings
                emb1 = table_to_emb[table1_name]
                emb2 = table_to_emb[table2_name]

            combined = combine_embeddings(emb1, emb2, combination_method)

            task_data[split_name].append({
                'embedding': combined.tolist() if isinstance(combined, np.ndarray) else combined,
                'label': label,
                'split': split_name
            })

    # Print statistics
    print("\n" + "="*60)
    print("TASK DATA PREPARATION")
    print("="*60)
    print(f"Embedding type: {embedding_type}")
    for split in ['train', 'valid', 'test']:
        if split in stats:
            total = stats[split]['total']
            skipped = stats[split]['skipped']
            kept = total - skipped
            print(f"{split.upper()}: {kept}/{total} samples (skipped {skipped} due to missing tables)")
    print("="*60)

    return task_data


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""
    def __init__(self, embeddings_list, task_type='classification', num_labels=2):
        self.embeddings = [item['embedding'] for item in embeddings_list]
        self.labels = [item['label'] for item in embeddings_list]
        self.task_type = task_type
        self.num_labels = num_labels

        # Auto-detect multi-label classification
        self.is_multi_label = False
        if task_type == 'classification':
            # Check if any label is a list with multiple elements
            for label in self.labels:
                if isinstance(label, list) and len(label) > 1:
                    self.is_multi_label = True
                    break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)

        if self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        elif self.is_multi_label:
            # Convert list of label indices to multi-hot encoding
            label_indices = self.labels[idx] if isinstance(self.labels[idx], list) else [self.labels[idx]]
            multi_hot = torch.zeros(self.num_labels, dtype=torch.float32)
            for label_idx in label_indices:
                if 0 <= label_idx < self.num_labels:
                    multi_hot[label_idx] = 1.0
            label = multi_hot
        else:
            # Single-label classification
            label_val = self.labels[idx] if not isinstance(self.labels[idx], list) else self.labels[idx][0]
            label = torch.tensor(label_val, dtype=torch.long)

        return embedding, label


def _resolve_loss_type(task_type, is_multi_label):
    """Determine loss type from task type and multi-label flag."""
    if task_type == 'regression':
        return 'mse'
    elif is_multi_label:
        return 'bce_with_logits'
    else:
        return 'cross_entropy'


def _resolve_metrics(task_type, is_multi_label):
    """Determine evaluation metrics from task type and multi-label flag."""
    if task_type == 'regression':
        return ['mse', 'r2', 'mae', 'pearson_r', 'spearman_r']
    elif is_multi_label:
        return ['subset_accuracy', 'hamming_accuracy', 'micro_f1', 'macro_f1']
    else:
        return ['accuracy', 'weighted_f1', 'macro_f1', 'auroc', 'precision', 'recall']


def main():
    parser = ArgumentParser(description="Train classifier on task-specific paired embeddings")
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Pickle file with individual table embeddings')
    parser.add_argument('--labels', type=str, required=True,
                        help='Labels JSON file defining the task')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of the task (for logging)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    # Config-driven mode (optional)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides CLI defaults)')
    parser.add_argument('--override', type=str, action='append', default=[],
                        help='Config overrides (e.g., --override training.optimizer.lr=1e-4)')

    # Embedding combination
    parser.add_argument('--embedding_type', type=str, default='column_mean',
                        choices=['cls', 'table', 'column_mean', 'token_mean', 'column'],
                        help='Which embedding type to use: cls/table/column_mean/token_mean (table-level), '
                             'column (uses specific columns from join_col_table1/join_col_table2 in labels)')
    parser.add_argument('--combination_method', type=str, default='concat',
                        choices=['concat', 'add', 'multiply', 'diff'],
                        help='How to combine table pair embeddings')

    # Task configuration
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Task type: classification or regression')

    # Classifier architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for 2-layer MLP')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of output labels (2 for binary classification, 1 for regression)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--seed', '--random_seed', type=int, default=42, dest='seed',
                        help='Random seed for reproducibility (--random_seed is deprecated)')

    # Hardware
    parser.add_argument('--num_workers', type=int, default=0)
    # Kept for backward compatibility with shell wrappers (ignored)
    parser.add_argument('--accelerator', type=str, default='gpu',
                        help='(Ignored, kept for wrapper compat)')
    parser.add_argument('--devices', type=int, default=1,
                        help='(Ignored, kept for wrapper compat)')

    # Probe type
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'linear', 'dummy'],
                        help='Probe type: mlp (PyTorch MLP), linear (sklearn), or dummy (majority/mean baseline)')

    args = parser.parse_args()

    # Load embeddings
    print(f"\nLoading table embeddings from: {args.embeddings}")
    with open(args.embeddings, 'rb') as f:
        table_embeddings = load_pickle(f)
    print(f"   Loaded {len(table_embeddings)} table embeddings")

    # Load labels
    print(f"\nLoading task labels from: {args.labels}")
    with open(args.labels, 'r') as f:
        labels = json.load(f)

    # Prepare task data (pair embeddings according to labels)
    print(f"\nPreparing task data...")
    print(f"   Task: {args.task_name}")
    print(f"   Embedding type: {args.embedding_type}")
    print(f"   Combination method: {args.combination_method}")

    task_data = prepare_task_data(
        table_embeddings,
        labels,
        embedding_type=args.embedding_type,
        combination_method=args.combination_method
    )

    # Auto-detect input dimension
    if len(task_data['train']) > 0:
        input_dim = len(task_data['train'][0]['embedding'])
        print(f"\nDetected input dimension: {input_dim}")
    else:
        raise ValueError("No training data available!")

    # Z-score normalize embeddings and regression targets (MLP path only,
    # matching row_prediction behaviour). Scalers are fit on train split only.
    emb_scaler = None
    label_scaler = None
    if args.head_type == 'mlp':
        from sklearn.preprocessing import StandardScaler

        train_embs = np.array([item['embedding'] for item in task_data['train']], dtype=np.float32)
        emb_scaler = StandardScaler()
        emb_scaler.fit(train_embs)

        for split_name in ('train', 'valid', 'test'):
            for item in task_data.get(split_name, []):
                item['embedding'] = emb_scaler.transform(
                    np.array(item['embedding'], dtype=np.float32).reshape(1, -1)
                )[0]

        if args.task_type == 'regression':
            train_labels_raw = np.array(
                [item['label'] for item in task_data['train']], dtype=np.float32
            )
            label_scaler = StandardScaler()
            label_scaler.fit(train_labels_raw.reshape(-1, 1))

            for split_name in ('train', 'valid', 'test'):
                for item in task_data.get(split_name, []):
                    item['label'] = float(label_scaler.transform(
                        np.array([[item['label']]], dtype=np.float32)
                    )[0, 0])

            print(f"\n   Regression target z-score: mean={label_scaler.mean_[0]:.6f}, scale={label_scaler.scale_[0]:.6f}")

        print(f"   Embedding z-score: applied (fit on {len(train_embs)} train samples)")

    # Create datasets
    train_dataset = EmbeddingDataset(task_data.get('train', []), task_type=args.task_type, num_labels=args.num_labels)
    valid_dataset = EmbeddingDataset(task_data.get('valid', []), task_type=args.task_type, num_labels=args.num_labels)
    test_dataset = EmbeddingDataset(task_data.get('test', []), task_type=args.task_type, num_labels=args.num_labels)

    # Check if multi-label classification is detected
    is_multi_label = train_dataset.is_multi_label
    if is_multi_label:
        print(f"\nDetected multi-label classification (samples have variable-length labels)")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Build config (from YAML or CLI args) ---
    from trl_bench.utils.downstream.config import load_config

    loss_type = _resolve_loss_type(args.task_type, is_multi_label)
    eval_metrics = _resolve_metrics(args.task_type, is_multi_label)

    # Detect which CLI flags the user actually passed (vs argparse defaults).
    # Handles both --flag value and --flag=value syntax.
    _explicitly_passed = set()
    for token in sys.argv[1:]:
        if token.startswith('--'):
            flag = token.lstrip('-').split('=', 1)[0].replace('-', '_')
            _explicitly_passed.add(flag)

    # Build overrides only from explicitly-passed CLI flags
    _cli_map = {
        'task_name':          f'task_name={args.task_name}',
        'task_type':          f'task_type={args.task_type}',
        'hidden_dim':         f'head.hidden_dim={args.hidden_dim}',
        'num_labels':         f'head.output_dim={args.num_labels}',
        'dropout_prob':       f'head.dropout={args.dropout_prob}',
        'combination_method': f'training.combination_method={args.combination_method}',
        'learning_rate':      f'training.optimizer.lr={args.learning_rate}',
        'batch_size':         f'training.batch_size={args.batch_size}',
        'max_epochs':         f'training.max_epochs={args.max_epochs}',
        'seed':               f'training.seed={args.seed}',
        'random_seed':        f'training.seed={args.seed}',
    }

    if args.config:
        # YAML is the source of truth; only explicitly-passed CLI flags override it
        user_cli_overrides = [v for k, v in _cli_map.items() if k in _explicitly_passed]
        # Always pass loss_type since it's auto-detected from data, not a default
        user_cli_overrides.append(f'training.loss.type={loss_type}')
        cfg = load_config(args.config, overrides=user_cli_overrides + args.override)
    else:
        # No YAML — use all CLI values (including defaults) as the config source
        all_cli_overrides = list(_cli_map.values()) + [f'training.loss.type={loss_type}']
        cfg = load_config(overrides=all_cli_overrides + args.override)

    # --- Train with Trainer ---
    from trl_bench.utils.downstream.trainer import Trainer as TaskTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set evaluation metrics in config
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.evaluation.metrics = eval_metrics
    OmegaConf.set_struct(cfg, True)

    if args.head_type == 'linear':
        # ── Linear probe path (sklearn) ──
        from trl_bench.utils.downstream.linear_probe import LinearProbeRunner

        def _extract_arrays(split_data, multi_label, num_labels):
            emb = np.array([item['embedding'] for item in split_data])
            if multi_label:
                # Encode as multi-hot (same as EmbeddingDataset.__getitem__)
                labs = np.zeros((len(split_data), num_labels), dtype=np.float32)
                for i, item in enumerate(split_data):
                    indices = item['label'] if isinstance(item['label'], list) else [item['label']]
                    for idx in indices:
                        if 0 <= idx < num_labels:
                            labs[i, idx] = 1.0
            else:
                labs = np.array([
                    item['label'] if not isinstance(item['label'], list) else item['label'][0]
                    for item in split_data
                ])
            return emb, labs

        train_emb_np, train_lab_np = _extract_arrays(
            task_data.get('train', []), is_multi_label, args.num_labels)
        val_data = task_data.get('valid', [])
        val_emb_np, val_lab_np = (
            _extract_arrays(val_data, is_multi_label, args.num_labels) if val_data
            else (None, None))
        test_emb_np, test_lab_np = _extract_arrays(
            task_data.get('test', []), is_multi_label, args.num_labels)

        print(f"\nRunning linear probe (sklearn)...")
        print(f"   Task type: {args.task_type}")
        if is_multi_label:
            print(f"   Multi-label: Yes")

        runner = LinearProbeRunner(cfg)
        raw_test_results = runner.run(
            train_emb=train_emb_np,
            train_labels=train_lab_np,
            test_emb=test_emb_np,
            test_labels=test_lab_np,
            task_type=args.task_type,
            metric_names=eval_metrics,
            val_emb=val_emb_np,
            val_labels=val_lab_np,
            multi_label=is_multi_label,
            threshold=float(cfg.evaluation.threshold),
        )
        # Strip test_ prefix for aggregation compatibility
        test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        results = {
            'task_name': args.task_name,
            'task_type': args.task_type,
            'head_type': 'linear',
            'seed': args.seed,
            'embedding_type': args.embedding_type,
            'combination_method': args.combination_method,
            'input_dim': input_dim,
            'num_labels': args.num_labels,
            'test_results': test_results,
            'data_stats': {
                'train': len(task_data.get('train', [])),
                'valid': len(task_data.get('valid', [])),
                'test': len(task_data.get('test', [])),
            }
        }

    elif args.head_type == 'dummy':
        # ── Dummy baseline path (label statistics only) ──
        from trl_bench.utils.downstream.dummy_probe import DummyProbeRunner

        def _extract_labels(split_data, multi_label, num_labels):
            if multi_label:
                labs = np.zeros((len(split_data), num_labels), dtype=np.float32)
                for i, item in enumerate(split_data):
                    indices = item['label'] if isinstance(item['label'], list) else [item['label']]
                    for idx in indices:
                        if 0 <= idx < num_labels:
                            labs[i, idx] = 1.0
            else:
                labs = np.array([
                    item['label'] if not isinstance(item['label'], list) else item['label'][0]
                    for item in split_data
                ])
            return labs

        train_lab_np = _extract_labels(
            task_data.get('train', []), is_multi_label, args.num_labels)
        test_lab_np = _extract_labels(
            task_data.get('test', []), is_multi_label, args.num_labels)

        print(f"\nRunning dummy baseline (label statistics only)...")
        print(f"   Task type: {args.task_type}")
        if is_multi_label:
            print(f"   Multi-label: Yes")

        runner = DummyProbeRunner()
        raw_test_results = runner.run(
            train_labels=train_lab_np,
            test_labels=test_lab_np,
            task_type=args.task_type,
            metric_names=eval_metrics,
            multi_label=is_multi_label,
            threshold=float(cfg.evaluation.threshold),
        )
        test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        results = {
            'task_name': args.task_name,
            'task_type': args.task_type,
            'head_type': 'dummy',
            'seed': args.seed,
            'embedding_type': args.embedding_type,
            'combination_method': args.combination_method,
            'input_dim': input_dim,
            'num_labels': args.num_labels,
            'test_results': test_results,
            'data_stats': {
                'train': len(task_data.get('train', [])),
                'valid': len(task_data.get('valid', [])),
                'test': len(task_data.get('test', [])),
            }
        }

    else:
        # ── MLP path (existing Trainer pipeline) ──
        print(f"\nInitializing model...")
        print(f"   Task type: {args.task_type}")
        if is_multi_label:
            print(f"   Multi-label: Yes ({loss_type})")
        print(f"   Architecture: {input_dim} -> {args.hidden_dim} -> {args.num_labels}")

        trainer = TaskTrainer(cfg, str(output_dir))
        trainer.setup(
            train_loader,
            valid_loader if len(valid_dataset) > 0 else None,
            test_loader if len(test_dataset) > 0 else None,
            input_dim=input_dim,
            output_dim=args.num_labels,
            multi_label=is_multi_label,
        )

        print(f"\nTraining on task: {args.task_name}")
        result = trainer.fit()

        test_results = {}
        scaled_test_results = None
        if len(test_dataset) > 0:
            print(f"\nEvaluating on test set...")
            raw_test_results = trainer.test()
            # Strip test_ prefix for aggregation compatibility
            test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        # Inverse-transform regression predictions for original-scale metrics
        if label_scaler is not None and len(test_dataset) > 0:
            from trl_bench.utils.downstream.metrics import compute_metrics
            scaled_test_results = test_results.copy()

            # Re-collect raw test labels (before z-score) from the label_scaler
            test_labels_orig = np.array(
                [item['label'] for item in task_data.get('test', [])], dtype=np.float32
            )
            test_labels_orig = label_scaler.inverse_transform(
                test_labels_orig.reshape(-1, 1)
            ).ravel()

            # Forward pass to get predictions in z-scored space, then inverse-transform
            trainer.model.eval()
            test_preds = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    logits = trainer.model(xb.to(trainer.device))
                    test_preds.append(logits.squeeze(-1).cpu().numpy())
            test_pred_scaled = np.concatenate(test_preds, axis=0)
            test_pred_orig = label_scaler.inverse_transform(
                test_pred_scaled.reshape(-1, 1)
            ).ravel()

            eval_metrics = _resolve_metrics(args.task_type, False)
            test_results = compute_metrics(test_pred_orig, test_labels_orig, eval_metrics)

        results = {
            'task_name': args.task_name,
            'task_type': args.task_type,
            'head_type': 'mlp',
            'seed': args.seed,
            'embedding_type': args.embedding_type,
            'combination_method': args.combination_method,
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'num_labels': args.num_labels,
            'test_results': test_results,
            'data_stats': {
                'train': len(train_dataset),
                'valid': len(valid_dataset),
                'test': len(test_dataset)
            }
        }
        if label_scaler is not None:
            results['target_zscore'] = True
            results['target_scaler'] = {
                'mean': float(label_scaler.mean_[0]),
                'scale': float(label_scaler.scale_[0]),
            }
            if scaled_test_results is not None:
                results['scaled_test_results'] = scaled_test_results

    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTask complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   Summary: {results_file}")


if __name__ == '__main__':
    main()
