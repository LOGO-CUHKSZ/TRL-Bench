#!/usr/bin/env python3
"""
Train a classifier for TabFact using pre-computed embeddings.

This script trains a lightweight classifier on frozen embeddings,
following the decoupled pipeline framework.

Standard embedding format (all generators should output this):
{
    'table_embeddings': {example_id: np.array(768,), ...},
    'labels': {example_id: int, ...},
    'statement_embeddings': {example_id: np.array(768,), ...}  # Optional
}

Two modes:
- Single-embedding: Only table_embeddings (e.g., TAPAS joint encoding)
- Two-embedding: Both table_embeddings and statement_embeddings (e.g., Doduo+BERT)

Usage:
    # Single-embedding mode (e.g., TAPAS joint or table-only)
    python train.py \
        --train_embeddings embeddings/tabfact/tapas/train.pkl \
        --val_embeddings embeddings/tabfact/tapas/validation.pkl \
        --output_dir checkpoints/tabfact/tapas \
        --device cuda

    # Two-embedding mode with concatenation (1536-dim)
    python train.py \
        --train_embeddings embeddings/tabfact/doduo/train.pkl \
        --val_embeddings embeddings/tabfact/doduo/validation.pkl \
        --output_dir checkpoints/tabfact/doduo_concat \
        --combine_method concat \
        --device cuda

    # Two-embedding mode with addition (768-dim)
    python train.py \
        --train_embeddings embeddings/tabfact/doduo/train.pkl \
        --val_embeddings embeddings/tabfact/doduo/validation.pkl \
        --output_dir checkpoints/tabfact/doduo_add \
        --combine_method add \
        --device cuda
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from omegaconf import OmegaConf

from trl_bench.utils.downstream.heads import MLPHead, ProjectedInteractionHead
from trl_bench.utils.downstream.config import load_config
from trl_bench.utils.downstream.trainer import Trainer


def TabFactClassifier(input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1,
                      dropout_first: bool = False):
    """Binary classifier for TabFact."""
    return MLPHead(
        input_dim=input_dim, output_dim=2,
        hidden_dim=hidden_dim, num_layers=2,
        activation="relu", dropout=dropout, dropout_first=dropout_first,
    )


def LinearClassifier(input_dim: int = 768, dropout: float = 0.1,
                     dropout_first: bool = False):
    """Single-layer linear classifier for ablation baselines."""
    return MLPHead(
        input_dim=input_dim, output_dim=2,
        num_layers=1, dropout=dropout, dropout_first=dropout_first,
    )


def combine_embeddings(table_emb: np.ndarray, statement_emb: np.ndarray,
                       method: str = 'concat') -> np.ndarray:
    """Combine table and statement embeddings.

    Args:
        table_emb: Table embedding (768-dim)
        statement_emb: Statement embedding (768-dim)
        method: 'concat' (1536-dim) or 'add' (768-dim)

    Returns:
        Combined embedding
    """
    if method == 'concat':
        return np.concatenate([table_emb, statement_emb])
    elif method == 'add':
        return table_emb + statement_emb
    else:
        raise ValueError(f"Unknown combine method: {method}. Use 'concat' or 'add'.")


def load_from_precomputed(table_emb_path: str, statement_emb_path: str,
                          labels_json: str, variant: str = 'column_mean',
                          combine_method: str = 'concat',
                          statement_only: bool = False):
    """Load and combine pre-computed table and statement embeddings.

    Bridges two embedding formats:
    - Column embeddings (unified v2): list of dicts with 'table_id' and
      'table_embedding' containing variant sub-keys
    - Text embeddings: list of dicts with 'text_id' and 'embedding'

    The TabFact JSONL provides the join key: each example has 'id', 'table_id',
    'statement', and 'label'. The table_id has a '.csv' suffix that is stripped
    to match column embedding keys.

    Args:
        table_emb_path: Path to column embeddings pickle (unified v2 format)
        statement_emb_path: Path to statement embeddings pickle (text embedding format)
        labels_json: Path to TabFact JSONL (for labels + table_id mapping)
        variant: Which table embedding variant to use: 'column_mean', 'cls_embedding',
                 or 'table_embedding'
        combine_method: How to combine table + statement embeddings ('concat' or 'add')

    Returns:
        embeddings, labels, ids (same as load_embeddings)
    """
    # 1. Load column embeddings → {table_id: embedding}
    with open(table_emb_path, 'rb') as f:
        table_data = pickle.load(f)

    table_lookup = {}
    for entry in table_data:
        tid = entry['table_id']
        table_emb_dict = entry.get('table_embedding', {})
        if variant not in table_emb_dict:
            available = list(table_emb_dict.keys())
            raise ValueError(
                f"Variant '{variant}' not found for table '{tid}'. "
                f"Available: {available}"
            )
        emb = table_emb_dict[variant]
        if emb is None:
            available = [k for k, v in table_emb_dict.items() if v is not None]
            raise ValueError(
                f"Variant '{variant}' is None for table '{tid}'. "
                f"Available non-None: {available}"
            )
        table_lookup[tid] = emb

    # 2. Load statement embeddings → {example_id: embedding}
    with open(statement_emb_path, 'rb') as f:
        stmt_data = pickle.load(f)

    stmt_lookup = {item['text_id']: item['embedding'] for item in stmt_data}

    # 3. Load JSONL → labels + table_id mapping
    with open(labels_json, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # 4. For each example: look up embeddings and combine
    embeddings = []
    labels = []
    ids = []
    missing_table = 0
    missing_stmt = 0

    for ex in examples:
        example_id = str(ex['id'])
        raw_table_id = ex['table_id']
        # Strip .csv suffix for column embedding lookup
        table_id = raw_table_id.replace('.csv', '') if raw_table_id.endswith('.csv') else raw_table_id

        if table_id not in table_lookup:
            missing_table += 1
            continue
        if example_id not in stmt_lookup:
            missing_stmt += 1
            continue

        table_emb = table_lookup[table_id]
        if statement_only:
            table_emb = np.zeros_like(table_emb)
        stmt_emb = stmt_lookup[example_id]
        combined = combine_embeddings(table_emb, stmt_emb, combine_method)

        embeddings.append(combined)
        labels.append(ex['label'])
        ids.append(example_id)

    if missing_table > 0:
        print(f"Warning: {missing_table}/{len(examples)} examples skipped (table embedding not found)")
    if missing_stmt > 0:
        print(f"Warning: {missing_stmt}/{len(examples)} examples skipped (statement embedding not found)")

    if len(embeddings) == 0:
        raise ValueError(
            f"No examples matched after joining table and statement embeddings. "
            f"Total examples: {len(examples)}, missing table: {missing_table}, "
            f"missing statement: {missing_stmt}"
        )

    matched = len(embeddings)
    if matched < len(examples) * 0.5:
        print(f"WARNING: Only {matched}/{len(examples)} ({100*matched/len(examples):.1f}%) "
              f"examples matched — check table_id / statement_id alignment")

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    return embeddings, labels, ids


def load_embeddings(filepath: str, combine_method: str = None):
    """Load embeddings from pickle file.

    Standard format (all embedding generators should output this):
    {
        'table_embeddings': {example_id: np.array(768,), ...},
        'labels': {example_id: int, ...},
        'statement_embeddings': {example_id: np.array(768,), ...}  # Optional
    }

    Two modes:
    1. Two-embedding mode: If both 'table_embeddings' and 'statement_embeddings' exist,
       combine them using combine_method ('concat' or 'add')
    2. Single-embedding mode: If only 'table_embeddings' exists,
       use table embeddings directly (like TAPAS joint encoding)

    Args:
        filepath: Path to pickle file
        combine_method: How to combine embeddings ('concat' or 'add').
                       Only used when statement_embeddings exists.

    Returns:
        embeddings, labels, ids
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    embeddings = []
    labels = []
    ids = []

    if 'table_embeddings' not in data or 'labels' not in data:
        raise ValueError(
            f"Invalid embedding format. Expected keys: 'table_embeddings', 'labels'. "
            f"Got: {list(data.keys())}"
        )

    has_statement = 'statement_embeddings' in data

    if has_statement:
        # Two-embedding mode: combine table and statement
        if combine_method is None:
            combine_method = 'concat'
        missing_stmt = 0
        missing_label = 0
        for example_id in data['table_embeddings'].keys():
            if example_id not in data['statement_embeddings']:
                missing_stmt += 1
                continue
            if example_id not in data['labels']:
                missing_label += 1
                continue
            table_emb = data['table_embeddings'][example_id]
            statement_emb = data['statement_embeddings'][example_id]
            combined = combine_embeddings(table_emb, statement_emb, combine_method)
            embeddings.append(combined)
            labels.append(data['labels'][example_id])
            ids.append(example_id)
        if missing_stmt > 0:
            print(f"  Skipped {missing_stmt} examples missing statement embeddings")
        if missing_label > 0:
            raise ValueError(
                f"{missing_label}/{len(data['table_embeddings'])} examples have table embeddings "
                f"but no labels in {filepath} — data may be corrupt"
            )
    else:
        # Single-embedding mode: use table embedding directly
        missing_label = 0
        for example_id in data['table_embeddings'].keys():
            if example_id not in data['labels']:
                missing_label += 1
                continue
            embeddings.append(data['table_embeddings'][example_id])
            labels.append(data['labels'][example_id])
            ids.append(example_id)
        if missing_label > 0:
            raise ValueError(
                f"{missing_label}/{len(data['table_embeddings'])} examples have table embeddings "
                f"but no labels in {filepath} — data may be corrupt"
            )

    if len(embeddings) == 0:
        raise ValueError(
            f"No usable examples in {filepath}. "
            f"Total table embeddings: {len(data['table_embeddings'])}"
        )

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    return embeddings, labels, ids


def main():
    parser = argparse.ArgumentParser(
        description="Train TabFact classifier on frozen embeddings"
    )
    parser.add_argument(
        '--train_embeddings',
        type=str,
        default=None,
        help='Path to training embeddings pickle file (legacy monolithic format)'
    )
    parser.add_argument(
        '--val_embeddings',
        type=str,
        default=None,
        help='Path to validation embeddings pickle file (legacy monolithic format)'
    )
    # Pre-computed embedding arguments (alternative to --train_embeddings)
    parser.add_argument(
        '--table_embeddings',
        type=str,
        default=None,
        help='Path to column embeddings pickle (unified v2 format)'
    )
    parser.add_argument(
        '--statement_embeddings',
        type=str,
        default=None,
        help='Path to statement embeddings pickle (text embedding format, for train split)'
    )
    parser.add_argument(
        '--val_statement_embeddings',
        type=str,
        default=None,
        help='Path to validation statement embeddings pickle'
    )
    parser.add_argument(
        '--labels_json',
        type=str,
        default=None,
        help='Path to TabFact JSONL (train split, for labels + table_id mapping)'
    )
    parser.add_argument(
        '--val_labels_json',
        type=str,
        default=None,
        help='Path to TabFact validation JSONL'
    )
    parser.add_argument(
        '--table_embedding_variant',
        type=str,
        default='column_mean',
        choices=['column_mean', 'cls_embedding', 'table_embedding', 'token_mean'],
        help='Which table embedding variant to use (default: column_mean)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/tabfact',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='mlp',
        choices=['linear', 'mlp'],
        help='Model type (linear or mlp)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for MLP'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs (overrides YAML max_epochs when passed)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay (overrides YAML when passed)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--combine_method',
        type=str,
        default=None,
        choices=['concat', 'add'],
        help='How to combine table and statement embeddings (for Doduo separate format). '
             'concat=1536-dim, add=768-dim. Ignored for TAPAS/legacy formats.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(_PROJECT_ROOT, 'configs/downstream/table_fact_verification.yaml'),
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--override',
        type=str,
        action='append',
        default=[],
        help='Config overrides (e.g., --override training.optimizer.lr=1e-4)'
    )
    parser.add_argument(
        '--head_type',
        type=str,
        default='mlp',
        choices=['mlp', 'linear', 'dummy', 'interaction'],
        help='Probe type: mlp (PyTorch MLP), linear (sklearn), dummy (majority/mean baseline), '
             'or interaction (projected interaction head for cross-modal alignment)'
    )
    parser.add_argument(
        '--statement_only',
        action='store_true',
        help='Zero out table embeddings to measure statement-only performance (modality ablation)'
    )

    args = parser.parse_args()

    # Validate arguments
    use_precomputed = args.table_embeddings is not None
    if use_precomputed:
        if not args.statement_embeddings or not args.labels_json:
            parser.error("--table_embeddings requires --statement_embeddings and --labels_json")
    elif not args.train_embeddings:
        parser.error("Either --train_embeddings or --table_embeddings is required")

    if args.statement_only and not use_precomputed:
        parser.error("--statement_only requires pre-computed mode (--table_embeddings + --statement_embeddings)")

    # ── Step 1: Detect explicitly-passed CLI flags (vs argparse defaults) ──
    # Follows the run_task.py pattern: YAML is the source of truth;
    # only flags the user actually typed override it.
    _explicitly_passed = set()
    for token in sys.argv[1:]:
        if token.startswith('--'):
            flag = token.lstrip('-').split('=', 1)[0].replace('-', '_')
            _explicitly_passed.add(flag)

    # ── Step 2 Phase A: Load config early (before embedding loading) ──
    # input_dim is not yet known, so it will be patched in Phase B.
    _cli_map = {
        'hidden_dim':     f'head.hidden_dim={args.hidden_dim}',
        'dropout':        f'head.dropout={args.dropout}',
        'lr':             f'training.optimizer.lr={args.lr}',
        'weight_decay':   f'training.optimizer.weight_decay={args.weight_decay}',
        'batch_size':     f'training.batch_size={args.batch_size}',
        'epochs':         f'training.max_epochs={args.epochs}',
        'seed':           f'training.seed={args.seed}',
        'device':         f'training.device={args.device}',
        'model_type':     f'head.num_layers={1 if args.model_type == "linear" else 2}',
    }
    # Only add combine_method if explicitly set (None default would become
    # the string 'None' in OmegaConf dotlist syntax, not null).
    if args.combine_method is not None:
        _cli_map['combine_method'] = f'training.combination_method={args.combine_method}'

    # Always-set overrides (derived from task, not user defaults)
    always_overrides = [
        'task_name=table_fact_verification',
        'head.output_dim=2',
    ]

    config_path = os.path.join(_PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    yaml_path = config_path if os.path.exists(config_path) else None

    if yaml_path:
        # YAML is source of truth; only explicitly-passed CLI flags override
        user_overrides = [v for k, v in _cli_map.items() if k in _explicitly_passed]
        cfg = load_config(yaml_path, overrides=always_overrides + user_overrides + (args.override or []))
    else:
        # No YAML — all CLI values (including defaults) become the config
        all_overrides = list(_cli_map.values())
        cfg = load_config(overrides=always_overrides + all_overrides + (args.override or []))

    # Read resolved values from config (not args)
    combine = str(cfg.training.combination_method) if cfg.training.combination_method else 'concat'
    seed = int(cfg.training.seed)
    batch_size = int(cfg.training.batch_size)
    device = str(cfg.training.device)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("TabFact Classifier Training")
    print("="*60)
    if use_precomputed:
        print(f"Mode: Pre-computed embeddings")
        print(f"Table embeddings: {args.table_embeddings}")
        print(f"Statement embeddings: {args.statement_embeddings}")
        print(f"Labels JSON: {args.labels_json}")
        print(f"Table embedding variant: {args.table_embedding_variant}")
    else:
        print(f"Mode: Monolithic embeddings")
        print(f"Train embeddings: {args.train_embeddings}")
        print(f"Val embeddings: {args.val_embeddings}")
    print(f"Model type: {'linear' if cfg.head.num_layers == 1 else 'mlp'}")
    print(f"Combine method: {combine}")
    print(f"Device: {device}")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print(f"Weight decay: {cfg.training.optimizer.weight_decay}")
    print("="*60)

    # Load embeddings
    print("\nLoading embeddings...")

    if use_precomputed:
        train_emb, train_labels, train_ids = load_from_precomputed(
            args.table_embeddings, args.statement_embeddings,
            args.labels_json, variant=args.table_embedding_variant,
            combine_method=combine,
            statement_only=args.statement_only,
        )
        print(f"Train: {len(train_emb)} examples, embedding dim: {train_emb.shape[1]}")

        if args.val_statement_embeddings and args.val_labels_json:
            val_emb, val_labels, val_ids = load_from_precomputed(
                args.table_embeddings, args.val_statement_embeddings,
                args.val_labels_json, variant=args.table_embedding_variant,
                combine_method=combine,
                statement_only=args.statement_only,
            )
            print(f"Val: {len(val_emb)} examples")
        else:
            from sklearn.model_selection import train_test_split
            train_emb, val_emb, train_labels, val_labels = train_test_split(
                train_emb, train_labels, test_size=0.1, random_state=seed
            )
            print(f"Split: {len(train_emb)} train, {len(val_emb)} val")
    else:
        train_emb, train_labels, train_ids = load_embeddings(
            args.train_embeddings, combine_method=combine
        )
        print(f"Train: {len(train_emb)} examples, embedding dim: {train_emb.shape[1]}")

        if args.val_embeddings:
            val_emb, val_labels, val_ids = load_embeddings(
                args.val_embeddings, combine_method=combine
            )
            print(f"Val: {len(val_emb)} examples")
        else:
            from sklearn.model_selection import train_test_split
            train_emb, val_emb, train_labels, val_labels = train_test_split(
                train_emb, train_labels, test_size=0.1, random_state=seed
            )
            print(f"Split: {len(train_emb)} train, {len(val_emb)} val")

    # Check label distribution
    train_entailed = sum(train_labels == 1)
    train_refuted = sum(train_labels == 0)
    print(f"\nLabel distribution (train):")
    print(f"  Entailed: {train_entailed} ({100*train_entailed/len(train_labels):.1f}%)")
    print(f"  Refuted: {train_refuted} ({100*train_refuted/len(train_labels):.1f}%)")

    if args.head_type == 'linear':
        # ── Linear probe path (sklearn) ──
        from trl_bench.utils.downstream.linear_probe import LinearProbeRunner

        # Linear probe uses the same cfg already loaded (seed is resolved)
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use val as test (TabFact train.py doesn't load test data)
        print(f"\nRunning linear probe (sklearn)...")
        print(f"  Note: evaluating on val split (test requires evaluate.py)")
        runner = LinearProbeRunner(cfg)
        raw_test_results = runner.run(
            train_emb=train_emb,
            train_labels=train_labels,
            test_emb=val_emb,
            test_labels=val_labels,
            task_type='classification',
            metric_names=['accuracy', 'macro_f1', 'auroc'],
        )
        # Strip test_ prefix for aggregation compatibility
        stripped = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        print(f"\nVal accuracy: {stripped.get('accuracy', 0):.4f}")
        print(f"Val macro_f1: {stripped.get('macro_f1', 0):.4f}")

        # Top-level metric keys matching aggregator TASK_METRICS ('accuracy')
        results = {
            'task_name': 'table_fact_verification',
            'head_type': 'linear',
            'statement_only': args.statement_only,
            **stripped,
            'data_stats': {
                'train': len(train_emb),
                'val_as_test': len(val_emb),
                'input_dim': train_emb.shape[1],
            },
        }
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        return

    elif args.head_type == 'dummy':
        # ── Dummy baseline path (label statistics only) ──
        from trl_bench.utils.downstream.dummy_probe import DummyProbeRunner

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use val as test (same as linear path — train.py doesn't load test data)
        print(f"\nRunning dummy baseline (label statistics only)...")
        print(f"  Note: dummy probe evaluating on val split (test requires evaluate.py)")
        runner = DummyProbeRunner()
        raw_test_results = runner.run(
            train_labels=train_labels,
            test_labels=val_labels,
            task_type='classification',
            metric_names=['accuracy', 'macro_f1', 'auroc'],
        )
        stripped = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        print(f"\nVal accuracy: {stripped.get('accuracy', 0):.4f}")
        print(f"Val macro_f1: {stripped.get('macro_f1', 0):.4f}")

        results = {
            'task_name': 'table_fact_verification',
            'head_type': 'dummy',
            'statement_only': args.statement_only,
            'dataset': args.dataset if hasattr(args, 'dataset') else 'tabfact',
            'seed': cfg.training.seed,
            **stripped,
            'data_stats': {
                'train': len(train_labels),
                'val_as_test': len(val_labels),
                'input_dim': train_emb.shape[1],
            },
        }
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        return

    # ── MLP / interaction path (existing Trainer pipeline) ──

    # ── Interaction head: apply default overrides before config is read ──
    def _override_contains(key_suffix, overrides):
        """Check if any --override dotlist entry targets the given config path suffix."""
        return any(key_suffix in o for o in (overrides or []))

    if args.head_type == 'interaction':
        # Interaction head requires pre-computed two-embedding mode
        if not use_precomputed:
            raise ValueError(
                "interaction head requires pre-computed mode "
                "(--table_embeddings + --statement_embeddings), not --train_embeddings"
            )
        if combine != 'concat':
            raise ValueError("interaction head requires combine_method='concat'")
        # Override defaults for interaction head (larger batch, weight decay)
        if 'batch_size' not in _explicitly_passed and not _override_contains('training.batch_size', args.override):
            OmegaConf.update(cfg, 'training.batch_size', 256)
            batch_size = 256
        if 'weight_decay' not in _explicitly_passed and not _override_contains('training.optimizer.weight_decay', args.override):
            OmegaConf.update(cfg, 'training.optimizer.weight_decay', 1e-4)

    # ── Step 2 Phase B: Patch input_dim into config now that embeddings are loaded ──
    input_dim = train_emb.shape[1]
    OmegaConf.update(cfg, 'head.input_dim', input_dim)

    # Create data loaders (using resolved config values)
    train_dataset = TensorDataset(
        torch.tensor(train_emb, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_emb, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 4: Build model from resolved config ──
    if args.head_type == 'interaction':
        # Extract exact table embedding dim from the pickle (NOT input_dim // 2,
        # since table dims vary by model: e.g. TURL=312, BERT=768)
        with open(args.table_embeddings, 'rb') as f:
            first_entry = pickle.load(f)[0]
        table_dim = np.array(first_entry['table_embedding'][args.table_embedding_variant]).shape[0]
        stmt_dim = input_dim - table_dim
        model = ProjectedInteractionHead(
            table_input_dim=table_dim,
            stmt_input_dim=stmt_dim,
            projection_dim=int(cfg.head.projection_dim),
            classifier_hidden_dim=int(cfg.head.classifier_hidden_dim),
            num_classes=2,
            dropout=float(cfg.head.dropout),
            interaction_type=str(cfg.head.interaction_type),
            normalize_projection=bool(cfg.head.normalize_projection),
        )
    elif cfg.head.num_layers == 1:
        model = LinearClassifier(input_dim=input_dim, dropout=cfg.head.dropout,
                                 dropout_first=cfg.head.dropout_first)
    else:
        model = TabFactClassifier(
            input_dim=input_dim,
            hidden_dim=cfg.head.hidden_dim,
            dropout=cfg.head.dropout,
            dropout_first=cfg.head.dropout_first,
        )
    print(f"\nModel: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # Custom checkpoint format: state_dict only (backward compatible)
    def _tabfact_best_ckpt(model, optimizer, epoch, metrics, config):
        return model.state_dict()

    trainer = Trainer(cfg, str(output_path), checkpoint_format_fn=_tabfact_best_ckpt)
    trainer.setup(
        train_loader, val_loader, None,
        input_dim=input_dim, output_dim=2,
        model=model,
    )

    print("\nTraining...")
    result = trainer.fit()

    # Save final model (state_dict only, like the original)
    torch.save(trainer.model.state_dict(), output_path / "final_model.pt")

    # Build history.json in original format
    history = []
    for h in result['history']:
        history.append({
            'epoch': h['epoch'],
            'train_loss': h.get('train_loss', 0),
            'train_acc': h.get('train_accuracy', 0),
            'train_f1': h.get('train_macro_f1', 0),
            'val_loss': h.get('val_loss', 0),
            'val_acc': h.get('val_accuracy', 0),
            'val_f1': h.get('val_macro_f1', 0),
        })

    best_epoch = result.get('best_epoch', len(history))
    best_val_loss = result.get('best_value', None)

    # ── Step 5: Save config.json from resolved config values ──
    if args.head_type == 'interaction':
        model_type = 'interaction'
    else:
        model_type = 'linear' if cfg.head.num_layers == 1 else 'mlp'
    config_out = {
        'seed': int(cfg.training.seed),
        'train_embeddings': args.train_embeddings,
        'val_embeddings': args.val_embeddings,
        'table_embeddings': args.table_embeddings,
        'statement_embeddings': args.statement_embeddings,
        'labels_json': args.labels_json,
        'table_embedding_variant': args.table_embedding_variant if use_precomputed else None,
        'model_type': model_type,
        'input_dim': input_dim,
        'hidden_dim': int(cfg.head.hidden_dim) if model_type == 'mlp' else None,
        'dropout': float(cfg.head.dropout),
        'dropout_first': bool(cfg.head.dropout_first),
        'epochs': int(cfg.training.max_epochs),
        'batch_size': int(cfg.training.batch_size),
        'lr': float(cfg.training.optimizer.lr),
        'weight_decay': float(cfg.training.optimizer.weight_decay),
        'combine_method': combine,
        'statement_only': args.statement_only,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss) if best_val_loss is not None else None,
        'timestamp': datetime.now().isoformat(),
        # Interaction head params (persisted for evaluate.py reconstruction)
        'table_input_dim': table_dim if args.head_type == 'interaction' else None,
        'stmt_input_dim': stmt_dim if args.head_type == 'interaction' else None,
        'projection_dim': int(cfg.head.projection_dim) if args.head_type == 'interaction' else None,
        'interaction_type': str(cfg.head.interaction_type) if args.head_type == 'interaction' else None,
        'normalize_projection': bool(cfg.head.normalize_projection) if args.head_type == 'interaction' else None,
        'classifier_hidden_dim': int(cfg.head.classifier_hidden_dim) if args.head_type == 'interaction' else None,
    }

    with open(output_path / "config.json", 'w') as f:
        json.dump(config_out, f, indent=2)

    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best epoch: {best_epoch}")
    if best_val_loss is not None:
        print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")

    # Load best model and print classification report
    best_path = output_path / "best_model.pt"
    if best_path.exists():
        best_sd = torch.load(best_path, map_location=device)
        # Handle both state_dict-only and wrapped checkpoint formats
        if isinstance(best_sd, dict) and 'model_state_dict' in best_sd:
            best_sd = best_sd['model_state_dict']
        model.load_state_dict(best_sd)
        model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                logits = model(batch_emb.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())

        print("\nClassification Report (Validation):")
        print(classification_report(all_labels, all_preds,
                                    target_names=['Refuted', 'Entailed']))


if __name__ == '__main__':
    main()
