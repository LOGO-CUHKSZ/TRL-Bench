#!/usr/bin/env python3
"""
Record linkage evaluation using row-level embeddings.

Given pre-computed row embeddings and a labels JSON file (train/valid/test
row-pair annotations), this script:
  1. Pairs row embeddings according to labels
  2. Combines them (concat/diff/add/multiply)
  3. Trains a 2-layer MLP binary classifier
  4. Reports accuracy, F1, precision, recall on the test split

Usage:
    python downstream_tasks/record_linkage/run_record_linkage.py \
        --embeddings embeddings/row/tabpfn/deepmatcher_beer.pkl \
        --labels datasets/record_linkage/deepmatcher_beer/labels.json \
        --task_name record_linkage_deepmatcher_beer \
        --output_dir results/evaluation/record_linkage/tabpfn/seed42
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from trl_bench.utils.downstream.run_task import combine_embeddings, EmbeddingDataset


def _build_table_lookup(row_embeddings):
    """Build table filename -> row_embeddings array lookup."""
    table_lookup = {}
    for entry in row_embeddings:
        table_id = entry['table_id']
        stem = table_id.replace('.csv', '')
        row_embs = entry['row_embeddings']
        if isinstance(row_embs, list):
            row_embs = np.array(row_embs)
        table_lookup[stem] = row_embs
        table_lookup[table_id] = row_embs

    print(f"Row embedding lookup: {len(table_lookup)} entries")
    for key, embs in table_lookup.items():
        if not key.endswith('.csv'):
            print(f"  {key}: {embs.shape}")

    return table_lookup


def _iterate_row_pairs(table_lookup, labels):
    """Iterate over row pairs from labels, yielding (emb1, emb2, label, split).

    Handles table lookup, row index validation, and skip tracking.
    Prints per-split statistics.
    """
    print("=" * 60)

    for split in ['train', 'valid', 'test']:
        pairs = labels.get(split, [])
        if not pairs:
            print(f"{split.upper()}: 0 pairs (empty split)")
            continue

        skipped = 0
        yielded = 0
        pos = 0

        for pair in pairs:
            t1_info = pair['table1']
            t2_info = pair['table2']

            t1_stem = t1_info['filename'].replace('.csv', '')
            t2_stem = t2_info['filename'].replace('.csv', '')

            t1_embs = table_lookup.get(t1_stem)
            if t1_embs is None:
                t1_embs = table_lookup.get(t1_info['filename'])
            t2_embs = table_lookup.get(t2_stem)
            if t2_embs is None:
                t2_embs = table_lookup.get(t2_info['filename'])

            if t1_embs is None or t2_embs is None:
                if skipped == 0:
                    missing = []
                    if t1_embs is None:
                        missing.append(f"table1={t1_info['filename']}")
                    if t2_embs is None:
                        missing.append(f"table2={t2_info['filename']}")
                    print(f"  WARNING: Missing embeddings for {', '.join(missing)}")
                skipped += 1
                continue

            row_idx1 = t1_info['row_idx']
            row_idx2 = t2_info['row_idx']

            if row_idx1 >= len(t1_embs):
                if skipped == 0:
                    print(f"  WARNING: row_idx {row_idx1} out of bounds for {t1_info['filename']} (has {len(t1_embs)} rows)")
                skipped += 1
                continue
            if row_idx2 >= len(t2_embs):
                if skipped == 0:
                    print(f"  WARNING: row_idx {row_idx2} out of bounds for {t2_info['filename']} (has {len(t2_embs)} rows)")
                skipped += 1
                continue

            label = int(pair['label'])
            yield t1_embs[row_idx1], t2_embs[row_idx2], label, split
            yielded += 1
            if label == 1:
                pos += 1

        total = len(pairs)
        neg = yielded - pos
        print(f"{split.upper()}: {yielded}/{total} pairs (pos={pos}, neg={neg}, skipped={skipped})")

        if skipped > 0 and skipped == total:
            print(f"  ERROR: All pairs skipped in {split} split!")
            print(f"  Available tables: {[k for k in table_lookup if not k.endswith('.csv')]}")
            sys.exit(1)

    print("=" * 60)


def prepare_record_linkage_data(row_embeddings, labels, combination_method='concat'):
    """
    Load row embeddings and pair them according to labels.

    Args:
        row_embeddings: List of dicts from row embedding pickle.
            Each dict has 'table_id' and 'row_embeddings' (n_rows, emb_dim).
        labels: Dict with 'train', 'valid', 'test' splits.
            Each entry has table1/table2 with 'filename' and 'row_idx', plus 'label'.
        combination_method: 'concat', 'diff', 'add', or 'multiply'

    Returns:
        Dict with train/valid/test lists of {'embedding': [...], 'label': 0/1}
    """
    table_lookup = _build_table_lookup(row_embeddings)

    task_data = {'train': [], 'valid': [], 'test': []}
    for emb1, emb2, label, split in _iterate_row_pairs(table_lookup, labels):
        combined = combine_embeddings(emb1, emb2, method=combination_method)
        task_data[split].append({'embedding': combined, 'label': label})

    return task_data


def prepare_record_linkage_cosine_scores(row_embeddings, labels):
    """
    Compute per-pair cosine similarity scores from raw row embeddings.

    Args:
        row_embeddings: List of dicts from row embedding pickle.
        labels: Dict with 'train', 'valid', 'test' splits.

    Returns:
        Dict with train/valid/test lists of {'score': float, 'label': int}
    """
    table_lookup = _build_table_lookup(row_embeddings)

    cosine_data = {'train': [], 'valid': [], 'test': []}
    for emb1, emb2, label, split in _iterate_row_pairs(table_lookup, labels):
        cos = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        cosine_data[split].append({'score': cos, 'label': label})

    return cosine_data


def main():
    parser = argparse.ArgumentParser(
        description="Train binary classifier on record linkage row-pair embeddings")
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Row embedding pickle (embeddings/row/{model}/{dataset}.pkl)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Labels JSON file (datasets/record_linkage/{dataset}/labels.json)')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name for logging')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    # Config-driven mode (optional)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides CLI defaults)')
    parser.add_argument('--override', type=str, action='append', default=[],
                        help='Config overrides (e.g., --override training.optimizer.lr=1e-4)')

    # Embedding combination
    parser.add_argument('--combination_method', type=str, default='concat',
                        choices=['concat', 'add', 'multiply', 'diff'],
                        help='How to combine row pair embeddings')

    # Classifier architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for 2-layer MLP')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of output labels (2 for binary classification)')

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Hardware
    parser.add_argument('--num_workers', type=int, default=0)

    # Probe type
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'linear', 'dummy', 'cosine_threshold'],
                        help='Probe type: mlp (PyTorch MLP), linear (sklearn), '
                             'dummy (majority/mean baseline), or cosine_threshold '
                             '(threshold on raw cosine similarity)')

    args = parser.parse_args()

    # Load row embeddings
    print(f"\nLoading row embeddings from: {args.embeddings}")
    with open(args.embeddings, 'rb') as f:
        row_embeddings = pickle.load(f)
    print(f"  Loaded {len(row_embeddings)} table entries")

    # Load labels
    print(f"\nLoading labels from: {args.labels}")
    with open(args.labels, 'r') as f:
        labels = json.load(f)

    # Prepare task data — cosine_threshold path bypasses combination + datasets
    print(f"\nPreparing record linkage data...")
    print(f"  Task: {args.task_name}")

    if args.head_type == 'cosine_threshold':
        print(f"  Mode: cosine similarity threshold (bypassing embedding combination)")
        cosine_data = prepare_record_linkage_cosine_scores(row_embeddings, labels)
        if not cosine_data.get('train'):
            raise ValueError("No training data available for cosine threshold!")
        if not cosine_data.get('test'):
            raise ValueError("No test data available for cosine threshold!")
        task_data = None
        input_dim = None
    else:
        print(f"  Combination method: {args.combination_method}")
        task_data = prepare_record_linkage_data(
            row_embeddings, labels,
            combination_method=args.combination_method,
        )

        # Auto-detect input dimension
        if len(task_data['train']) > 0:
            input_dim = len(task_data['train'][0]['embedding'])
            print(f"\nDetected input dimension: {input_dim}")
        else:
            raise ValueError("No training data available!")

        # Z-score normalize combined pair features for learned probes (mlp).
        # Fit on training pairs only; skip for cosine_threshold (raw geometry)
        # and linear/dummy (LinearProbeRunner handles normalization internally).
        if args.head_type == 'mlp':
            from sklearn.preprocessing import StandardScaler
            train_embs_raw = np.array([item['embedding'] for item in task_data['train']], dtype=np.float32)
            emb_scaler = StandardScaler()
            emb_scaler.fit(train_embs_raw)

            for split_name in ('train', 'valid', 'test'):
                for item in task_data.get(split_name, []):
                    item['embedding'] = emb_scaler.transform(
                        np.array(item['embedding'], dtype=np.float32).reshape(1, -1)
                    )[0]

            print(f"   Embedding z-score: applied (fit on {len(train_embs_raw)} train pairs)")

        # Create datasets
        train_dataset = EmbeddingDataset(task_data.get('train', []),
                                         task_type='classification',
                                         num_labels=args.num_labels)
        valid_dataset = EmbeddingDataset(task_data.get('valid', []),
                                         task_type='classification',
                                         num_labels=args.num_labels)
        test_dataset = EmbeddingDataset(task_data.get('test', []),
                                        task_type='classification',
                                        num_labels=args.num_labels)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    # Build config (from YAML or CLI args)
    from trl_bench.utils.downstream.config import load_config

    # Detect which CLI flags the user actually passed
    _explicitly_passed = set()
    for token in sys.argv[1:]:
        if token.startswith('--'):
            flag = token.lstrip('-').split('=', 1)[0].replace('-', '_')
            _explicitly_passed.add(flag)

    _cli_map = {
        'task_name':          f'task_name={args.task_name}',
        'combination_method': f'training.combination_method={args.combination_method}',
        'hidden_dim':         f'head.hidden_dim={args.hidden_dim}',
        'num_labels':         f'head.output_dim={args.num_labels}',
        'dropout_prob':       f'head.dropout={args.dropout_prob}',
        'learning_rate':      f'training.optimizer.lr={args.learning_rate}',
        'batch_size':         f'training.batch_size={args.batch_size}',
        'max_epochs':         f'training.max_epochs={args.max_epochs}',
        'seed':               f'training.seed={args.seed}',
    }

    if args.config:
        user_cli_overrides = [v for k, v in _cli_map.items() if k in _explicitly_passed]
        user_cli_overrides.append('training.loss.type=cross_entropy')
        cfg = load_config(args.config, overrides=user_cli_overrides + args.override)
    else:
        default_config = str(Path(_repo_root) / 'configs' / 'downstream' / 'record_linkage.yaml')
        all_cli_overrides = list(_cli_map.values()) + ['training.loss.type=cross_entropy']
        cfg = load_config(default_config, overrides=all_cli_overrides + args.override)

    # Train with Trainer
    from trl_bench.utils.downstream.trainer import Trainer as TaskTrainer
    from omegaconf import OmegaConf

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set evaluation metrics in config
    OmegaConf.set_struct(cfg, False)
    cfg.evaluation.metrics = ['f1', 'accuracy', 'weighted_f1', 'macro_f1', 'precision', 'recall', 'auroc']
    OmegaConf.set_struct(cfg, True)

    metrics_list = list(cfg.evaluation.metrics)
    # cosine_threshold passes hard int predictions — AUROC needs soft scores
    if args.head_type == 'cosine_threshold':
        metrics_list = [m for m in metrics_list if m != 'auroc']

    if args.head_type == 'linear':
        # ── Linear probe path (sklearn) ──
        from trl_bench.utils.downstream.linear_probe import LinearProbeRunner

        def _extract_arrays(split_data):
            emb = np.array([item['embedding'] for item in split_data])
            lab = np.array([item['label'] for item in split_data])
            return emb, lab

        train_emb_np, train_lab_np = _extract_arrays(task_data.get('train', []))
        val_emb_np, val_lab_np = (
            _extract_arrays(task_data['valid']) if task_data.get('valid') else (None, None))
        test_emb_np, test_lab_np = _extract_arrays(task_data.get('test', []))

        print(f"\nRunning linear probe (sklearn)...")
        runner = LinearProbeRunner(cfg)
        raw_test_results = runner.run(
            train_emb=train_emb_np,
            train_labels=train_lab_np,
            test_emb=test_emb_np,
            test_labels=test_lab_np,
            task_type='classification',
            metric_names=metrics_list,
            val_emb=val_emb_np,
            val_labels=val_lab_np,
        )

        # Strip test_ prefix (matching MLP path convention for aggregation)
        test_results = {
            k.removeprefix('test_'): v for k, v in raw_test_results.items()
        }

        results = {
            'task_name': args.task_name,
            'task_type': 'classification',
            'head_type': 'linear',
            'seed': args.seed,
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

        train_lab_np = np.array([item['label'] for item in task_data.get('train', [])])
        test_lab_np = np.array([item['label'] for item in task_data.get('test', [])])

        print(f"\nRunning dummy baseline (label statistics only)...")
        runner = DummyProbeRunner()
        raw_test_results = runner.run(
            train_labels=train_lab_np,
            test_labels=test_lab_np,
            task_type='classification',
            metric_names=metrics_list,
        )
        test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        results = {
            'task_name': args.task_name,
            'task_type': 'classification',
            'head_type': 'dummy',
            'seed': args.seed,
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

    elif args.head_type == 'cosine_threshold':
        # ── Cosine similarity threshold baseline ──
        from trl_bench.utils.downstream.cosine_threshold_probe import CosineThresholdRunner

        train_scores = np.array([item['score'] for item in cosine_data.get('train', [])])
        train_lab_np = np.array([item['label'] for item in cosine_data.get('train', [])])
        test_scores = np.array([item['score'] for item in cosine_data.get('test', [])])
        test_lab_np = np.array([item['label'] for item in cosine_data.get('test', [])])

        val_scores, val_lab_np = None, None
        if cosine_data.get('valid'):
            val_scores = np.array([item['score'] for item in cosine_data['valid']])
            val_lab_np = np.array([item['label'] for item in cosine_data['valid']])

        print(f"\nRunning cosine similarity threshold baseline...")
        runner = CosineThresholdRunner(optimize_metric='f1')
        raw_test_results = runner.run(
            train_scores=train_scores, train_labels=train_lab_np,
            test_scores=test_scores, test_labels=test_lab_np,
            metric_names=metrics_list,
            val_scores=val_scores, val_labels=val_lab_np,
        )

        # Separate metadata from metrics, then strip test_ prefix
        cosine_meta = raw_test_results.pop('cosine_threshold_meta', {})
        test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        results = {
            'task_name': args.task_name,
            'task_type': 'classification',
            'head_type': 'cosine_threshold',
            'seed': args.seed,
            'combination_method': 'cosine_similarity',
            'num_labels': args.num_labels,
            'cosine_threshold_meta': cosine_meta,
            'test_results': test_results,
            'data_stats': {
                'train': len(cosine_data.get('train', [])),
                'valid': len(cosine_data.get('valid', [])),
                'test': len(cosine_data.get('test', [])),
            }
        }

    else:
        # ── MLP path (existing Trainer pipeline) ──
        print(f"\nInitializing model...")
        print(f"  Architecture: {input_dim} -> {args.hidden_dim} -> {args.num_labels}")

        trainer = TaskTrainer(cfg, str(output_dir))
        trainer.setup(
            train_loader,
            valid_loader if len(valid_dataset) > 0 else None,
            test_loader if len(test_dataset) > 0 else None,
            input_dim=input_dim,
            output_dim=args.num_labels,
            multi_label=False,
        )

        print(f"\nTraining on task: {args.task_name}")
        result = trainer.fit()

        raw_test_results = {}
        if len(test_dataset) > 0:
            print(f"\nEvaluating on test set...")
            raw_test_results = trainer.test()

        # Strip 'test_' prefix for aggregation compatibility
        test_results = {
            k.removeprefix('test_'): v for k, v in raw_test_results.items()
        }

        results = {
            'task_name': args.task_name,
            'task_type': 'classification',
            'head_type': 'mlp',
            'seed': args.seed,
            'combination_method': args.combination_method,
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'num_labels': args.num_labels,
            'test_results': test_results,
            'data_stats': {
                'train': len(train_dataset),
                'valid': len(valid_dataset),
                'test': len(test_dataset),
            }
        }

    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTask complete!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"  Summary: {results_file}")


if __name__ == '__main__':
    main()
