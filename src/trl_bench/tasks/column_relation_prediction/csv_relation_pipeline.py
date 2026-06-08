#!/usr/bin/env python3
"""
Complete pipeline for relation extraction using csv_dataset_with_relations

This pipeline:
1. Loads pre-computed embeddings
2. Matches by column_id (verified 100% alignment)
3. Creates column pairs for relation extraction
4. Trains lightweight classifier

Key insight: csv_dataset_with_relations has EXTRA columns not in relation annotations.
Solution: Only use annotated column_ids for training.

Supports two embedding input modes:

  (A) Unified format (--embeddings_file): A single .pkl file containing a list of dicts
      with 'table_id' and 'column_embeddings'. Train/test split comes from dataset metadata.

  (B) Legacy split format (--embeddings_dir): Separate train/test embedding files.
      embeddings_dir/
        ├── train_embeddings.pkl
        └── test_embeddings.pkl

  dataset_dir/
    ├── train/
    │   └── train_metadata.json    # Table metadata with relation annotations
    └── test/
        └── test_metadata.json     # Table metadata with relation annotations

Usage:
    # Unified format (preferred):
    python csv_relation_pipeline.py \
        --embeddings_file=embeddings/column/starmie/WikiCT_relation.pkl \
        --dataset_dir=datasets/WikiCT_relation \
        --epochs=20 \
        --output_dir=results/evaluation/column_relation_prediction/starmie

    # Legacy split format:
    python csv_relation_pipeline.py \
        --embeddings_dir=embeddings/doduo_wikict_relation \
        --dataset_dir=datasets/WikiCT_relation \
        --epochs=2 \
        --output_dir=results/evaluation/column_relation_prediction/doduo
"""

import json
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import os

# Resolve project root (two levels up from this script)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    """Build a DataLoader with conservative worker settings for large relation datasets."""
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
    }
    if num_workers > 0:
        # file_system reduces the number of inherited file descriptors versus the
        # default strategy and is safer on shared systems with lower ulimit -n.
        torch.multiprocessing.set_sharing_strategy('file_system')
        kwargs['persistent_workers'] = True
    return DataLoader(dataset, **kwargs)


def load_unified_embeddings(path):
    """Load unified v2.0 format: list of dicts with 'table_id' and 'column_embeddings'.

    Returns dict keyed by normalized table_id -> list of column embeddings (sorted by key).
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    embeddings_dict = {}
    for item in data:
        table_id = item['table_id']
        # Strip "table_NNNNNN_" prefix to match metadata IDs
        if table_id.startswith('table_') and '_' in table_id[6:]:
            table_id = table_id.split('_', 2)[-1]
        col_embs = item['column_embeddings']
        sorted_keys = sorted(col_embs.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
        embeddings_dict[table_id] = [col_embs[k] for k in sorted_keys]
    return embeddings_dict


def load_legacy_embeddings(path):
    """Load legacy split format: dict with 'embeddings' and 'table_ids' keys.

    Returns dict keyed by table_id -> list of column embeddings.
    """
    with open(path, 'rb') as f:
        emb_data = pickle.load(f)
    all_embeddings = emb_data['embeddings']
    all_table_ids = emb_data['table_ids']
    embeddings_dict = {}
    for tid, emb_matrix in zip(all_table_ids, all_embeddings):
        embeddings_dict[tid] = [emb_matrix[i] for i in range(emb_matrix.shape[0])]
    return embeddings_dict


class CSVRelationDataset(Dataset):
    """
    Dataset for csv_dataset_with_relations with column_id-based matching

    Handles the fact that csv_dataset has extra columns not in relation annotations.
    Uses explicit column_id matching instead of position-based.
    """

    def __init__(self,
                 embeddings_dict: dict,
                 metadata_path: str,
                 split: str = 'train',
                 max_tables: int = None,
                 use_first_col_pairs: bool = True):
        """
        Args:
            embeddings_dict: Dict mapping table_id -> list of column embeddings
            metadata_path: Path to csv_dataset_with_relations metadata JSON
            split: 'train' or 'test' (for display)
            max_tables: Limit number of tables for debugging
            use_first_col_pairs: If True, pair all columns with first column (Doduo style)
        """
        self.embeddings_dict = embeddings_dict
        print(f"Using {len(self.embeddings_dict)} tables with embeddings")

        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"Loaded {len(metadata)} tables from metadata")

        # Build samples
        self.samples = []
        self.use_first_col_pairs = use_first_col_pairs

        print(f"Building {split} dataset...")
        print(f"Pairing strategy: {'First column pairs (Doduo style)' if use_first_col_pairs else 'All pairwise combinations'}")

        processed_tables = 0
        skipped_tables = 0
        skipped_single_col = 0
        total_pairs = 0

        for table in tqdm(metadata, desc="Processing tables"):
            if max_tables and processed_tables >= max_tables:
                break

            table_id = table['table_id']

            # Skip tables not in embeddings
            if table_id not in self.embeddings_dict:
                skipped_tables += 1
                continue

            # Get relation annotations (only annotated columns)
            rel_annots = table['relation_annotations']

            # Skip tables with < 2 annotated columns
            if len(rel_annots) < 2:
                skipped_single_col += 1
                continue

            # Get embeddings for this table (all columns)
            table_embeddings = self.embeddings_dict[table_id]

            # Create mapping: column_id -> (embedding, label)
            col_data = {}
            for rel_ann in rel_annots:
                col_id = rel_ann['column_id']

                # Check if column_id is within bounds
                if col_id >= len(table_embeddings):
                    print(f"Warning: {table_id} col_id {col_id} >= {len(table_embeddings)} embeddings")
                    continue

                embedding = table_embeddings[col_id]
                label = np.array(rel_ann['relation_ids'], dtype=np.float32)

                col_data[col_id] = {
                    'embedding': embedding,
                    'label': label
                }

            # Need at least 2 columns with data
            if len(col_data) < 2:
                skipped_single_col += 1
                continue

            # Create column pairs
            col_ids = sorted(col_data.keys())

            if use_first_col_pairs:
                # Doduo style: pair first column with all others
                first_col_id = col_ids[0]
                first_emb = col_data[first_col_id]['embedding']

                for col_id in col_ids[1:]:
                    pair_emb = np.concatenate([first_emb, col_data[col_id]['embedding']])
                    label = col_data[col_id]['label']

                    self.samples.append({
                        'embedding': pair_emb,
                        'label': label,
                        'table_id': table_id,
                        'col_i': first_col_id,
                        'col_j': col_id
                    })
                    total_pairs += 1
            else:
                # All pairwise combinations
                for i in range(len(col_ids)):
                    for j in range(i+1, len(col_ids)):
                        col_i = col_ids[i]
                        col_j = col_ids[j]

                        pair_emb = np.concatenate([col_data[col_i]['embedding'],
                                                   col_data[col_j]['embedding']])
                        label = col_data[col_j]['label']

                        self.samples.append({
                            'embedding': pair_emb,
                            'label': label,
                            'table_id': table_id,
                            'col_i': col_i,
                            'col_j': col_j
                        })
                        total_pairs += 1

            processed_tables += 1

        print(f"\n✓ Created {len(self.samples)} column pairs from {processed_tables} tables")
        if skipped_tables > 0:
            print(f"  Skipped {skipped_tables} tables (not in embeddings)")
        if skipped_single_col > 0:
            print(f"  Skipped {skipped_single_col} tables (< 2 annotated columns)")

        # Get number of relation classes and embedding dimension from first sample
        if len(self.samples) > 0:
            self.num_relations = len(self.samples[0]['label'])
            # Pair embedding is concatenation of two column embeddings
            self.pair_embedding_dim = len(self.samples[0]['embedding'])
            self.single_embedding_dim = self.pair_embedding_dim // 2
            print(f"  Number of relation types: {self.num_relations}")
            print(f"  Single embedding dimension: {self.single_embedding_dim}")
            print(f"  Pair embedding dimension: {self.pair_embedding_dim}")
        else:
            raise ValueError("No samples created! Check data alignment.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'embedding': torch.FloatTensor(sample['embedding']),
            'label': torch.FloatTensor(sample['label'])
        }


from trl_bench.utils.downstream.heads import MLPHead
from trl_bench.utils.downstream.config import load_config
from trl_bench.utils.downstream.trainer import Trainer, DefaultTaskSpec


class _RelationTaskSpec(DefaultTaskSpec):
    """TaskSpec for CSVRelationDataset which yields dict batches."""

    def forward_and_loss(self, model, batch, device, criterion):
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        return loss, logits

    def extract_targets(self, batch):
        return batch['label']


def main():
    parser = argparse.ArgumentParser(
        description="Train relation classifier on csv_dataset_with_relations embeddings"
    )
    parser.add_argument('--embeddings_file', type=str, default=None,
                       help='Single unified .pkl embedding file (preferred)')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                       help='Directory containing split embedding files (legacy)')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory containing dataset metadata with relation annotations')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden layer dimension (default: from YAML config)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (default: from YAML config)')
    parser.add_argument('--max_tables', type=int, default=None,
                       help='Limit tables for debugging')
    parser.add_argument('--use_all_pairs', action='store_true',
                       help='Use all pairwise combinations instead of first-column pairs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers (default: 0 for cluster safety)')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(_PROJECT_ROOT, 'results', 'evaluation', 'column_relation_prediction'),
                       help='Output directory for saving models')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config training.seed)')
    parser.add_argument('--config', type=str,
                       default='configs/downstream/column_relation_prediction.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--override', type=str, action='append', default=[],
                       help='Config overrides')
    parser.add_argument('--head_type', type=str, default='mlp',
                       choices=['mlp', 'linear', 'dummy'],
                       help='Probe type: mlp (PyTorch MLP), linear (sklearn), or dummy (majority/mean baseline)')

    args = parser.parse_args()

    # Validate that exactly one embedding source is provided
    if not args.embeddings_file and not args.embeddings_dir:
        parser.error("Must specify either --embeddings_file or --embeddings_dir")
    if args.embeddings_file and args.embeddings_dir:
        parser.error("Cannot specify both --embeddings_file and --embeddings_dir")

    # Resolve relative paths against cwd
    if args.embeddings_file:
        args.embeddings_file = os.path.abspath(args.embeddings_file)
    if args.embeddings_dir:
        args.embeddings_dir = os.path.abspath(args.embeddings_dir)
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    train_metadata_path = os.path.join(args.dataset_dir, 'train', 'train_metadata.json')
    test_metadata_path = os.path.join(args.dataset_dir, 'test', 'test_metadata.json')

    # Verify required files exist
    if args.embeddings_file:
        required_files = {
            'embeddings_file': args.embeddings_file,
            'dataset_dir/train/train_metadata.json': train_metadata_path,
            'dataset_dir/test/test_metadata.json': test_metadata_path
        }
    else:
        train_embeddings_path = os.path.join(args.embeddings_dir, 'train_embeddings.pkl')
        test_embeddings_path = os.path.join(args.embeddings_dir, 'test_embeddings.pkl')
        required_files = {
            'embeddings_dir/train_embeddings.pkl': train_embeddings_path,
            'embeddings_dir/test_embeddings.pkl': test_embeddings_path,
            'dataset_dir/train/train_metadata.json': train_metadata_path,
            'dataset_dir/test/test_metadata.json': test_metadata_path
        }

    missing_files = []
    for filename, filepath in required_files.items():
        if not os.path.exists(filepath):
            missing_files.append(f"{filename} -> {filepath}")

    if missing_files:
        print("Error: Missing required files:")
        for filename in missing_files:
            print(f"  - {filename}")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CSV_DATASET_WITH_RELATIONS RELATION EXTRACTION")
    print("=" * 60)
    if args.embeddings_file:
        print(f"Embeddings file: {args.embeddings_file}")
    else:
        print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pairing strategy: {'All pairs' if args.use_all_pairs else 'First column pairs (Doduo)'}")
    print()

    # Load embeddings
    if args.embeddings_file:
        print(f"Loading unified embeddings from {args.embeddings_file}...")
        unified_dict = load_unified_embeddings(args.embeddings_file)
        print(f"Loaded {len(unified_dict)} tables from unified file")
        train_emb_dict = unified_dict
        test_emb_dict = unified_dict
    else:
        print(f"Loading legacy split embeddings from {args.embeddings_dir}...")
        train_emb_dict = load_legacy_embeddings(train_embeddings_path)
        test_emb_dict = load_legacy_embeddings(test_embeddings_path)

    # Create datasets
    train_dataset = CSVRelationDataset(
        train_emb_dict,
        train_metadata_path,
        split='train',
        max_tables=args.max_tables,
        use_first_col_pairs=not args.use_all_pairs
    )

    test_dataset = CSVRelationDataset(
        test_emb_dict,
        test_metadata_path,
        split='test',
        max_tables=args.max_tables,
        use_first_col_pairs=not args.use_all_pairs
    )

    # Create dataloaders
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = make_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Build config first (YAML is source of truth, CLI args and --override can tune)
    config_path = os.path.join(_PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    # Only override head settings when explicitly provided on CLI
    cli_overrides = [
        f'task_name=column_relation_prediction',
        f'head.input_dim={train_dataset.pair_embedding_dim}',
        f'head.output_dim={train_dataset.num_relations}',
        f'training.optimizer.lr={args.lr}',
        f'training.loss.type=bce_with_logits',
        f'training.batch_size={args.batch_size}',
        f'training.max_epochs={args.epochs}',
        f'training.device={args.device}',
        'training.scheduler.type=none',
        'evaluation.metrics=[micro_f1,macro_f1,subset_accuracy,hamming_accuracy]',
    ]
    if args.hidden_dim is not None:
        cli_overrides.append(f'head.hidden_dim={args.hidden_dim}')
    if args.dropout is not None:
        cli_overrides.append(f'head.dropout={args.dropout}')
    if args.seed is not None:
        cli_overrides.append(f'training.seed={args.seed}')

    cfg = load_config(
        config_path if os.path.exists(config_path) else None,
        overrides=cli_overrides + (args.override or []),
    )

    if args.head_type == 'linear':
        # ── Linear probe path (sklearn) ──
        import sys
        sys.path.insert(0, _PROJECT_ROOT)
        from trl_bench.utils.downstream.linear_probe import LinearProbeRunner

        # Extract flat numpy arrays from datasets
        # CSVRelationDataset.__getitem__ returns a dict, not a tuple
        n_train = len(train_dataset)
        n_test = len(test_dataset)
        train_emb_list, train_lab_list = [], []
        for i in range(n_train):
            sample = train_dataset[i]
            train_emb_list.append(sample['embedding'].numpy())
            train_lab_list.append(sample['label'].numpy())
        test_emb_list, test_lab_list = [], []
        for i in range(n_test):
            sample = test_dataset[i]
            test_emb_list.append(sample['embedding'].numpy())
            test_lab_list.append(sample['label'].numpy())

        import numpy as np
        train_emb_np = np.array(train_emb_list)
        train_lab_np = np.array(train_lab_list)
        test_emb_np = np.array(test_emb_list)
        test_lab_np = np.array(test_lab_list)

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nRunning linear probe (sklearn, multi-label, fixed_C)...")
        runner = LinearProbeRunner(cfg)
        raw_test_results = runner.run(
            train_emb=train_emb_np,
            train_labels=train_lab_np,
            test_emb=test_emb_np,
            test_labels=test_lab_np,
            task_type='classification',
            metric_names=['micro_f1', 'macro_f1', 'subset_accuracy', 'hamming_accuracy'],
            multi_label=True,
        )
        # Strip test_ prefix for aggregation compatibility
        stripped = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        # Print in same format as MLP path (SLURM template scrapes these)
        parts = ["Final test:"]
        for k in sorted(raw_test_results.keys()):
            if k.startswith('test_') and k != 'test_loss':
                v = raw_test_results[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
        print("  ".join(parts))

        # Save results (top-level metric keys matching aggregator expectations)
        import json as _json
        results_file = os.path.join(output_dir, 'results.json')
        results = {
            'task_name': 'column_relation_prediction',
            'head_type': 'linear',
            **stripped,
            'data_stats': {'train': n_train, 'test': n_test},
        }
        with open(results_file, 'w') as f:
            _json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir}")

    elif args.head_type == 'dummy':
        # ── Dummy baseline path (multi-label, label statistics only) ──
        import numpy as np
        from trl_bench.utils.downstream.dummy_probe import DummyProbeRunner

        n_train = len(train_dataset)
        n_test = len(test_dataset)
        train_lab_list = []
        for i in range(n_train):
            sample = train_dataset[i]
            train_lab_list.append(sample['label'].numpy())
        test_lab_list = []
        for i in range(n_test):
            sample = test_dataset[i]
            test_lab_list.append(sample['label'].numpy())

        train_lab_np = np.array(train_lab_list)
        test_lab_np = np.array(test_lab_list)

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nRunning dummy baseline (multi-label, label statistics only)...")
        runner = DummyProbeRunner()
        raw_test_results = runner.run(
            train_labels=train_lab_np,
            test_labels=test_lab_np,
            task_type='classification',
            metric_names=['micro_f1', 'macro_f1', 'subset_accuracy', 'hamming_accuracy'],
            multi_label=True,
        )
        stripped = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        parts = ["Final test:"]
        for k in sorted(raw_test_results.keys()):
            if k.startswith('test_') and k != 'test_loss':
                v = raw_test_results[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
        print("  ".join(parts))

        import json as _json
        results_file = os.path.join(output_dir, 'results.json')
        results = {
            'task_name': 'column_relation_prediction',
            'head_type': 'dummy',
            **stripped,
            'data_stats': {'train': n_train, 'test': n_test},
        }
        with open(results_file, 'w') as f:
            _json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir}")

    else:
        # ── MLP path (existing Trainer pipeline) ──
        # Z-score normalize pair embeddings (fit on training pairs only)
        import numpy as _np
        from sklearn.preprocessing import StandardScaler
        train_pair_embs = _np.array([s['embedding'] for s in train_dataset.samples], dtype=_np.float32)
        emb_scaler = StandardScaler()
        emb_scaler.fit(train_pair_embs)

        for dataset in (train_dataset, test_dataset):
            for s in dataset.samples:
                s['embedding'] = emb_scaler.transform(
                    _np.array(s['embedding'], dtype=_np.float32).reshape(1, -1)
                )[0]

        # Rebuild dataloaders with normalized data
        train_loader = make_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = make_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"   Embedding z-score: applied (fit on {len(train_pair_embs)} train pairs)")

        # Create model from resolved config
        model = MLPHead(
            input_dim=cfg.head.input_dim,
            output_dim=cfg.head.output_dim,
            hidden_dim=cfg.head.hidden_dim,
            num_layers=cfg.head.num_layers,
            activation=cfg.head.activation,
            dropout=cfg.head.dropout,
            dropout_first=cfg.head.dropout_first,
        )

        print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Custom checkpoint format — save resolved head geometry for reproducibility
        def _relation_ckpt(model, optimizer, epoch, metrics, config):
            best_f1 = (metrics.get('val_micro_f1', metrics.get('train_micro_f1', 0))
                       if metrics else 0)
            return {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'epoch': epoch,
                'best_f1': best_f1,
                'args': vars(args),
                'head_config': {
                    'input_dim': cfg.head.input_dim,
                    'output_dim': cfg.head.output_dim,
                    'hidden_dim': cfg.head.hidden_dim,
                    'num_layers': cfg.head.num_layers,
                    'activation': cfg.head.activation,
                    'dropout': cfg.head.dropout,
                    'dropout_first': bool(cfg.head.dropout_first),
                },
            }

        task_spec = _RelationTaskSpec(cfg)

        trainer = Trainer(cfg, args.output_dir, checkpoint_format_fn=_relation_ckpt)
        trainer.setup(
            train_loader, None, test_loader,
            input_dim=train_dataset.pair_embedding_dim,
            output_dim=train_dataset.num_relations,
            model=model,
        )

        result = trainer.fit(task_spec=task_spec)

        # Final test evaluation on best checkpoint
        test_metrics = trainer.test(task_spec=task_spec)
        parts = ["Final test:"]
        for k in sorted(test_metrics.keys()):
            parts.append(f"{k}={test_metrics[k]:.4f}")
        print("  ".join(parts))

        # Save results.json (top-level metric keys matching aggregator expectations)
        stripped = {k.removeprefix('test_'): v for k, v in test_metrics.items()}
        results = {
            'task_name': 'column_relation_prediction',
            'head_type': 'mlp',
            'seed': cfg.training.seed,
            **stripped,
            'data_stats': {
                'train': len(train_dataset),
                'test': len(test_dataset),
                'num_relations': train_dataset.num_relations,
                'pair_embedding_dim': train_dataset.pair_embedding_dim,
            },
        }
        results_file = os.path.join(args.output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

        best_val = result['best_value']
        print(f"\n{'='*60}")
        print(f"Training complete! Best epoch: {result['best_epoch']}, "
              f"Best value: {best_val:.4f}" if best_val is not None else
              f"Training complete! {result['best_epoch']} epochs")
        print(f"Model saved to: {args.output_dir}/best_model.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
