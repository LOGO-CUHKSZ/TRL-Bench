#!/usr/bin/env python3
"""
Train Column Type Prediction Classifier

Trains a lightweight classifier on top of frozen column embeddings
to predict semantic column types.

Usage:
    python train_ct_mode4.py \
        --embeddings embeddings/column/tabsketchfm/sato.pkl \
        --dataset sato \
        --num_epochs 10 --learning_rate 5e-4
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_unified_embeddings(embeddings_path, labels_csv, class_to_idx=None):
    """Load unified v2.0 column embeddings and pair with labels from CSV.

    Args:
        embeddings_path: Path to unified embeddings .pkl file
        labels_csv: Path to CSV with (table_id, column_id, class) rows
        class_to_idx: Optional pre-built {class_name: index} mapping.
            If provided, uses this shared vocabulary instead of building
            one from the CSV. This ensures train/test label consistency
            when class sets differ between splits.

    Returns:
        table_embeddings: list of (num_cols, dim) arrays per table
        table_labels: list of (num_cols, num_classes) one-hot arrays per table
        table_masks: list of (num_cols,) masks per table
        table_ids: list of table id strings
        num_classes: total number of unique classes
    """
    print(f"Loading embeddings from: {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    # Build table_id -> column_embeddings map
    emb_map = {}
    for item in data:
        tid = item['table_id']
        ce = item['column_embeddings']
        if isinstance(ce, dict):
            sorted_keys = sorted(ce.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
            emb_map[tid] = np.array([ce[k] for k in sorted_keys])
        else:
            emb_map[tid] = np.array(ce)

    print(f"  Loaded {len(emb_map)} tables")

    # Load labels
    print(f"Loading labels from: {labels_csv}")
    df = pd.read_csv(labels_csv)
    if class_to_idx is None:
        classes = sorted(df['class'].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(class_to_idx)
    print(f"  {len(df)} rows, {num_classes} classes")

    # Group by table_id and build per-table arrays
    table_embeddings = []
    table_labels = []
    table_masks = []
    table_ids_out = []
    missing = 0

    for table_id, group in df.groupby('table_id'):
        tid_str = f'table_{table_id}' if isinstance(table_id, (int, np.integer)) else str(table_id)
        if tid_str not in emb_map:
            missing += 1
            continue

        emb = emb_map[tid_str]
        group_sorted = group.sort_values('column_id')

        n_cols = len(group_sorted)
        label_arr = np.zeros((n_cols, num_classes), dtype=np.float32)
        mask_arr = np.ones(n_cols, dtype=np.float32)
        col_embs = []

        for i, (_, row) in enumerate(group_sorted.iterrows()):
            col_id = int(row['column_id'])
            if col_id < emb.shape[0]:
                col_embs.append(emb[col_id])
                label_arr[i, class_to_idx[row['class']]] = 1.0
            else:
                col_embs.append(np.zeros(emb.shape[1], dtype=np.float32))
                mask_arr[i] = 0.0

        table_embeddings.append(np.array(col_embs, dtype=np.float32))
        table_labels.append(label_arr)
        table_masks.append(mask_arr)
        table_ids_out.append(tid_str)

    if missing:
        print(f"  Warning: {missing} tables not found in embeddings")
    print(f"  Matched {len(table_ids_out)} tables")

    return table_embeddings, table_labels, table_masks, table_ids_out, num_classes


class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for column embeddings with labels"""

    def __init__(self, embeddings, labels, masks, table_ids):
        self.embeddings = embeddings
        self.labels = labels
        self.labels_masks = masks
        self.table_ids = table_ids
        print(f"  Dataset: {len(self.embeddings)} tables, embedding dim={self.embeddings[0].shape[-1]}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embeddings': torch.FloatTensor(self.embeddings[idx]),
            'labels': torch.FloatTensor(self.labels[idx]),
            'labels_mask': torch.FloatTensor(self.labels_masks[idx]),
            'table_id': self.table_ids[idx]
        }


def collate_fn(batch):
    """Collate function to handle variable-length columns"""
    max_cols = max(item['embeddings'].shape[0] for item in batch)

    batch_size = len(batch)
    hidden_size = batch[0]['embeddings'].shape[1]
    num_types = batch[0]['labels'].shape[1]

    embeddings = torch.zeros(batch_size, max_cols, hidden_size)
    labels = torch.zeros(batch_size, max_cols, num_types)
    labels_masks = torch.zeros(batch_size, max_cols)
    table_ids = []

    for i, item in enumerate(batch):
        num_cols = item['embeddings'].shape[0]
        embeddings[i, :num_cols] = item['embeddings']
        labels[i, :num_cols] = item['labels']
        labels_masks[i, :num_cols] = item['labels_mask']
        table_ids.append(item['table_id'])

    return embeddings, labels, labels_masks, table_ids


from trl_bench.utils.downstream.heads import MLPHead
from trl_bench.utils.downstream.config import load_config
from trl_bench.utils.downstream.losses import MaskedBCELoss
from trl_bench.utils.downstream.trainer import Trainer, DefaultTaskSpec, seed_everything


def CTMode4Classifier(hidden_size=312, num_types=255, hidden_dim=256,
                      dropout=0.1, dropout_first=False):
    """Factory for CT Mode 4 classifier."""
    return MLPHead(input_dim=hidden_size, output_dim=num_types,
                   hidden_dim=hidden_dim, num_layers=2, dropout=dropout,
                   dropout_first=dropout_first)


def ct_forward_with_loss(model, embeddings, labels=None, labels_mask=None):
    """Forward pass returning (logits, loss).

    Kept for backward compatibility with evaluate_ct_mode4.py.
    When labels/labels_mask are None, loss is returned as None.
    """
    logits = model(embeddings)
    if labels is not None and labels_mask is not None:
        loss_fn = MaskedBCELoss()
        loss = loss_fn(logits, labels, labels_mask)
    else:
        loss = None
    return logits, loss


def compute_map(logits, labels, labels_mask):
    """Compute Mean Average Precision"""
    preds = torch.sigmoid(logits)
    preds_flat = preds.view(-1, preds.shape[-1])
    labels_flat = labels.view(-1, labels.shape[-1])
    mask_flat = labels_mask.view(-1)

    aps = []
    for i in range(len(preds_flat)):
        if mask_flat[i] == 0:
            continue
        pred = preds_flat[i].cpu().numpy()
        label = labels_flat[i].cpu().numpy()
        pos_indices = np.where(label == 1)[0]
        if len(pos_indices) == 0:
            continue
        ranked_indices = np.argsort(-pred)
        hits = 0
        ap_sum = 0
        for rank, idx in enumerate(ranked_indices):
            if label[idx] == 1:
                hits += 1
                ap_sum += hits / (rank + 1)
        if hits > 0:
            aps.append(ap_sum / hits)

    return np.mean(aps) if aps else 0.0


class CTTaskSpec(DefaultTaskSpec):
    """TaskSpec for column type prediction with masked BCE and MAP metric."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._masked_bce = MaskedBCELoss()

    def forward_and_loss(self, model, batch, device, criterion):
        embeddings, labels, labels_mask, _ = batch
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        labels_mask = labels_mask.to(device)

        logits = model(embeddings)
        loss = self._masked_bce(logits, labels, labels_mask)
        return loss, logits

    def extract_targets(self, batch):
        # Return (labels, mask) tuple for compute_metrics
        return (batch[1], batch[2])

    def compute_metrics(self, outputs, targets):
        # targets is (labels_list, masks_list) from extract_targets
        # We need to reconstruct for MAP computation
        labels, masks = targets
        logits_t = torch.tensor(outputs)
        labels_t = torch.tensor(labels)
        masks_t = torch.tensor(masks)
        map_score = compute_map(logits_t, labels_t, masks_t)

        # Compute micro/macro F1 (argmax prediction vs true class)
        preds_flat = torch.sigmoid(logits_t).view(-1, logits_t.shape[-1])
        labels_flat = labels_t.view(-1, labels_t.shape[-1])
        mask_flat = masks_t.view(-1)
        valid = mask_flat == 1
        if valid.any():
            y_pred = preds_flat[valid].argmax(dim=1).cpu().numpy()
            y_true = labels_flat[valid].argmax(dim=1).cpu().numpy()
            micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            micro_f1 = 0.0
            macro_f1 = 0.0

        return {'map': map_score, 'micro_f1': micro_f1, 'macro_f1': macro_f1}


def main():
    parser = argparse.ArgumentParser(description='Train column type prediction classifier')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to unified column embeddings .pkl file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., sato). Loads labels from {dataset}/train.csv and {dataset}/test.csv')
    parser.add_argument('--output_dir', type=str,
                        default='column_type_classifier',
                        help='Output directory for trained model')
    parser.add_argument('--batch_size', type=int,
                        default=20,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=None,
                        help='Learning rate (default: from YAML config)')
    parser.add_argument('--num_epochs', type=int,
                        default=2,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int,
                        default=100,
                        help='Warmup steps for learning rate scheduler')
    parser.add_argument('--dropout', type=float,
                        default=None,
                        help='Dropout rate (default: from YAML config)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int,
                        default=42,
                        help='Random seed')
    parser.add_argument('--wandb_project', type=str,
                        default='column-type-prediction',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str,
                        default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--config', type=str,
                        default='configs/downstream/column_type_prediction.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--override', type=str, action='append', default=[],
                        help='Config overrides')
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'linear', 'dummy'],
                        help='Probe type: mlp (PyTorch MLP), linear (sklearn), or dummy (majority/mean baseline)')

    args = parser.parse_args()

    # Set random seeds
    seed_everything(args.seed)

    # Resolve --output_dir relative to the current working directory, not the
    # script's location. trl-bench-run and the slurm dispatchers compose the
    # output path as `<results_dir>/evaluation/<task>/<model>/...` with
    # results_dir defaulting to `./results`; resolving relative to script_dir
    # (the script's own directory under src/trl_bench/tasks/...) used to land
    # output inside the source tree (src/trl_bench/tasks/column_type_prediction/
    # results/...), bypassing the user's --results-dir entirely.
    args.output_dir = os.path.abspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

    train_csv = f'{args.dataset}/train.csv'
    test_csv = f'{args.dataset}/test.csv'

    print("="*80)
    print("Column Type Prediction Classifier Training")
    print("="*80)
    print(f"Embeddings: {args.embeddings}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Build shared class vocabulary from union of train+test classes
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    all_classes = sorted(set(train_df['class'].unique()) | set(test_df['class'].unique()))
    shared_class_to_idx = {c: i for i, c in enumerate(all_classes)}
    num_classes = len(all_classes)
    print(f"Shared vocabulary: {num_classes} classes "
          f"(train={train_df['class'].nunique()}, test={test_df['class'].nunique()})")

    # Load datasets
    print("\nLoading training data...")
    train_emb, train_lab, train_mask, train_ids, _ = load_unified_embeddings(
        args.embeddings, train_csv, class_to_idx=shared_class_to_idx)
    print("\nLoading test data...")
    test_emb, test_lab, test_mask, test_ids, _ = load_unified_embeddings(
        args.embeddings, test_csv, class_to_idx=shared_class_to_idx)

    train_dataset = EmbeddingDataset(train_emb, train_lab, train_mask, train_ids)
    test_dataset = EmbeddingDataset(test_emb, test_lab, test_mask, test_ids)

    # Create dataloaders
    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    # Get dimensions from first sample
    hidden_size = train_dataset.embeddings[0].shape[-1]
    num_types = num_classes
    print(f"Hidden size: {hidden_size}")
    print(f"Num types: {num_types}")
    print()

    # Build config (head settings come from YAML, overridable via --override)
    project_root = str(Path(__file__).parent.parent.parent)
    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config

    # Detect which CLI flags the user actually passed (vs argparse defaults)
    _explicitly_passed = set()
    for token in sys.argv[1:]:
        if token.startswith('--'):
            flag = token.lstrip('-').split('=', 1)[0].replace('-', '_')
            _explicitly_passed.add(flag)

    # Values that MUST always override (computed from data or task-specific)
    always_overrides = [
        f'task_name=column_type_prediction',
        f'head.input_dim={hidden_size}',
        f'head.output_dim={num_types}',
        f'training.loss.type=masked_bce',
        'evaluation.metrics=[map,micro_f1,macro_f1]',
    ]

    # Values gated by _explicitly_passed — only override YAML when user
    # explicitly passed the flag, preventing argparse defaults from silently
    # replacing YAML values.
    _cli_map = {
        'learning_rate': f'training.optimizer.lr={args.learning_rate}',
        'batch_size':    f'training.batch_size={args.batch_size}',
        'num_epochs':    f'training.max_epochs={args.num_epochs}',
        'seed':          f'training.seed={args.seed}',
        'device':        f'training.device={args.device}',
        'dropout':       f'head.dropout={args.dropout}',
    }

    if os.path.exists(config_path):
        user_cli_overrides = [v for k, v in _cli_map.items() if k in _explicitly_passed]
        cfg = load_config(
            config_path,
            overrides=always_overrides + user_cli_overrides + (args.override or []),
        )
    else:
        # No config file — apply all overrides, filtering out None values
        all_cli_overrides = [v for v in _cli_map.values() if not v.endswith('=None')]
        cfg = load_config(
            None,
            overrides=always_overrides + all_cli_overrides + (args.override or []),
        )

    # Print resolved hyperparameters (after config merge, so values are final).
    # Note: DataLoaders and initial seed_everything() ran before config load
    # using argparse values.  Trainer.setup() re-seeds from cfg before model
    # init, so the config seed governs weight initialization and training.
    print(f"Resolved hyperparameters:")
    print(f"  Learning rate: {cfg.training.optimizer.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Max epochs:    {cfg.training.max_epochs}")
    print(f"  Seed:          {cfg.training.seed} (initial RNG: {args.seed})")
    print()

    # Set wandb config
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.training.logging.wandb = True
    cfg.training.logging.wandb_project = args.wandb_project
    if args.wandb_run_name:
        cfg.training.logging.wandb_run_name = args.wandb_run_name
    OmegaConf.set_struct(cfg, True)

    if args.head_type == 'linear':
        # ── Linear probe path (sklearn) ──
        # Flatten variable-length table data into per-column samples.
        # Each valid column becomes one (embedding, class_index) sample.
        # The CT task uses BCEWithLogitsLoss (multi-label) but evaluates via argmax,
        # so the linear probe treats it as multi-class classification directly.
        from trl_bench.utils.downstream.linear_probe import LinearProbeRunner

        def _flatten_ct(table_embs, table_labs, table_masks):
            """Flatten per-table variable-length data to per-column flat arrays."""
            all_emb, all_lab = [], []
            for emb, lab, mask in zip(table_embs, table_labs, table_masks):
                for i in range(len(mask)):
                    if mask[i] > 0:
                        all_emb.append(emb[i])
                        all_lab.append(np.argmax(lab[i]))  # one-hot -> class index
            return np.array(all_emb), np.array(all_lab)

        train_emb_flat, train_lab_flat = _flatten_ct(train_emb, train_lab, train_mask)
        test_emb_flat, test_lab_flat = _flatten_ct(test_emb, test_lab, test_mask)
        print(f"Flattened: train={len(train_emb_flat)} columns, test={len(test_emb_flat)} columns")

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        runner = LinearProbeRunner(cfg)
        raw_test_results = runner.run(
            train_emb=train_emb_flat,
            train_labels=train_lab_flat,
            test_emb=test_emb_flat,
            test_labels=test_lab_flat,
            task_type='classification',
            metric_names=['accuracy', 'micro_f1', 'macro_f1'],
        )
        # Strip test_ prefix for aggregation compatibility
        stripped = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

        # Print in SLURM-scrapable format
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
            'task_name': 'column_type_prediction',
            'head_type': 'linear',
            **stripped,
            'data_stats': {
                'train_columns': len(train_emb_flat),
                'test_columns': len(test_emb_flat),
                'num_types': num_types,
                'hidden_size': hidden_size,
            },
            'class_to_idx': shared_class_to_idx,
        }
        with open(results_file, 'w') as f:
            _json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir}")
        return

    elif args.head_type == 'dummy':
        # ── Dummy baseline path (label statistics only) ──
        from trl_bench.utils.downstream.dummy_probe import DummyProbeRunner

        def _flatten_ct_labels(table_labs, table_masks):
            """Flatten per-table variable-length labels to per-column class indices."""
            all_lab = []
            for lab, mask in zip(table_labs, table_masks):
                for i in range(len(mask)):
                    if mask[i] > 0:
                        all_lab.append(np.argmax(lab[i]))
            return np.array(all_lab)

        train_lab_flat = _flatten_ct_labels(train_lab, train_mask)
        test_lab_flat = _flatten_ct_labels(test_lab, test_mask)
        print(f"Flattened: train={len(train_lab_flat)} columns, test={len(test_lab_flat)} columns")

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nRunning dummy baseline (label statistics only)...")
        runner = DummyProbeRunner()
        raw_test_results = runner.run(
            train_labels=train_lab_flat,
            test_labels=test_lab_flat,
            task_type='classification',
            metric_names=['accuracy', 'micro_f1', 'macro_f1'],
            num_classes=num_classes,
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
            'task_name': 'column_type_prediction',
            'head_type': 'dummy',
            **stripped,
            'data_stats': {
                'train_columns': len(train_lab_flat),
                'test_columns': len(test_lab_flat),
                'num_types': num_types,
                'hidden_size': hidden_size,
            },
            'class_to_idx': shared_class_to_idx,
        }
        with open(results_file, 'w') as f:
            _json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir}")
        return

    # ── MLP path (existing Trainer pipeline) ──
    # Z-score normalize embeddings (fit on valid training columns only)
    from sklearn.preprocessing import StandardScaler
    all_train_cols = []
    for emb, mask in zip(train_emb, train_mask):
        for i in range(len(mask)):
            if mask[i] > 0:
                all_train_cols.append(emb[i])
    emb_scaler = StandardScaler()
    emb_scaler.fit(np.array(all_train_cols, dtype=np.float32))

    for emb_list in (train_emb, test_emb):
        for j, emb in enumerate(emb_list):
            emb_list[j] = emb_scaler.transform(emb.astype(np.float32))

    # Rebuild datasets with normalized embeddings
    train_dataset = EmbeddingDataset(train_emb, train_lab, train_mask, train_ids)
    test_dataset = EmbeddingDataset(test_emb, test_lab, test_mask, test_ids)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=args.batch_size, collate_fn=collate_fn)
    print(f"   Embedding z-score: applied (fit on {len(all_train_cols)} train columns)")

    # Initialize model from config (all head settings from YAML, overridable via --override)
    model = MLPHead(input_dim=hidden_size, output_dim=num_types,
                    hidden_dim=cfg.head.hidden_dim, num_layers=cfg.head.num_layers,
                    dropout=cfg.head.dropout, dropout_first=cfg.head.dropout_first)

    task_spec = CTTaskSpec(cfg)

    # Custom checkpoint format: best saves contain metadata, per-epoch saves
    # contain optimizer state + all metrics.
    def _ct_ckpt(model, optimizer, epoch, metrics, config, is_best=False):
        # Use val_* if available, fall back to train_* (normal runs have no val loader,
        # so trainer rewrites monitor from val_map -> train_map)
        def _best(key):
            return metrics.get(f'val_{key}', metrics.get(f'train_{key}', 0)) if metrics else 0

        # Save full head config so the evaluator can reconstruct any geometry
        head_cfg = {
            'hidden_size': hidden_size,
            'num_types': num_types,
            'hidden_dim': cfg.head.hidden_dim,
            'num_layers': cfg.head.num_layers,
            'dropout': cfg.head.dropout,
            'dropout_first': bool(cfg.head.dropout_first),
        }

        scaler_cfg = {
            'emb_scaler_mean': emb_scaler.mean_.tolist(),
            'emb_scaler_scale': emb_scaler.scale_.tolist(),
        }

        if is_best:
            return {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_map': _best('map'),
                'best_micro_f1': _best('micro_f1'),
                'best_macro_f1': _best('macro_f1'),
                **head_cfg,
                **scaler_cfg,
                'class_to_idx': shared_class_to_idx,
            }
        else:
            return {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'train_loss': metrics.get('train_loss', 0) if metrics else 0,
                'train_map': metrics.get('train_map', 0) if metrics else 0,
                'train_micro_f1': metrics.get('train_micro_f1', 0) if metrics else 0,
                **head_cfg,
                **scaler_cfg,
                'class_to_idx': shared_class_to_idx,
            }

    trainer = Trainer(cfg, args.output_dir, checkpoint_format_fn=_ct_ckpt)
    trainer.setup(
        train_dataloader, None, test_dataloader,
        input_dim=hidden_size, output_dim=num_types,
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
    import json as _json
    mlp_results = {
        'task_name': 'column_type_prediction',
        'head_type': 'mlp',
        'seed': cfg.training.seed,
        **stripped,
        'data_stats': {
            'train_tables': len(train_dataset),
            'test_tables': len(test_dataset),
            'num_types': num_types,
            'hidden_size': hidden_size,
        },
        'class_to_idx': shared_class_to_idx,
    }
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        _json.dump(mlp_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    best_val = result['best_value']
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best epoch: {result['best_epoch']}, Best value: {best_val:.4f}"
          if best_val is not None else
          f"{result['best_epoch']} epochs completed")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
