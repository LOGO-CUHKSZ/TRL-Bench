"""
Row Prediction Downstream Training Script (Trainer pipeline).

Trains an MLP head on frozen row-level embeddings using the unified
Trainer pipeline (shared with all other downstream tasks).

Usage:
    python train_downstream.py --embedding_dir embeddings/row_prediction/dae/openml_3
    python train_downstream.py --embedding_dir embeddings --config path/to/config.yaml
    python train_downstream.py --embedding_dir embeddings --label_column class --task classification
"""

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

from pathlib import Path

# ---------------------------------------------------------------------------
# Import bootstrapping: load local embedding_utils by explicit path BEFORE
# adding repo root to sys.path (which would shadow the local utils/ package).
# Same importlib pattern used by other task scripts.
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_emb_spec = importlib.util.spec_from_file_location(
    'embedding_utils', _script_dir / 'utils' / 'embedding_utils.py')
_emb_mod = importlib.util.module_from_spec(_emb_spec)
_emb_spec.loader.exec_module(_emb_mod)
load_embeddings = _emb_mod.load_embeddings
get_available_labels = _emb_mod.get_available_labels
load_embeddings_for_label = _emb_mod.load_embeddings_for_label

# Now add repo root for utils.downstream.* imports
_repo_root = str(_script_dir.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from torch.utils.data import TensorDataset, DataLoader
from trl_bench.utils.downstream.trainer import Trainer as TaskTrainer
from trl_bench.utils.downstream.config import load_config
from omegaconf import open_dict


def _regression_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics on a single target vector."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr, spearmanr

    mse_value = float(mean_squared_error(y_true, y_pred))
    if len(y_true) < 2:
        pearson_val, spearman_val = 0.0, 0.0
    else:
        pr = pearsonr(y_true, y_pred)
        sr = spearmanr(y_true, y_pred)
        pearson_val = 0.0 if np.isnan(pr.statistic) else float(pr.statistic)
        spearman_val = 0.0 if np.isnan(sr.statistic) else float(sr.statistic)
    return {
        'loss': mse_value,
        'mse': mse_value,
        'rmse': float(np.sqrt(mse_value)),
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'pearson_r': pearson_val,
        'spearman_r': spearman_val,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train downstream MLP on frozen row embeddings (Trainer pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect task type)
  python train_downstream.py --embedding_dir embeddings/row_prediction/dae/openml_3

  # Specify config and label column
  python train_downstream.py --embedding_dir embeddings --config configs/downstream/row_prediction.yaml --label_column class

  # Force regression task
  python train_downstream.py --embedding_dir embeddings --task regression
        """
    )

    parser.add_argument('--embedding_dir', type=str, default='embeddings',
                        help='Directory containing embeddings (default: embeddings)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation/row_prediction',
                        help='Directory to save results (default: results/evaluation/row_prediction)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (default: configs/downstream/row_prediction.yaml)')
    parser.add_argument('--label_column', type=str, default=None,
                        help='Specific label to predict. Default: iterate over all available labels.')
    parser.add_argument('--task', type=str, default='auto',
                        choices=['auto', 'classification', 'regression'],
                        help='Fallback task type when metadata does not specify task_type (default: auto)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config training.seed)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (stored in result JSON for aggregation)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (stored in result JSON for aggregation)')
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'linear', 'dummy'],
                        help='Probe type: mlp (PyTorch MLP), linear (sklearn), or dummy (majority/mean baseline)')
    parser.add_argument('--variant', type=str, default=None,
                        help='Embedding variant label (e.g., pca128_from768). Written to result JSON for aggregation.')

    args = parser.parse_args()

    config_path = args.config or os.path.join(_repo_root, 'configs', 'downstream', 'row_prediction.yaml')

    if not args.quiet:
        print("=" * 80)
        print("Row Prediction — Trainer Pipeline")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Embedding dir: {args.embedding_dir}")
        print(f"  Output dir:    {args.output_dir}")
        print(f"  Config:        {config_path}")
        print(f"  Task:          {args.task}")
        print("=" * 80)

    # ========================================================================
    # Resolve target labels
    # ========================================================================
    available_labels, label_task_types = get_available_labels(args.embedding_dir)

    if args.label_column:
        if args.label_column not in available_labels:
            print(f"Error: '{args.label_column}' not found. Available: {available_labels}")
            sys.exit(1)
        target_labels = [args.label_column]
    elif len(available_labels) > 1:
        target_labels = available_labels
        if not args.quiet:
            print(f"\nMulti-label mode: training on {len(target_labels)} labels: {target_labels}")
    elif len(available_labels) == 1:
        target_labels = available_labels
    else:
        target_labels = [None]  # no label info, load_embeddings will handle

    # Read label_filename_map from metadata for consistent output dirs
    _full_meta = {}
    _meta_json = os.path.join(args.embedding_dir, "metadata.json")
    if os.path.exists(_meta_json):
        with open(_meta_json, 'r') as f:
            _full_meta = json.load(f)
    label_filename_map = _full_meta.get('label_filename_map', {})

    all_label_results = {}

    for label_col in target_labels:
        if len(target_labels) > 1:
            print(f"\n{'=' * 80}")
            print(f"Training for label: '{label_col}'")
            print(f"{'=' * 80}")

        # ----------------------------------------------------------------
        # 1. Load embeddings
        # ----------------------------------------------------------------
        if not args.quiet:
            print(f"\n1. Loading embeddings from {args.embedding_dir}...")

        try:
            if label_col is not None:
                (train_emb, train_labels, val_emb, val_labels,
                 test_emb, test_labels, metadata) = \
                    load_embeddings_for_label(args.embedding_dir, label_col, verbose=not args.quiet)
            else:
                (train_emb, train_labels, val_emb, val_labels,
                 test_emb, test_labels, metadata) = \
                    load_embeddings(args.embedding_dir, verbose=not args.quiet)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nError: {exc}")
            if label_col is not None and len(target_labels) > 1:
                print(f"Skipping label '{label_col}'...")
                continue
            sys.exit(1)

        if train_labels is None or test_labels is None:
            print(f"\nError: No labels found for '{label_col}'")
            if len(target_labels) > 1:
                continue
            sys.exit(1)

        # ----------------------------------------------------------------
        # 2. Drop rows with NaN labels
        # ----------------------------------------------------------------
        if train_labels.dtype.kind == 'f':
            train_nan = np.isnan(train_labels)
            test_nan = np.isnan(test_labels)
        else:
            # Classification: encode_label_column uses -1 as the NaN sentinel.
            # Legitimate -1 class labels don't occur in our datasets.
            train_nan = (train_labels == -1)
            test_nan = (test_labels == -1)
        if train_nan.any() or test_nan.any():
            n_train_drop = int(train_nan.sum())
            n_test_drop = int(test_nan.sum())
            train_emb = train_emb[~train_nan]
            train_labels = train_labels[~train_nan]
            test_emb = test_emb[~test_nan]
            test_labels = test_labels[~test_nan]
            if not args.quiet:
                print(f"   Dropped NaN labels: {n_train_drop} train, {n_test_drop} test "
                      f"({n_train_drop + n_test_drop} total)")
            if len(train_labels) == 0 or len(test_labels) == 0:
                print(f"\nError: No samples remaining after dropping NaN labels for '{label_col}'")
                if len(target_labels) > 1:
                    continue
                sys.exit(1)

        # Normalize partial val (labels but no embeddings, or vice versa)
        if (val_emb is None) != (val_labels is None):
            val_emb, val_labels = None, None

        # Drop NaN labels from val split (when present)
        if val_labels is not None:
            if val_labels.dtype.kind == 'f':
                val_nan = np.isnan(val_labels)
            else:
                val_nan = (val_labels == -1)
            if val_nan.any():
                n_val_drop = int(val_nan.sum())
                val_emb = val_emb[~val_nan]
                val_labels = val_labels[~val_nan]
                if not args.quiet:
                    print(f"   Dropped NaN val labels: {n_val_drop}")
                if len(val_labels) == 0:
                    val_emb, val_labels = None, None

        # ----------------------------------------------------------------
        # 3. Determine task type
        # ----------------------------------------------------------------
        if label_col and label_col in label_task_types and label_task_types[label_col]:
            detected_task = label_task_types[label_col]
            if not args.quiet:
                print(f"\n2. Task type (from metadata): {detected_task.upper()}")
        elif args.task != 'auto':
            detected_task = args.task
            if not args.quiet:
                print(f"\n2. Task type (manual): {detected_task.upper()}")
        else:
            print(f"WARNING: No task_type in metadata for '{label_col}'; "
                  f"using heuristic auto-detection.")
            unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
            is_float = train_labels.dtype in [np.float32, np.float64]
            num_unique = len(unique_labels)
            total_samples = len(train_labels) + len(test_labels)
            uniqueness_ratio = num_unique / total_samples

            if is_float and uniqueness_ratio > 0.1:
                detected_task = 'regression'
            elif num_unique > 20:
                detected_task = 'regression'
            else:
                detected_task = 'classification'

            if detected_task == 'regression' and num_unique <= 100 and not is_float:
                print(f"WARNING: Heuristic chose regression but labels are integers with "
                      f"{num_unique} unique values. Use --task classification to override.")

            if not args.quiet:
                print(f"\n2. Auto-detected task type: {detected_task.upper()}")
                print(f"   Unique values: {num_unique}")
                print(f"   Uniqueness ratio: {uniqueness_ratio:.4f}")
                print(f"   Label dtype: {train_labels.dtype}")

        # ----------------------------------------------------------------
        # 4. Load config (per-label, with task_type override)
        # ----------------------------------------------------------------
        metrics_list = ['accuracy', 'weighted_f1', 'macro_f1', 'auroc'] if detected_task == 'classification' else ['mse', 'rmse', 'r2', 'mae', 'pearson_r', 'spearman_r']

        seed_overrides = [f'training.seed={args.seed}'] if args.seed is not None else []
        cfg = load_config(
            config_path,
            overrides=[
                f'task_name=row_prediction_{label_col}' if label_col else 'task_name=row_prediction',
                f'task_type={detected_task}',
            ] + seed_overrides,
        )
        with open_dict(cfg):
            cfg.evaluation.metrics = metrics_list

        # ----------------------------------------------------------------
        # 5. Label encoding and data preparation
        # ----------------------------------------------------------------
        n_classes = None
        label_map = None
        val_labels_enc = None

        if detected_task == 'classification':
            train_classes = np.unique(train_labels)
            n_classes = len(train_classes)
            label_map = {c: i for i, c in enumerate(train_classes)}

            # Validate: warn + skip label if test has unseen classes
            test_classes = set(np.unique(test_labels))
            unseen = test_classes - set(train_classes)
            if unseen:
                print(f"WARNING: Test set for '{label_col}' has {len(unseen)} class(es) "
                      f"not in train: {sorted(unseen)}. Skipping this label.")
                continue

            train_labels_enc = np.array([label_map[l] for l in train_labels])
            test_labels_enc = np.array([label_map[l] for l in test_labels])

            # Encode val labels (degrade gracefully if unseen classes)
            if val_emb is not None and val_labels is not None:
                val_unseen = set(np.unique(val_labels)) - set(train_classes)
                if val_unseen:
                    print(f"WARNING: Val set for '{label_col}' has unseen class(es) "
                          f"{sorted(val_unseen)}; discarding canonical val.")
                    val_emb, val_labels = None, None
                else:
                    val_labels_enc = np.array([label_map[l] for l in val_labels])

            output_dim = n_classes
            label_tensor_fn = torch.LongTensor
        else:
            train_labels_enc = train_labels.astype(np.float32)
            test_labels_enc = test_labels.astype(np.float32)
            if val_emb is not None and val_labels is not None:
                val_labels_enc = val_labels.astype(np.float32)
            output_dim = 1
            label_tensor_fn = torch.FloatTensor

        # ----------------------------------------------------------------
        # 5b. Per-label output directory
        # ----------------------------------------------------------------
        if label_col is not None:
            sanitized = label_filename_map.get(label_col, label_col)
            label_output_dir = os.path.join(args.output_dir, sanitized)
        else:
            label_output_dir = args.output_dir
        os.makedirs(label_output_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # 6. Canonical val availability
        # ----------------------------------------------------------------
        has_canonical_val = (val_emb is not None and val_labels is not None
                            and val_labels_enc is not None
                            and len(val_emb) > 0)

        # ----------------------------------------------------------------
        # 7. Linear probe path (sklearn) or MLP path (PyTorch Trainer)
        # ----------------------------------------------------------------
        if args.head_type == 'linear':
            from trl_bench.utils.downstream.linear_probe import LinearProbeRunner

            if not args.quiet:
                print(f"\n3. Running linear probe (sklearn)...")

            runner = LinearProbeRunner(cfg)
            raw_test_results = runner.run(
                train_emb=train_emb,
                train_labels=train_labels_enc,
                test_emb=test_emb,
                test_labels=test_labels_enc,
                task_type=detected_task,
                metric_names=metrics_list,
                val_emb=val_emb if has_canonical_val else None,
                val_labels=val_labels_enc if has_canonical_val else None,
            )
            # Strip test_ prefix for aggregation compatibility
            test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

            data_stats = {
                'train': len(train_emb),
                'test': len(test_emb),
                'input_dim': train_emb.shape[1],
                'n_classes': n_classes if detected_task == 'classification' else None,
            }
            if has_canonical_val:
                data_stats['val'] = len(val_emb)

            results = {
                'task_name': f'row_prediction_{label_col}' if label_col else 'row_prediction',
                'task': 'row_prediction',
                'task_type': detected_task,
                'head_type': 'linear',
                'seed': cfg.training.seed,
                'model': args.model,
                'dataset': args.dataset,
                'variant': args.variant,
                'label_column': label_col,
                'test_results': test_results,
                'data_stats': data_stats,
            }
            if detected_task == 'classification' and label_map is not None:
                results['label_map'] = {str(k): int(v) for k, v in label_map.items()}

        elif args.head_type == 'dummy':
            from trl_bench.utils.downstream.dummy_probe import DummyProbeRunner

            if not args.quiet:
                print(f"\n3. Running dummy baseline (label statistics only)...")

            runner = DummyProbeRunner()
            raw_test_results = runner.run(
                train_labels=train_labels_enc,
                test_labels=test_labels_enc,
                task_type=detected_task,
                metric_names=metrics_list,
            )
            test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}

            data_stats = {
                'train': len(train_emb),
                'test': len(test_emb),
                'input_dim': train_emb.shape[1],
                'n_classes': n_classes if detected_task == 'classification' else None,
            }
            if has_canonical_val:
                data_stats['val'] = len(val_emb)

            results = {
                'task_name': f'row_prediction_{label_col}' if label_col else 'row_prediction',
                'task': 'row_prediction',
                'task_type': detected_task,
                'head_type': 'dummy',
                'seed': cfg.training.seed,
                'model': args.model,
                'dataset': args.dataset,
                'variant': args.variant,
                'label_column': label_col,
                'test_results': test_results,
                'data_stats': data_stats,
            }
            if detected_task == 'classification' and label_map is not None:
                results['label_map'] = {str(k): int(v) for k, v in label_map.items()}

        else:
            # MLP path (Trainer pipeline) with z-score normalization
            from sklearn.preprocessing import StandardScaler

            label_scaler = None
            scaled_test_results = None

            if has_canonical_val:
                # Use canonical val split — train on full train set
                emb_scaler = StandardScaler()
                train_emb_s = emb_scaler.fit_transform(train_emb)
                val_emb_s = emb_scaler.transform(val_emb)
                test_emb_s = emb_scaler.transform(test_emb)
                train_eval_emb = train_emb_s
                train_eval_labels = train_labels_enc

                train_labels_fit = train_labels_enc
                val_labels_fit = val_labels_enc
                test_labels_fit = test_labels_enc
                if detected_task == 'regression':
                    label_scaler = StandardScaler()
                    train_labels_fit = label_scaler.fit_transform(
                        train_labels_enc.reshape(-1, 1)
                    ).astype(np.float32).ravel()
                    val_labels_fit = label_scaler.transform(
                        val_labels_enc.reshape(-1, 1)
                    ).astype(np.float32).ravel()
                    test_labels_fit = label_scaler.transform(
                        test_labels_enc.reshape(-1, 1)
                    ).astype(np.float32).ravel()

                train_ds = TensorDataset(torch.FloatTensor(train_emb_s),
                                         label_tensor_fn(train_labels_fit))
                val_ds = TensorDataset(torch.FloatTensor(val_emb_s),
                                       label_tensor_fn(val_labels_fit))
                val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

                n_train_actual = len(train_emb)
                n_val_actual = len(val_emb)
            else:
                # Fallback for v1 data: split 10% from train
                np.random.seed(cfg.training.seed)
                n_val = int(len(train_emb) * 0.1)
                indices = np.random.permutation(len(train_emb))
                val_idx, train_idx = indices[:n_val], indices[n_val:]

                # Fit scaler on train partition only (no val leakage)
                emb_scaler = StandardScaler()
                train_emb_s = emb_scaler.fit_transform(train_emb[train_idx])
                test_emb_s = emb_scaler.transform(test_emb)
                train_eval_emb = train_emb_s
                train_eval_labels = train_labels_enc[train_idx]

                train_labels_fit = train_labels_enc[train_idx]
                test_labels_fit = test_labels_enc
                if detected_task == 'regression':
                    label_scaler = StandardScaler()
                    train_labels_fit = label_scaler.fit_transform(
                        train_labels_enc[train_idx].reshape(-1, 1)
                    ).astype(np.float32).ravel()
                    test_labels_fit = label_scaler.transform(
                        test_labels_enc.reshape(-1, 1)
                    ).astype(np.float32).ravel()

                train_ds = TensorDataset(torch.FloatTensor(train_emb_s),
                                         label_tensor_fn(train_labels_fit))

                if n_val > 0:
                    val_emb_s = emb_scaler.transform(train_emb[val_idx])
                    val_labels_fit = train_labels_enc[val_idx]
                    if detected_task == 'regression':
                        val_labels_fit = label_scaler.transform(
                            train_labels_enc[val_idx].reshape(-1, 1)
                        ).astype(np.float32).ravel()
                    val_ds = TensorDataset(torch.FloatTensor(val_emb_s),
                                           label_tensor_fn(val_labels_fit))
                    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
                else:
                    val_loader = None

                n_train_actual = len(train_idx)
                n_val_actual = n_val

            test_ds = TensorDataset(torch.FloatTensor(test_emb_s),
                                    label_tensor_fn(test_labels_fit))
            train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

            if not args.quiet:
                print(f"\n3. Training MLP head...")

            trainer = TaskTrainer(cfg, str(label_output_dir))
            trainer.setup(
                train_loader, val_loader, test_loader,
                input_dim=train_emb.shape[1],
                output_dim=output_dim,
            )
            fit_result = trainer.fit()
            raw_test_results = trainer.test()
            # Strip test_ prefix for aggregation compatibility
            test_results = {k.removeprefix('test_'): v for k, v in raw_test_results.items()}
            train_results = None
            if detected_task == 'regression' and label_scaler is not None:
                scaled_test_results = test_results.copy()
                train_preds = []
                train_eval_ds = TensorDataset(torch.FloatTensor(train_eval_emb))
                train_eval_loader = DataLoader(train_eval_ds, batch_size=cfg.training.batch_size, shuffle=False)
                trainer.model.eval()
                with torch.no_grad():
                    for xb, in train_eval_loader:
                        logits = trainer.model(xb.to(trainer.device))
                        train_preds.append(logits.squeeze(-1).cpu().numpy())
                train_pred_scaled = np.concatenate(train_preds, axis=0) if train_preds else np.array([], dtype=np.float32)
                train_pred = label_scaler.inverse_transform(
                    train_pred_scaled.reshape(-1, 1)
                ).ravel()
                train_results = _regression_metrics_dict(train_eval_labels, train_pred)

                test_preds = []
                with torch.no_grad():
                    for xb, _ in test_loader:
                        logits = trainer.model(xb.to(trainer.device))
                        test_preds.append(logits.squeeze(-1).cpu().numpy())
                test_pred_scaled = np.concatenate(test_preds, axis=0) if test_preds else np.array([], dtype=np.float32)
                test_pred = label_scaler.inverse_transform(
                    test_pred_scaled.reshape(-1, 1)
                ).ravel()
                test_results = _regression_metrics_dict(test_labels_enc, test_pred)

            data_stats = {
                'train': n_train_actual,
                'test': len(test_emb),
                'input_dim': train_emb.shape[1],
                'n_classes': n_classes if detected_task == 'classification' else None,
            }
            if has_canonical_val:
                data_stats['val'] = n_val_actual

            results = {
                'task_name': f'row_prediction_{label_col}' if label_col else 'row_prediction',
                'task': 'row_prediction',
                'task_type': detected_task,
                'head_type': 'mlp',
                'seed': cfg.training.seed,
                'model': args.model,
                'dataset': args.dataset,
                'variant': args.variant,
                'label_column': label_col,
                'test_results': test_results,
                'training': {
                    'best_epoch': fit_result['best_epoch'],
                    'best_value': fit_result['best_value']
                        if fit_result['best_value'] is not None and math.isfinite(fit_result['best_value'])
                        else None,
                    'total_epochs': len(fit_result['history']),
                },
                'data_stats': data_stats,
            }
            if detected_task == 'classification' and label_map is not None:
                results['label_map'] = {str(k): int(v) for k, v in label_map.items()}
            if scaled_test_results is not None:
                results['train_results'] = train_results
                results['scaled_test_results'] = scaled_test_results
                results['target_zscore'] = True
                results['target_zscore_split_mode'] = 'canonical_val' if has_canonical_val else 'fallback_train_split'
                results['target_scaler'] = {
                    'mean': float(label_scaler.mean_[0]),
                    'scale': float(label_scaler.scale_[0]),
                }

        results_file = os.path.join(label_output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"\n4. Results saved to {label_output_dir}/")
            print(f"   {results_file}")
            if args.head_type == 'mlp':
                print(f"   {os.path.join(label_output_dir, 'best_model.pt')}")

        # Track for final summary
        all_label_results[label_col] = {
            'task': detected_task,
            'test_results': test_results,
            'output_dir': label_output_dir,
        }

    # ====================================================================
    # Final summary
    # ====================================================================
    if not all_label_results:
        print("\n" + "=" * 80)
        print(f"ERROR: No labels were successfully trained!")
        print(f"  All {len(target_labels)} label(s) failed. Check errors above.")
        print("=" * 80)
        return 1

    print("\n" + "=" * 80)
    n_failed = len(target_labels) - len(all_label_results)
    if n_failed > 0:
        print(f"Downstream training completed with {n_failed} label(s) skipped.")
    else:
        print(f"Downstream training completed!")
    for lc, info in all_label_results.items():
        label_str = f"'{lc}'" if lc else "(default)"
        tr = info['test_results']
        if info['task'] == 'classification':
            acc = tr.get('accuracy', 'N/A')
            print(f"  {label_str}: accuracy {acc:.4f}" if isinstance(acc, float) else f"  {label_str}: accuracy {acc}")
        else:
            r2_val = tr.get('r2', 'N/A')
            print(f"  {label_str}: R² {r2_val:.4f}" if isinstance(r2_val, float) else f"  {label_str}: R² {r2_val}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
