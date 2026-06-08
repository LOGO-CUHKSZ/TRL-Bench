"""
Shared linear probe evaluation tier for all downstream tasks.

Uses sklearn convex solvers (LogisticRegression, Ridge) on frozen embeddings
with z-score normalization and optional C/alpha sweep. Deterministic — no seed
averaging needed.

Usage from any task script:
    runner = LinearProbeRunner(cfg)
    results = runner.run(train_emb, train_labels, test_emb, test_labels,
                         task_type='classification', metric_names=['accuracy', 'macro_f1'])
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from .metrics import compute_metrics, METRIC_DIRECTIONS

logger = logging.getLogger(__name__)


class LinearProbeRunner:
    """Sklearn-based linear probe for frozen embedding evaluation.

    Supports binary/multi-class classification (LogisticRegression),
    multi-label classification (OneVsRestClassifier), and regression (Ridge).
    """

    def __init__(self, config: DictConfig):
        lp = config.get('linear_probe', {})
        self.C_values = list(lp.get('C_values', [0.01, 0.1, 1.0, 10.0, 100.0]))
        self.alpha_values = list(lp.get('alpha_values', [0.01, 0.1, 1.0, 10.0, 100.0]))
        self.fixed_C = float(lp.get('fixed_C', 1.0))
        self.fixed_alpha = float(lp.get('fixed_alpha', 1.0))
        self.max_iter = int(lp.get('max_iter', 5000))
        self.normalize = bool(lp.get('normalize', True))
        self.sweep = bool(lp.get('sweep', True))
        self.refit_trainval = bool(lp.get('refit_trainval', True))
        self.seed = int(config.get('training', {}).get('seed', 42))

    def run(
        self,
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray,
        test_labels: np.ndarray,
        task_type: str,
        metric_names: List[str],
        val_emb: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        multi_label: bool = False,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Run linear probe: normalize, optionally sweep, fit, evaluate.

        Args:
            train_emb: Training embeddings (N_train, D).
            train_labels: Training labels (N_train,) or (N_train, C) for multi-label.
            test_emb: Test embeddings (N_test, D).
            test_labels: Test labels.
            task_type: 'classification' or 'regression'.
            metric_names: Metric names from METRIC_REGISTRY.
            val_emb: Optional validation embeddings for C/alpha sweep.
            val_labels: Optional validation labels.
            multi_label: Whether this is a multi-label classification task.
            threshold: Threshold for multi-label metrics.

        Returns:
            Dict with test_<metric> keys (matching Trainer.test() format)
            plus linear_probe_meta with sweep details.
        """
        # --- Step 1: Val split (before scaler to prevent leakage) ---
        has_val = val_emb is not None and val_labels is not None
        do_sweep = self.sweep and not (multi_label and not has_val)

        # Guard: disable sweep for very small training sets where a 10% split
        # would leave too few samples for reliable fitting
        if do_sweep and not has_val and len(train_emb) < 20:
            logger.warning("Training set too small (%d samples) for val split; "
                           "disabling sweep, using fixed hyperparameter",
                           len(train_emb))
            do_sweep = False

        if not has_val and do_sweep:
            train_emb, train_labels, val_emb, val_labels = self._split_val(
                train_emb, train_labels, task_type)
            has_val = True

        # --- Step 2: Z-score normalization ---
        # Keep raw copies for the refit step (Step 4) to avoid double-scaling.
        raw_train_emb = train_emb
        raw_train_labels = train_labels
        raw_val_emb = val_emb
        raw_val_labels = val_labels
        raw_test_emb = test_emb

        scaler = None
        label_scaler = None
        if self.normalize:
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_emb)
            if has_val:
                val_emb = scaler.transform(val_emb)
            test_emb = scaler.transform(test_emb)

            if task_type == 'regression':
                label_scaler = StandardScaler()
                train_labels = label_scaler.fit_transform(
                    train_labels.reshape(-1, 1)).ravel()
                if has_val:
                    val_labels = label_scaler.transform(
                        val_labels.reshape(-1, 1)).ravel()

        # --- Step 3: C/alpha sweep ---
        if do_sweep and has_val:
            best_hp = self._sweep(
                train_emb, train_labels, val_emb, val_labels,
                task_type, multi_label, metric_names, threshold)
        else:
            best_hp = self.fixed_alpha if task_type == 'regression' else self.fixed_C
            if not do_sweep and multi_label and not (val_emb is not None):
                logger.info("Multi-label without val split: skipping sweep, using fixed_C=%.4f", best_hp)

        # --- Step 4: Final fit ---
        if self.refit_trainval and has_val:
            # Concatenate RAW (unscaled) data, then fit a fresh scaler on
            # the combined set to avoid double-scaling.
            final_train_emb = np.concatenate([raw_train_emb, raw_val_emb], axis=0)
            final_train_labels = np.concatenate([raw_train_labels, raw_val_labels], axis=0)
            final_train_labels_raw = final_train_labels if task_type == 'regression' else None
            if self.normalize:
                scaler = StandardScaler()
                final_train_emb = scaler.fit_transform(final_train_emb)
                test_emb = scaler.transform(raw_test_emb)
                if task_type == 'regression':
                    label_scaler = StandardScaler()
                    final_train_labels = label_scaler.fit_transform(
                        final_train_labels.reshape(-1, 1)).ravel()
        else:
            final_train_emb = train_emb
            final_train_labels = train_labels
            final_train_labels_raw = raw_train_labels if task_type == 'regression' else None

        model = self._build_model(task_type, multi_label, best_hp)
        model.fit(final_train_emb, final_train_labels)

        # --- Step 5: Predict on test (and train for regression metrics) ---
        train_outputs = None
        test_outputs = self._get_outputs(model, test_emb, task_type, multi_label)
        if task_type == 'regression':
            train_outputs = self._get_outputs(model, final_train_emb, task_type, multi_label)

        # Inverse-transform regression targets for metric computation
        test_labels_for_metrics = test_labels
        if task_type == 'regression' and label_scaler is not None:
            train_outputs = label_scaler.inverse_transform(
                train_outputs.reshape(-1, 1)).ravel()
            test_outputs = label_scaler.inverse_transform(
                test_outputs.reshape(-1, 1)).ravel()

        # --- Step 6: Compute metrics ---
        test_metrics = compute_metrics(test_outputs, test_labels_for_metrics,
                                       metric_names, threshold=threshold)
        train_metrics = None
        if task_type == 'regression':
            train_metrics = compute_metrics(train_outputs, final_train_labels_raw,
                                            metric_names, threshold=threshold)

        # Format as test_<metric> to match Trainer.test() output
        results = {f'test_{k}': v for k, v in test_metrics.items()}
        if train_metrics is not None:
            results.update({f'train_{k}': v for k, v in train_metrics.items()})
            results['train_loss'] = train_metrics.get(metric_names[0])
        results['test_loss'] = None  # not applicable for convex solver

        results['linear_probe_meta'] = {
            'selected_C': float(best_hp) if task_type == 'classification' else None,
            'selected_alpha': float(best_hp) if task_type == 'regression' else None,
            'sweep': do_sweep,
            'normalize': self.normalize,
            'refit_trainval': self.refit_trainval,
            'multi_label': multi_label,
            'max_iter': self.max_iter,
            'train_samples': len(final_train_emb),
            'test_samples': len(test_emb),
            'input_dim': train_emb.shape[1] if train_emb.ndim == 2 else None,
        }

        return results

    def _split_val(self, train_emb, train_labels, task_type):
        """Split 10% from train as val. Stratified for single-label classification."""
        n = len(train_emb)
        n_val = max(1, int(0.1 * n))
        rng = np.random.RandomState(self.seed)

        if task_type == 'classification' and train_labels.ndim == 1:
            # Stratified split — fall back to random if classes are too small
            from sklearn.model_selection import train_test_split
            try:
                tr_emb, v_emb, tr_lab, v_lab = train_test_split(
                    train_emb, train_labels, test_size=n_val,
                    random_state=self.seed, stratify=train_labels)
                return tr_emb, tr_lab, v_emb, v_lab
            except ValueError:
                logger.warning("Stratified split failed (too few samples per class), "
                               "falling back to random split")
        # Random (non-stratified) split
        indices = rng.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return (train_emb[train_idx], train_labels[train_idx],
                train_emb[val_idx], train_labels[val_idx])

    def _sweep(self, train_emb, train_labels, val_emb, val_labels,
               task_type, multi_label, metric_names, threshold):
        """Sweep C (classification) or alpha (regression) on val set."""
        values = self.alpha_values if task_type == 'regression' else self.C_values
        primary_metric = metric_names[0]
        higher_is_better = METRIC_DIRECTIONS.get(primary_metric, 'max') == 'max'

        best_val = float('-inf') if higher_is_better else float('inf')
        best_hp = values[0]

        for hp in values:
            model = self._build_model(task_type, multi_label, hp)
            model.fit(train_emb, train_labels)
            outputs = self._get_outputs(model, val_emb, task_type, multi_label)
            metrics = compute_metrics(outputs, val_labels,
                                      [primary_metric], threshold=threshold)
            val_score = metrics[primary_metric]

            if (higher_is_better and val_score > best_val) or \
               (not higher_is_better and val_score < best_val):
                best_val = val_score
                best_hp = hp

        hp_name = 'alpha' if task_type == 'regression' else 'C'
        logger.info("Sweep selected %s=%.4f (val %s=%.4f)",
                    hp_name, best_hp, primary_metric, best_val)
        return best_hp

    def _build_model(self, task_type, multi_label, hp_value):
        """Build the sklearn model with given hyperparameter."""
        if task_type == 'classification':
            base = LogisticRegression(
                solver='lbfgs', penalty='l2', C=hp_value,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            if multi_label:
                return OneVsRestClassifier(base)
            return base
        else:
            return Ridge(alpha=hp_value, random_state=self.seed)

    def _get_outputs(self, model, emb, task_type, multi_label):
        """Get model outputs in a format compatible with compute_metrics().

        For single-label classification: predict_proba() (N, C) — metrics.py
            uses argmax for accuracy, argmax for F1.
        For multi-label classification: decision_function() (N, C) — metrics.py
            applies expit + threshold internally.
        For regression: predict() (N,).
        """
        if task_type == 'classification':
            if multi_label:
                return model.decision_function(emb)
            else:
                return model.predict_proba(emb)
        else:
            return model.predict(emb)
