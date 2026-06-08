"""
Dummy baseline probe for all supervised downstream tasks.

Predicts from training label statistics only — ignores embeddings entirely.
Used as a no-skill floor to contextualize learned probe results.

- Single-label classification: predicts majority class (most_frequent)
- Multi-label classification: outputs logit(label_prevalence) per label
- Regression: predicts training mean

Usage from any task script:
    runner = DummyProbeRunner()
    results = runner.run(train_labels=..., test_labels=...,
                         task_type='classification', metric_names=['accuracy', 'macro_f1'])
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class DummyProbeRunner:
    """Dummy baseline: predicts from training label statistics only.

    Mirrors LinearProbeRunner.run() interface so task scripts can swap
    it in with minimal changes.
    """

    def run(
        self,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        task_type: str,
        metric_names: List[str],
        train_emb: Optional[np.ndarray] = None,
        test_emb: Optional[np.ndarray] = None,
        val_emb: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        multi_label: bool = False,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run dummy baseline: compute label statistics, predict, evaluate.

        Args:
            train_labels: Training labels (N_train,) or (N_train, C) for multi-label.
            test_labels: Test labels.
            task_type: 'classification' or 'regression'.
            metric_names: Metric names from METRIC_REGISTRY.
            train_emb: Ignored (accepted for interface compatibility).
            test_emb: Ignored (accepted for interface compatibility).
            val_emb: Ignored.
            val_labels: Ignored.
            multi_label: Whether this is a multi-label classification task.
            threshold: Threshold for multi-label metrics.
            num_classes: Explicit number of classes (for shared vocabularies
                where test may have classes not seen in training).

        Returns:
            Dict with test_<metric> keys (matching LinearProbeRunner format)
            plus dummy_probe_meta with baseline details.
        """
        n_test = len(test_labels)

        if task_type == 'classification':
            if multi_label:
                outputs = self._multi_label_outputs(train_labels, n_test)
                meta = self._multi_label_meta(train_labels)
            else:
                outputs = self._single_label_outputs(
                    train_labels, n_test, num_classes)
                meta = self._single_label_meta(train_labels, num_classes)
        else:
            outputs = self._regression_outputs(train_labels, n_test)
            meta = self._regression_meta(train_labels)

        metrics = compute_metrics(outputs, test_labels, metric_names,
                                  threshold=threshold)

        results = {f'test_{k}': v for k, v in metrics.items()}
        results['test_loss'] = None
        results['dummy_probe_meta'] = {
            'task_type': task_type,
            'multi_label': multi_label,
            'train_samples': len(train_labels),
            'test_samples': n_test,
            **meta,
        }

        return results

    # ── Single-label classification ──────────────────────────────────

    def _single_label_outputs(self, train_labels, n_test, num_classes=None):
        """Build (N_test, C) probability array from training class frequencies."""
        labels_int = train_labels.astype(int)
        if num_classes is not None:
            n_classes = num_classes
        else:
            n_classes = int(labels_int.max()) + 1

        counts = np.bincount(labels_int, minlength=n_classes).astype(np.float64)
        freq = counts / counts.sum()

        # Broadcast to (N_test, C)
        return np.tile(freq, (n_test, 1)).astype(np.float32)

    def _single_label_meta(self, train_labels, num_classes=None):
        labels_int = train_labels.astype(int)
        majority_class = int(np.bincount(labels_int).argmax())
        majority_fraction = float(
            np.bincount(labels_int).max() / len(labels_int))
        n_classes = num_classes if num_classes is not None else int(labels_int.max()) + 1
        return {
            'strategy': 'most_frequent',
            'majority_class': majority_class,
            'majority_fraction': majority_fraction,
            'n_classes': n_classes,
        }

    # ── Multi-label classification ───────────────────────────────────

    def _multi_label_outputs(self, train_labels, n_test):
        """Build (N_test, C) logit-prevalence array for multi-label."""
        from scipy.special import logit

        # train_labels: (N_train, C) binary
        prevalence = train_labels.mean(axis=0)  # (C,)

        # Clamp to avoid logit(-inf) / logit(+inf)
        eps = 1e-7
        prevalence = np.clip(prevalence, eps, 1.0 - eps)

        scores = logit(prevalence)  # (C,)
        return np.tile(scores, (n_test, 1)).astype(np.float32)

    def _multi_label_meta(self, train_labels):
        prevalence = train_labels.mean(axis=0)
        return {
            'strategy': 'label_prior',
            'n_labels': train_labels.shape[1],
            'mean_prevalence': float(prevalence.mean()),
            'labels_above_threshold': int((prevalence >= 0.5).sum()),
        }

    # ── Regression ───────────────────────────────────────────────────

    def _regression_outputs(self, train_labels, n_test):
        """Fill (N_test,) with training mean."""
        train_mean = float(np.mean(train_labels))
        return np.full(n_test, train_mean, dtype=np.float32)

    def _regression_meta(self, train_labels):
        return {
            'strategy': 'mean',
            'training_mean': float(np.mean(train_labels)),
            'training_std': float(np.std(train_labels)),
        }
