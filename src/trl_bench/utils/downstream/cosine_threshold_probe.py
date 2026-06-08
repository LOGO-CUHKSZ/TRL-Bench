"""
Cosine similarity threshold baseline for binary pair classification.

Bypasses embedding combination and training entirely: computes cosine
similarity between raw embedding pairs, sweeps a threshold on validation
(or train) to maximize a target metric, and applies to test.

Usage from a task script:
    runner = CosineThresholdRunner(optimize_metric='f1')
    results = runner.run(
        train_scores=train_cos, train_labels=train_y,
        test_scores=test_cos, test_labels=test_y,
        metric_names=['f1', 'accuracy', 'precision', 'recall'],
        val_scores=val_cos, val_labels=val_y,
    )
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import compute_metrics, METRIC_DIRECTIONS

logger = logging.getLogger(__name__)


class CosineThresholdRunner:
    """Cosine similarity threshold baseline for binary pair tasks.

    Mirrors DummyProbeRunner/LinearProbeRunner interface conventions so
    task scripts can swap it in with minimal changes.
    """

    def __init__(self, optimize_metric: str = 'f1'):
        self.optimize_metric = optimize_metric

    def run(
        self,
        train_scores: np.ndarray,
        train_labels: np.ndarray,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
        metric_names: List[str],
        val_scores: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run cosine threshold baseline: sweep thresholds, evaluate on test.

        Args:
            train_scores: (N_train,) cosine similarities
            train_labels: (N_train,) binary 0/1 labels
            test_scores: (N_test,) cosine similarities
            test_labels: (N_test,) binary 0/1 labels
            metric_names: Metrics to compute on test (e.g. ['f1', 'accuracy'])
            val_scores: Optional (N_val,) cosine similarities for threshold tuning
            val_labels: Optional (N_val,) binary labels for threshold tuning

        Returns:
            Dict with test_<metric> keys, test_loss=None, and
            cosine_threshold_meta with threshold selection details.
        """
        # Guard: require non-empty train and test
        if len(train_scores) == 0:
            raise ValueError("CosineThresholdRunner: train_scores is empty")
        if len(test_scores) == 0:
            raise ValueError("CosineThresholdRunner: test_scores is empty")

        # Determine tuning set: prefer validation, fall back to train
        if val_scores is not None and val_labels is not None and len(val_scores) > 0:
            tune_scores = val_scores
            tune_labels = val_labels
            tuning_set = 'validation'
        else:
            tune_scores = train_scores
            tune_labels = train_labels
            tuning_set = 'train'

        # Warn and filter non-finite scores (NaN/Inf from degenerate embeddings)
        tune_finite = np.isfinite(tune_scores)
        if not tune_finite.all():
            n_bad = int((~tune_finite).sum())
            logger.warning(
                "%d non-finite cosine scores in %s tuning set — filtering",
                n_bad, tuning_set,
            )
            tune_scores = tune_scores[tune_finite]
            tune_labels = tune_labels[tune_finite]
            if len(tune_scores) == 0:
                raise ValueError(
                    "CosineThresholdRunner: all tuning scores are non-finite"
                )

        # Build threshold candidates from unique score midpoints + sentinels
        unique_scores = np.unique(tune_scores)
        if len(unique_scores) <= 1:
            # Degenerate case: all scores identical
            candidates = np.array([unique_scores[0] - 0.01, unique_scores[0] + 0.01])
        else:
            midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
            # Sentinels: below min (predict all 1) and above max (predict all 0)
            sentinel_low = unique_scores[0] - 0.01
            sentinel_high = unique_scores[-1] + 0.01
            candidates = np.concatenate([[sentinel_low], midpoints, [sentinel_high]])

        # Determine optimization direction
        higher_is_better = METRIC_DIRECTIONS.get(self.optimize_metric, 'max') == 'max'

        # Sweep thresholds
        best_threshold = candidates[0]
        best_score = -np.inf if higher_is_better else np.inf

        for t in candidates:
            preds = (tune_scores >= t).astype(np.int64)
            result = compute_metrics(preds, tune_labels, [self.optimize_metric])
            score = result[self.optimize_metric]

            if higher_is_better and score > best_score:
                best_score = score
                best_threshold = t
            elif not higher_is_better and score < best_score:
                best_score = score
                best_threshold = t

        logger.info(
            "Threshold sweep: %d candidates, best=%.4f (%s=%.4f on %s)",
            len(candidates), best_threshold, self.optimize_metric,
            best_score, tuning_set,
        )

        # Warn about non-finite test scores (don't filter — preserve alignment
        # with test_labels — but report them in metadata)
        test_finite = np.isfinite(test_scores)
        n_test_nonfinite = int((~test_finite).sum())
        if n_test_nonfinite > 0:
            logger.warning(
                "%d non-finite cosine scores in test set", n_test_nonfinite,
            )

        # Apply best threshold to test
        test_preds = (test_scores >= best_threshold).astype(np.int64)
        test_metrics = compute_metrics(test_preds, test_labels, metric_names)

        # Build return dict (test_ prefix convention)
        output = {f'test_{k}': v for k, v in test_metrics.items()}
        output['test_loss'] = None

        # Safe score_range: use only finite values to avoid JSON NaN
        finite_test = test_scores[test_finite]
        if len(finite_test) > 0:
            score_range = [float(finite_test.min()), float(finite_test.max())]
        else:
            score_range = [None, None]

        output['cosine_threshold_meta'] = {
            'best_threshold': float(best_threshold),
            'optimize_metric': self.optimize_metric,
            'tuning_best_score': float(best_score),
            'tuning_set': tuning_set,
            'n_candidates': len(candidates),
            'train_samples': len(train_scores),
            'val_samples': len(val_scores) if val_scores is not None else 0,
            'test_samples': len(test_scores),
            'test_nonfinite': n_test_nonfinite,
            'score_range': score_range,
        }

        return output
