"""
Unified metric computation for downstream tasks.

All metrics operate on numpy arrays (outputs and targets) and return
a dict of metric_name -> float. The Trainer prefixes these with
train_/val_/test_ automatically.
"""

import numpy as np
from typing import Dict, List, Optional


def _ensure_numpy(x):
    """Convert to numpy if needed."""
    if hasattr(x, 'numpy'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accuracy(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    """Classification accuracy. Handles both logit and index inputs."""
    if outputs.ndim == 2:
        preds = outputs.argmax(axis=1)
    else:
        preds = outputs
    return float(np.mean(preds == targets))


def _f1_score(outputs: np.ndarray, targets: np.ndarray,
              average: str, threshold: float = 0.5,
              zero_division: int = 0) -> float:
    """Shared F1 implementation. Handles single-label (argmax) and
    multi-label (sigmoid + threshold) prediction conversion."""
    from sklearn.metrics import f1_score
    if outputs.ndim == 2 and targets.ndim == 2:
        from scipy.special import expit
        preds = (expit(outputs) >= threshold).astype(int)
    elif outputs.ndim == 2:
        preds = outputs.argmax(axis=1)
    else:
        preds = outputs
    return float(f1_score(targets, preds, average=average,
                          zero_division=zero_division))


def f1(outputs: np.ndarray, targets: np.ndarray,
       threshold: float = 0.5, **kwargs) -> float:
    """Positive-class (binary) F1 — the standard metric in entity matching literature."""
    return _f1_score(outputs, targets, 'binary', threshold, zero_division=0)


def macro_f1(outputs: np.ndarray, targets: np.ndarray,
             threshold: float = 0.5, **kwargs) -> float:
    return _f1_score(outputs, targets, 'macro', threshold, zero_division=0)


def weighted_f1(outputs: np.ndarray, targets: np.ndarray,
                threshold: float = 0.5, **kwargs) -> float:
    return _f1_score(outputs, targets, 'weighted', threshold, zero_division=1)


def micro_f1(outputs: np.ndarray, targets: np.ndarray,
             threshold: float = 0.5, **kwargs) -> float:
    return _f1_score(outputs, targets, 'micro', threshold, zero_division=0)


def subset_accuracy(outputs: np.ndarray, targets: np.ndarray,
                    threshold: float = 0.5, **kwargs) -> float:
    """Exact-match accuracy for multi-label classification."""
    from scipy.special import expit
    preds = (expit(outputs) >= threshold).astype(float)
    return float(np.mean(np.all(preds == targets, axis=1)))


def hamming_accuracy(outputs: np.ndarray, targets: np.ndarray,
                     threshold: float = 0.5, **kwargs) -> float:
    """Per-label accuracy for multi-label classification."""
    from scipy.special import expit
    preds = (expit(outputs) >= threshold).astype(float)
    return float(np.mean(preds == targets))


def precision(outputs: np.ndarray, targets: np.ndarray,
              threshold: float = 0.5, **kwargs) -> float:
    """Binary classification precision."""
    from sklearn.metrics import precision_score
    if outputs.ndim == 2:
        preds = outputs.argmax(axis=1)
    else:
        preds = outputs
    return float(precision_score(targets, preds, average='binary', zero_division=0))


def recall(outputs: np.ndarray, targets: np.ndarray,
           threshold: float = 0.5, **kwargs) -> float:
    """Binary classification recall."""
    from sklearn.metrics import recall_score
    if outputs.ndim == 2:
        preds = outputs.argmax(axis=1)
    else:
        preds = outputs
    return float(recall_score(targets, preds, average='binary', zero_division=0))


def mse(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    from sklearn.metrics import mean_squared_error
    return float(mean_squared_error(targets, outputs.squeeze()))


def rmse(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(targets, outputs.squeeze())))


def mae(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    from sklearn.metrics import mean_absolute_error
    return float(mean_absolute_error(targets, outputs.squeeze()))


def r2(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(targets, outputs.squeeze()))


def _to_probs(outputs: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities if needed (softmax for 2-d arrays)."""
    if outputs.ndim != 2:
        return outputs
    # If rows already sum to ~1, assume probabilities (from predict_proba)
    row_sums = outputs.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=0.01):
        return outputs
    # Apply softmax
    exp = np.exp(outputs - outputs.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def auroc(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    """Area Under the ROC Curve. Handles logits, probabilities, binary and multi-class."""
    from sklearn.metrics import roc_auc_score
    unique_classes = np.unique(targets)
    if len(unique_classes) < 2:
        return 0.5
    if outputs.ndim != 2:
        return 0.5  # cannot compute AUROC from hard predictions
    probs = _to_probs(outputs)
    n_classes = probs.shape[1]
    # Guard: test split may be missing classes present during training
    if len(unique_classes) < n_classes:
        if n_classes == 2:
            # Binary case: still works with probs[:, 1]
            return float(roc_auc_score(targets, probs[:, 1]))
        return 0.5
    if n_classes == 2:
        return float(roc_auc_score(targets, probs[:, 1]))
    return float(roc_auc_score(targets, probs, multi_class='ovr', average='weighted'))


def pearson_r(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    """Pearson correlation coefficient for regression."""
    from scipy.stats import pearsonr
    preds = np.atleast_1d(outputs.squeeze())
    if len(preds) < 2:
        return 0.0
    r, _ = pearsonr(targets, preds)
    return 0.0 if np.isnan(r) else float(r)


def spearman_r(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> float:
    """Spearman rank correlation coefficient for regression."""
    from scipy.stats import spearmanr
    preds = np.atleast_1d(outputs.squeeze())
    if len(preds) < 2:
        return 0.0
    r, _ = spearmanr(targets, preds)
    return 0.0 if np.isnan(r) else float(r)


# Registry: metric name -> function
METRIC_REGISTRY = {
    'accuracy': accuracy,
    'f1': f1,
    'macro_f1': macro_f1,
    'weighted_f1': weighted_f1,
    'micro_f1': micro_f1,
    'subset_accuracy': subset_accuracy,
    'hamming_accuracy': hamming_accuracy,
    'precision': precision,
    'recall': recall,
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'auroc': auroc,
    'pearson_r': pearson_r,
    'spearman_r': spearman_r,
}

# For each metric, whether higher is better
METRIC_DIRECTIONS = {
    'accuracy': 'max',
    'f1': 'max',
    'macro_f1': 'max',
    'weighted_f1': 'max',
    'micro_f1': 'max',
    'subset_accuracy': 'max',
    'hamming_accuracy': 'max',
    'precision': 'max',
    'recall': 'max',
    'mse': 'min',
    'rmse': 'min',
    'mae': 'min',
    'r2': 'max',
    'auroc': 'max',
    'pearson_r': 'max',
    'spearman_r': 'max',
}


def compute_metrics(
    outputs: np.ndarray,
    targets: np.ndarray,
    metric_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute multiple metrics at once.

    Args:
        outputs: Raw model outputs (logits). Shape depends on task.
        targets: Ground truth labels.
        metric_names: List of metric names from METRIC_REGISTRY.
        threshold: Threshold for multi-label metrics.

    Returns:
        Dict mapping metric_name -> value.
    """
    outputs = _ensure_numpy(outputs)
    targets = _ensure_numpy(targets)

    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
            )
        results[name] = METRIC_REGISTRY[name](
            outputs, targets, threshold=threshold
        )
    return results
