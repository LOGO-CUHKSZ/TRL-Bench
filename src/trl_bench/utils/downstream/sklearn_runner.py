"""
Sklearn-based fit/predict runner for downstream tasks.

Replaces the per-model train-and-evaluate loops formerly in downstream_utils.py
with a unified runner. Returns results but does NOT write any files —
file I/O stays in the calling script (train_downstream.py).
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from .heads import get_sklearn_head


def _train_and_evaluate_classifier(
    clf,
    clf_name: str,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate a single classifier."""
    from sklearn.metrics import accuracy_score

    if verbose:
        print(f"\n{clf_name}")
        print("-" * 80)
        print(f"   Training...", end=' ')

    start_time = time.time()
    clf.fit(train_embeddings, train_labels)
    train_time = time.time() - start_time
    if verbose:
        print(f"done ({train_time:.2f}s)")

    train_preds = clf.predict(train_embeddings)
    train_accuracy = accuracy_score(train_labels, train_preds)

    if verbose:
        print(f"   Evaluating on test set...", end=' ')
    start_time = time.time()
    test_preds = clf.predict(test_embeddings)
    inference_time = time.time() - start_time
    test_accuracy = accuracy_score(test_labels, test_preds)

    if verbose:
        print(f"done ({inference_time:.2f}s)")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'inference_time': inference_time,
        'predictions': test_preds,
        'model': clf,
    }


def _train_and_evaluate_regressor(
    reg,
    reg_name: str,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate a single regressor."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if verbose:
        print(f"\n{reg_name}")
        print("-" * 80)
        print(f"   Training...", end=' ')

    start_time = time.time()
    reg.fit(train_embeddings, train_labels)
    train_time = time.time() - start_time
    if verbose:
        print(f"done ({train_time:.2f}s)")

    train_preds = reg.predict(train_embeddings)
    train_mse = mean_squared_error(train_labels, train_preds)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_labels, train_preds)
    train_r2 = r2_score(train_labels, train_preds)

    if verbose:
        print(f"   Evaluating on test set...", end=' ')
    start_time = time.time()
    test_preds = reg.predict(test_embeddings)
    inference_time = time.time() - start_time
    test_mse = mean_squared_error(test_labels, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_labels, test_preds)
    test_r2 = r2_score(test_labels, test_preds)

    if verbose:
        print(f"done ({inference_time:.2f}s)")
        print(f"   Test MSE:  {test_mse:.4f}")
        print(f"   Test RMSE: {test_rmse:.4f}")
        print(f"   Test R2:   {test_r2:.4f}")

    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_time': train_time,
        'inference_time': inference_time,
        'predictions': test_preds,
        'model': reg,
    }


class SklearnRunner:
    """Unified sklearn fit/predict runner.

    Returns results dict per model. Does NOT write any files.
    The return dict uses flat keys matching the format consumed by
    train_downstream.py.
    """

    def __init__(self, config: Optional[DictConfig] = None, use_gpu: bool = True):
        """
        Args:
            config: Optional DictConfig. Currently used for seed only.
            use_gpu: Whether to use GPU for models that support it.
        """
        self.use_gpu = use_gpu
        if config is not None and hasattr(config, 'training'):
            self.seed = config.training.get('seed', 42)
        else:
            self.seed = 42

    def fit_and_evaluate(
        self,
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray,
        test_labels: np.ndarray,
        task_type: str = 'classification',
        model_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Run fit + evaluate for each requested model.

        Args:
            train_emb: Training embeddings (N_train x D).
            train_labels: Training labels (N_train,).
            test_emb: Test embeddings (N_test x D).
            test_labels: Test labels (N_test,).
            task_type: 'classification' or 'regression'.
            model_names: List of model names (from get_sklearn_head).
                If None, uses all default models for the task type.
            verbose: Print progress.

        Returns:
            Dict mapping model_name -> results dict with flat keys.
        """
        if model_names is None:
            if task_type == 'classification':
                model_names = ['logistic', 'random_forest', 'svm', 'mlp', 'xgboost']
            else:
                model_names = ['linear', 'ridge', 'lasso', 'random_forest',
                               'svm', 'mlp', 'xgboost']

        results = {}
        for name in model_names:
            model = get_sklearn_head(
                name, task=task_type, use_gpu=self.use_gpu,
                random_state=self.seed,
            )

            if task_type == 'classification':
                results[name] = _train_and_evaluate_classifier(
                    model, name, train_emb, train_labels,
                    test_emb, test_labels, verbose=verbose
                )
            else:
                results[name] = _train_and_evaluate_regressor(
                    model, name, train_emb, train_labels,
                    test_emb, test_labels, verbose=verbose
                )

        return results
