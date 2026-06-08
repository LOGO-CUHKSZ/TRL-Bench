"""Evaluation utilities for semantic parsing."""

from .metrics import compute_accuracy, compute_oracle_accuracy


def evaluate(predictions, gold_examples) -> dict:
    """Evaluate predictions against gold examples.

    Args:
        predictions: List of prediction dicts with 'answer' key
        gold_examples: List of gold examples with 'answer' key

    Returns:
        Dictionary with accuracy and oracle_accuracy
    """
    return {
        'accuracy': compute_accuracy(predictions, gold_examples),
        'oracle_accuracy': compute_oracle_accuracy(predictions, gold_examples),
    }


__all__ = [
    'evaluate',
    'compute_accuracy',
    'compute_oracle_accuracy',
]
