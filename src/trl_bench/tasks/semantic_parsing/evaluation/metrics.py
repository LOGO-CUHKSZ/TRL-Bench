"""Evaluation metrics for semantic parsing."""

from typing import List, Dict, Any


def compute_accuracy(predictions: List[Dict], gold_examples: List[Dict]) -> float:
    """Compute top-1 accuracy.

    Args:
        predictions: List of prediction dicts with 'answer' or 'correct' key
        gold_examples: List of gold examples (may be used for comparison)

    Returns:
        Accuracy as a float between 0 and 1
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred in predictions:
        # Check if prediction is correct
        if pred.get('correct', False):
            correct += 1
        elif 'answer' in pred and 'gold_answer' in pred:
            if pred['answer'] == pred['gold_answer']:
                correct += 1

    return correct / len(predictions)


def compute_oracle_accuracy(predictions: List[Dict], gold_examples: List[Dict]) -> float:
    """Compute oracle accuracy (correct program exists in beam).

    Args:
        predictions: List of prediction dicts with 'beam' containing candidates
        gold_examples: List of gold examples

    Returns:
        Oracle accuracy as a float between 0 and 1
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred in predictions:
        # Check if any program in beam is correct
        beam = pred.get('beam', [pred])
        if any(p.get('correct', False) for p in beam):
            correct += 1

    return correct / len(predictions)
