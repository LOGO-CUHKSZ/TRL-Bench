"""Evaluation metrics for table retrieval."""

from typing import List, Dict, Union
import numpy as np


def compute_recall_at_k(
    predictions: np.ndarray,
    ground_truth: List[str],
    id2table: Dict[int, str],
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """
    Compute Recall@k for different k values.

    Args:
        predictions: np.ndarray of shape (Q, K) - indices of retrieved tables
        ground_truth: List of correct table IDs for each query
        id2table: Dict mapping index to table_id
        k_values: List of k values to compute

    Returns:
        Dict mapping metric name to value
    """
    results = {}
    num_queries = len(ground_truth)

    for k in k_values:
        if k > predictions.shape[1]:
            k = predictions.shape[1]

        hits = 0
        for i, gt_table_id in enumerate(ground_truth):
            # Get predicted table IDs for top-k
            pred_indices = predictions[i, :k]
            pred_table_ids = [id2table.get(int(idx), "") for idx in pred_indices]

            if gt_table_id in pred_table_ids:
                hits += 1

        recall = hits / num_queries
        results[f"Recall@{k}"] = recall

    return results


def compute_mrr(
    predictions: np.ndarray,
    ground_truth: List[str],
    id2table: Dict[int, str]
) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        predictions: np.ndarray of shape (Q, K) - indices of retrieved tables
        ground_truth: List of correct table IDs for each query
        id2table: Dict mapping index to table_id

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for i, gt_table_id in enumerate(ground_truth):
        # Get all predicted table IDs
        pred_table_ids = [id2table.get(int(idx), "") for idx in predictions[i]]

        if gt_table_id in pred_table_ids:
            rank = pred_table_ids.index(gt_table_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def compute_ndcg_at_k(
    predictions: np.ndarray,
    ground_truth: List[str],
    id2table: Dict[int, str],
    k: int = 10
) -> float:
    """
    Compute NDCG@k (assuming binary relevance).

    Args:
        predictions: np.ndarray of shape (Q, K) - indices of retrieved tables
        ground_truth: List of correct table IDs for each query
        id2table: Dict mapping index to table_id
        k: Cutoff for NDCG

    Returns:
        NDCG@k score
    """
    ndcg_scores = []

    for i, gt_table_id in enumerate(ground_truth):
        pred_table_ids = [id2table.get(int(idx), "") for idx in predictions[i, :k]]

        # Binary relevance: 1 if correct, 0 otherwise
        relevances = [1.0 if tid == gt_table_id else 0.0 for tid in pred_table_ids]

        # DCG
        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))

        # Ideal DCG (best case: correct table at rank 1)
        idcg = 1.0 / np.log2(2)  # = 1.0

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print metrics."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    for name, value in sorted(metrics.items()):
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")
    print("=" * 50 + "\n")
