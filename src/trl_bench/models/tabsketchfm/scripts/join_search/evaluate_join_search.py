"""
Evaluate join search results by computing Recall@K, Precision@K, MRR, MAP.

Usage:
    python scripts/join_search/evaluate_join_search.py \
        --results results/wiki_join_search_results.pkl \
        --ground_truth embeddings/wiki_join_search_ground_truth.pkl \
        --k_values 1,5,10,20,50
"""

import pickle
import argparse
import numpy as np
from collections import defaultdict


def recall_at_k(retrieved, relevant, k):
    """
    Compute Recall@K.

    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if len(relevant) == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved, relevant, k):
    """
    Compute Precision@K.

    Precision@K = |retrieved[:k] ∩ relevant| / k
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return len(retrieved_at_k & relevant_set) / k


def average_precision(retrieved, relevant):
    """
    Compute Average Precision (AP).

    AP = (1/|relevant|) * Σ(Precision@k * rel(k))
    where rel(k) = 1 if item at rank k is relevant, 0 otherwise
    """
    if len(relevant) == 0:
        return 0.0

    relevant_set = set(relevant)
    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(retrieved):
        if item in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(relevant_set)


def reciprocal_rank(retrieved, relevant):
    """
    Compute Reciprocal Rank (RR).

    RR = 1 / rank of first relevant item
    """
    relevant_set = set(relevant)

    for i, item in enumerate(retrieved):
        if item in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def evaluate_retrieval(results, ground_truth, k_values=[1, 5, 10, 20, 50]):
    """
    Evaluate retrieval results.

    Args:
        results: Dict mapping queries to retrieved items
        ground_truth: Dict mapping queries to relevant items
        k_values: List of K values for Recall@K and Precision@K

    Returns:
        Dict with evaluation metrics
    """
    metrics = defaultdict(list)

    # Track queries
    total_queries = 0
    missing_queries = 0

    for query, relevant in ground_truth.items():
        total_queries += 1

        if query not in results:
            missing_queries += 1
            # Add zeros for missing queries
            for k in k_values:
                metrics[f'recall@{k}'].append(0.0)
                metrics[f'precision@{k}'].append(0.0)
            metrics['ap'].append(0.0)
            metrics['rr'].append(0.0)
            continue

        retrieved = results[query]

        # Recall and Precision at K
        for k in k_values:
            metrics[f'recall@{k}'].append(recall_at_k(retrieved, relevant, k))
            metrics[f'precision@{k}'].append(precision_at_k(retrieved, relevant, k))

        # Average Precision
        metrics['ap'].append(average_precision(retrieved, relevant))

        # Reciprocal Rank
        metrics['rr'].append(reciprocal_rank(retrieved, relevant))

    # Compute averages
    avg_metrics = {}
    for metric_name, values in metrics.items():
        avg_metrics[metric_name] = np.mean(values)

    # Add MAP and MRR
    avg_metrics['map'] = avg_metrics['ap']
    avg_metrics['mrr'] = avg_metrics['rr']

    # Add summary stats
    avg_metrics['_meta'] = {
        'total_queries': total_queries,
        'missing_queries': missing_queries,
        'coverage': (total_queries - missing_queries) / total_queries if total_queries > 0 else 0.0
    }

    return avg_metrics


def print_results(metrics, k_values):
    """Print evaluation results in a nice format."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    # Meta info
    meta = metrics.get('_meta', {})
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total queries:    {meta.get('total_queries', 'N/A')}")
    print(f"   Missing queries:  {meta.get('missing_queries', 'N/A')}")
    print(f"   Coverage:         {meta.get('coverage', 0.0)*100:.2f}%")

    # Recall@K
    print(f"\n📈 Recall@K:")
    for k in k_values:
        recall = metrics.get(f'recall@{k}', 0.0)
        print(f"   Recall@{k:<3} = {recall:.4f} ({recall*100:.2f}%)")

    # Precision@K
    print(f"\n🎯 Precision@K:")
    for k in k_values:
        precision = metrics.get(f'precision@{k}', 0.0)
        print(f"   Precision@{k:<3} = {precision:.4f} ({precision*100:.2f}%)")

    # MRR and MAP
    print(f"\n🏆 Ranking Metrics:")
    print(f"   MRR (Mean Reciprocal Rank) = {metrics.get('mrr', 0.0):.4f}")
    print(f"   MAP (Mean Average Precision) = {metrics.get('map', 0.0):.4f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate join search results")
    parser.add_argument('--results', type=str, required=True,
                        help='Pickle file with search results')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Pickle file with ground truth')
    parser.add_argument('--k_values', type=str, default='1,5,10,20,50',
                        help='Comma-separated K values (default: 1,5,10,20,50)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output file to save metrics (JSON)')

    args = parser.parse_args()

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]

    print(f"📂 Loading results from: {args.results}")
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    print(f"   Loaded {len(results)} query results")

    print(f"\n📂 Loading ground truth from: {args.ground_truth}")
    with open(args.ground_truth, 'rb') as f:
        ground_truth = pickle.load(f)
    print(f"   Loaded {len(ground_truth)} ground truth queries")

    print(f"\n🔍 Evaluating with K values: {k_values}")

    metrics = evaluate_retrieval(results, ground_truth, k_values)

    print_results(metrics, k_values)

    # Save to file if requested
    if args.output:
        import json
        # Remove numpy types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in metrics.items() if k != '_meta'}
        metrics_json['_meta'] = metrics['_meta']

        with open(args.output, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"\n💾 Metrics saved to: {args.output}")


if __name__ == '__main__':
    main()
