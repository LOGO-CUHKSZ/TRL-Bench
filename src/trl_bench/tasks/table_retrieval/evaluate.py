#!/usr/bin/env python
"""
Evaluate table retrieval performance.

Query embeddings are always BERT. Table embeddings can be from any encoder.
Supports both hybrid (encoder + BERT) and model-only (raw encoder) table embeddings.

Usage (Hybrid — recommended for non-BERT):
    python -m downstream_tasks.table_retrieval.evaluate \
        --table_embeddings embeddings/column/tapas_bert_hybrid/nq_tables.pkl \
        --table_id_mapping datasets/nq_tables/csv/table_id_to_csv.json \
        --query_embeddings embeddings/table_retrieval/bert/queries_dev.pkl \
        --questions_path datasets/nq_tables/json/dev.json \
        --projection_head assets/checkpoints/table_retrieval/tapas_bert_hybrid/best_model.pt \
        --output_path results/evaluation/table_retrieval/tapas_bert_hybrid_dev.json \
        --gpu

Usage (Model-only — baseline):
    python -m downstream_tasks.table_retrieval.evaluate \
        --table_embeddings embeddings/column/tapas/nq_tables.pkl \
        --embedding_type column_mean \
        --table_id_mapping datasets/nq_tables/csv/table_id_to_csv.json \
        --query_embeddings embeddings/table_retrieval/bert/queries_dev.pkl \
        --questions_path datasets/nq_tables/json/dev.json \
        --projection_head assets/checkpoints/table_retrieval/tapas_model_only/best_model.pt \
        --output_path results/evaluation/table_retrieval/tapas_model_only_dev.json \
        --gpu
"""

import argparse
import json
import numpy as np
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Get TRL project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (release tree uses trl_bench.tasks.table_retrieval.*; the
# legacy downstream_tasks.* import path was a source-repo artifact that did
# not get rewritten when the release was carved out. Smoke campaign 2026-05-20
# hit this with F07 mpnet x table_retrieval x nq_tables.)
from trl_bench.tasks.table_retrieval.utils.data_utils import (
    load_questions,
    load_table_embeddings,
    load_query_embeddings,
    build_csv_to_table_id_mapping,
)
from trl_bench.tasks.table_retrieval.utils.faiss_utils import build_index, search_index
from trl_bench.tasks.table_retrieval.utils.metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k, print_metrics
from trl_bench.tasks.table_retrieval.models.projection_head import DualProjectionHead


def evaluate_retrieval(
    table_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    questions: list,
    id2table: dict,
    k_values: list = [1, 5, 10, 20, 50, 100],
    use_gpu: bool = False,
) -> dict:
    """
    Run retrieval evaluation.

    Args:
        table_embeddings: Table embedding matrix
        query_embeddings: Query embedding matrix
        questions: List of question dicts with table_id ground truth
        id2table: Mapping from index to table_id
        k_values: List of k values for Recall@k
        use_gpu: Whether to use GPU for FAISS

    Returns:
        Dict of evaluation metrics
    """
    # Build FAISS index
    index = build_index(table_embeddings, use_gpu=use_gpu)

    # Search
    max_k = max(k_values)
    distances, predictions = search_index(index, query_embeddings, k=max_k)

    # Get ground truth
    ground_truth = [q['table_id'] for q in questions]

    # Compute metrics
    recall_results = compute_recall_at_k(predictions, ground_truth, id2table, k_values)
    mrr = compute_mrr(predictions, ground_truth, id2table)

    return {**recall_results, "MRR": mrr}


def main():
    parser = argparse.ArgumentParser(description="Evaluate table retrieval")

    # Default paths using TRL conventions
    default_output = str(PROJECT_ROOT / "results/evaluation/table_retrieval/results.json")

    parser.add_argument("--table_embeddings", type=str, required=True,
                        help="Path to table embeddings (.pkl) — hybrid or raw model embeddings")
    parser.add_argument("--query_embeddings", type=str, required=True,
                        help="Path to query embeddings (.pkl) — always BERT")
    parser.add_argument("--questions_path", type=str, required=True,
                        help="Path to questions JSON file")
    parser.add_argument("--output_path", type=str, default=default_output,
                        help="Path to save results (.json)")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of results to retrieve")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for FAISS")
    parser.add_argument("--embedding_type", type=str, default="column_mean",
                        choices=["table_embedding", "cls_embedding", "column_mean", "token_mean"],
                        help="Embedding type to use (for pickle files)")
    parser.add_argument("--projection_head", type=str, default=None,
                        help="Path to trained projection head checkpoint (.pt)")
    parser.add_argument("--table_id_mapping", type=str, default=None,
                        help="Path to table_id_to_csv.json for remapping CSV-based table_ids")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load embeddings from pkl
    print("Loading embeddings...")

    tid_mapping = None
    if args.table_id_mapping:
        tid_mapping = build_csv_to_table_id_mapping(args.table_id_mapping)
    table_emb_dict = load_table_embeddings(args.table_embeddings, args.embedding_type, table_id_mapping=tid_mapping)
    table_ids = sorted(table_emb_dict.keys())
    table_emb = np.stack([table_emb_dict[tid] for tid in table_ids]).astype(np.float32)
    id2table = {i: tid for i, tid in enumerate(table_ids)}

    query_emb_dict = load_query_embeddings(args.query_embeddings)

    print(f"Table embeddings: {table_emb.shape}")
    print(f"Query embeddings: {len(query_emb_dict)}")

    # Load ground truth
    questions, ground_truth = load_questions(args.questions_path)
    print(f"Loaded {len(questions)} questions")
    print(f"id2table mapping size: {len(id2table)}")

    # Build query embedding array aligned with questions
    query_ids = [q['question_id'] for q in questions]
    query_emb = np.stack([query_emb_dict[qid] for qid in query_ids]).astype(np.float32)

    # Apply projection head if provided
    if args.projection_head:
        print(f"Loading projection head from {args.projection_head}...")
        device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        model = DualProjectionHead.load(args.projection_head).to(device)
        model.eval()

        with torch.no_grad():
            table_tensor = torch.tensor(table_emb, dtype=torch.float32, device=device)
            batch_size = 1024
            projected_tables = []
            for i in range(0, len(table_tensor), batch_size):
                batch = table_tensor[i:i+batch_size]
                projected = model.forward_table(batch)
                projected_tables.append(projected.cpu().numpy())
            table_emb = np.concatenate(projected_tables, axis=0)

            query_tensor = torch.tensor(query_emb, dtype=torch.float32, device=device)
            query_emb = model.forward_query(query_tensor).cpu().numpy()

        print(f"Projected embeddings: tables {table_emb.shape}, queries {query_emb.shape}")

    # Build FAISS index
    print("Building FAISS index...")
    index = build_index(table_emb, use_gpu=args.gpu)

    # Search
    print(f"Searching for top-{args.k} tables...")
    distances, predictions = search_index(index, query_emb, k=args.k)

    # Compute metrics
    print("Computing metrics...")
    k_values = [1, 5, 10, 20, 50, 100]
    k_values = [k for k in k_values if k <= args.k]

    recall_results = compute_recall_at_k(predictions, ground_truth, id2table, k_values)
    mrr = compute_mrr(predictions, ground_truth, id2table)

    # NDCG (gated to actual retrieval depth)
    ndcg_results = {}
    for k in [10, 20]:
        if k <= args.k:
            ndcg_results[f"NDCG@{k}"] = compute_ndcg_at_k(predictions, ground_truth, id2table, k=k)

    # Build results in TRL format
    results = {
        "task": "table_retrieval",
        "dataset": "nq_tables",
        "split": "dev" if "dev" in args.questions_path else "test" if "test" in args.questions_path else "train",
        "model_checkpoint": args.projection_head,
        "metrics": {
            **recall_results,
            "MRR": mrr,
            **ndcg_results,
        },
        "num_queries": len(questions),
        "num_tables": len(id2table),
        "timestamp": datetime.now().isoformat(),
    }

    # Print results
    print_metrics(results["metrics"], "Table Retrieval Results")

    # Save results
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
