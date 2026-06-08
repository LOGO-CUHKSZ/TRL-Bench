#!/usr/bin/env python
"""
Mine hard negatives for retrieval training.

This script retrieves top-k tables for each training query and identifies
hard negatives (retrieved but incorrect tables) for contrastive training.

Query embeddings are always BERT. Table embeddings can be from any encoder;
use hybrid (encoder + BERT) table embeddings for non-BERT encoders.

Usage from TRL project root:
    python -m downstream_tasks.table_retrieval.mine_hard_negatives \
        --table_embeddings embeddings/column/tapas_bert_hybrid/nq_tables.pkl \
        --table_id_mapping datasets/nq_tables/csv/table_id_to_csv.json \
        --query_embeddings embeddings/table_retrieval/bert/queries_train.pkl \
        --questions datasets/nq_tables/json/train.json \
        --output_path embeddings/table_retrieval/hard_negatives/train_tapas_hybrid.json \
        --top_k 100 \
        --num_hard_negatives 5
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Get TRL project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (release tree uses trl_bench.tasks.table_retrieval.*)
from trl_bench.tasks.table_retrieval.utils.faiss_utils import build_index, search_index
from trl_bench.tasks.table_retrieval.utils.data_utils import (
    load_table_embeddings,
    load_query_embeddings as _load_query_embeddings,
    load_training_data,
    build_csv_to_table_id_mapping,
)
from trl_bench.tasks.table_retrieval.models.projection_head import DualProjectionHead


def project_embeddings(
    embeddings: np.ndarray,
    projection_head: DualProjectionHead,
    mode: str,  # 'table' or 'query'
    device: str = 'cuda',
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Project embeddings through trained projection head.

    Args:
        embeddings: np.ndarray of shape (N, D)
        projection_head: Trained DualProjectionHead model
        mode: 'table' or 'query'
        device: Device to run on
        batch_size: Batch size for projection

    Returns:
        Projected embeddings as np.ndarray
    """
    projection_head.eval()
    projected = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(
                embeddings[i:i+batch_size],
                dtype=torch.float32,
                device=device
            )
            if mode == 'table':
                proj = projection_head.forward_table(batch)
            else:
                proj = projection_head.forward_query(batch)
            projected.append(proj.cpu().numpy())

    return np.concatenate(projected, axis=0)


def mine_hard_negatives(
    table_embeddings: np.ndarray,
    id2table: dict,
    table2id: dict,
    query_embeddings_dict: dict,
    questions: list,
    top_k: int = 100,
    num_hard_negatives: int = 5,
    projection_head: DualProjectionHead = None,
    device: str = 'cuda',
) -> dict:
    """
    Mine hard negatives by retrieving top-k tables for each query.

    Args:
        table_embeddings: Table embeddings array
        id2table: Mapping from index to table_id
        table2id: Mapping from table_id to index
        query_embeddings_dict: Dict mapping question_id -> embedding
        questions: List of question dicts with table_id ground truth
        top_k: Number of tables to retrieve
        num_hard_negatives: Number of hard negatives to keep
        projection_head: Optional trained projection head
        device: Device for projection

    Returns:
        Dict mapping question_id -> list of hard negative table_ids
    """
    # Optionally project embeddings
    if projection_head is not None:
        print("Projecting table embeddings through trained head...")
        table_emb_proj = project_embeddings(
            table_embeddings, projection_head, 'table', device
        )

        # Project query embeddings
        print("Projecting query embeddings...")
        query_ids = [q['question_id'] for q in questions]
        query_emb_array = np.stack([query_embeddings_dict[qid] for qid in query_ids])
        query_emb_proj = project_embeddings(
            query_emb_array, projection_head, 'query', device
        )
        query_embeddings_proj = {qid: query_emb_proj[i] for i, qid in enumerate(query_ids)}
    else:
        table_emb_proj = table_embeddings
        query_embeddings_proj = query_embeddings_dict

    # Build FAISS index
    print("Building FAISS index...")
    index = build_index(table_emb_proj.astype(np.float32), use_gpu=True)

    # Retrieve for each query
    print("Retrieving hard negatives...")
    hard_negatives = {}
    stats = {'total': 0, 'with_gold_in_topk': 0, 'avg_hard_neg': 0}

    for q in tqdm(questions, desc="Mining hard negatives"):
        q_id = q['question_id']
        gold_table = q['table_id']

        # Get query embedding
        query_emb = query_embeddings_proj[q_id]
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        # Retrieve top-k
        _, indices = search_index(index, query_emb, k=top_k)
        retrieved_ids = [id2table[idx] for idx in indices[0]]

        # Filter out gold table
        hard_neg_ids = [tid for tid in retrieved_ids if tid != gold_table][:num_hard_negatives]
        hard_negatives[q_id] = hard_neg_ids

        # Stats
        stats['total'] += 1
        if gold_table in retrieved_ids:
            stats['with_gold_in_topk'] += 1
        stats['avg_hard_neg'] += len(hard_neg_ids)

    # Print stats
    stats['avg_hard_neg'] /= max(1, stats['total'])
    print(f"\nMining Statistics:")
    print(f"  Total queries: {stats['total']}")
    print(f"  Gold in top-{top_k}: {stats['with_gold_in_topk']} ({100*stats['with_gold_in_topk']/stats['total']:.1f}%)")
    print(f"  Avg hard negatives per query: {stats['avg_hard_neg']:.1f}")

    return hard_negatives


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for retrieval training")

    # Default paths using TRL conventions
    default_output = str(PROJECT_ROOT / "embeddings/table_retrieval/hard_negatives/train.json")

    # Input paths
    parser.add_argument("--table_embeddings", type=str, required=True,
                        help="Path to table embeddings (.pkl)")
    parser.add_argument("--table_id_mapping", type=str, default=None,
                        help="Path to table_id_to_csv.json for remapping CSV-based table_ids")
    parser.add_argument("--query_embeddings", type=str, required=True,
                        help="Path to query embeddings (.pkl)")
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to questions file (.json)")

    # Optional: use trained projection head
    parser.add_argument("--projection_head", type=str, default=None,
                        help="Path to trained projection head (.pt)")

    # Output
    parser.add_argument("--output_path", type=str, default=default_output,
                        help="Output path for hard negatives (.json)")

    # Mining parameters
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of tables to retrieve")
    parser.add_argument("--num_hard_negatives", type=int, default=5,
                        help="Number of hard negatives to keep per query")

    parser.add_argument("--embedding_type", type=str, default="column_mean",
                        choices=["column_mean", "cls_embedding", "table_embedding", "token_mean"],
                        help="Embedding type to extract from pickle (default: column_mean)")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load table embeddings
    print("Loading table embeddings...")
    tid_mapping = build_csv_to_table_id_mapping(args.table_id_mapping) if args.table_id_mapping else None
    table_emb_dict = load_table_embeddings(args.table_embeddings, embedding_type=args.embedding_type, table_id_mapping=tid_mapping)
    table_ids = sorted(table_emb_dict.keys())
    table_embeddings = np.stack([table_emb_dict[tid] for tid in table_ids])
    id2table = {i: tid for i, tid in enumerate(table_ids)}
    table2id = {tid: i for i, tid in enumerate(table_ids)}
    print(f"Loaded {len(table_embeddings)} table embeddings")

    # Load query embeddings
    print("Loading query embeddings...")
    query_embeddings_dict = _load_query_embeddings(args.query_embeddings)
    questions = load_training_data(args.questions)
    print(f"Loaded {len(query_embeddings_dict)} query embeddings")

    # Load projection head if provided
    projection_head = None
    if args.projection_head:
        print(f"Loading projection head from {args.projection_head}...")
        projection_head = DualProjectionHead.load(args.projection_head, device)

    # Mine hard negatives
    hard_negatives = mine_hard_negatives(
        table_embeddings=table_embeddings,
        id2table=id2table,
        table2id=table2id,
        query_embeddings_dict=query_embeddings_dict,
        questions=questions,
        top_k=args.top_k,
        num_hard_negatives=args.num_hard_negatives,
        projection_head=projection_head,
        device=device,
    )

    # Save results
    print(f"\nSaving hard negatives to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(hard_negatives, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
