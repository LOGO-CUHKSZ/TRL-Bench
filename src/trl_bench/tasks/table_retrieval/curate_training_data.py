"""
Curate training data with hard negatives from full corpus retrieval.

This follows the NQT-Retrieval approach:
1. Run zero-shot retrieval on FULL corpus for each training question
2. Categorize retrieved tables into:
   - positive_ctxs: tables containing the answer
   - hard_negative_ctxs: top-k tables that don't contain answer (confusing negatives)
   - negative_ctxs: lower-ranked tables without answer

Query embeddings are always BERT. Table embeddings can be from any encoder;
use hybrid (encoder + BERT) table embeddings for non-BERT encoders.

Usage:
    python -m downstream_tasks.table_retrieval.curate_training_data \
        --table_embeddings embeddings/column/tapas_bert_hybrid/nq_tables.pkl \
        --table_id_mapping datasets/nq_tables/csv/table_id_to_csv.json \
        --query_embeddings embeddings/table_retrieval/bert/queries_train.pkl \
        --questions datasets/nq_tables/json/train.json \
        --output_path datasets/nq_tables/json/train_curated.json \
        --top_k 100 \
        --num_hard_negatives 5
"""

import argparse
import json
import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Get TRL project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trl_bench.tasks.table_retrieval.utils.data_utils import (
    load_table_embeddings,
    load_query_embeddings as _load_query_embeddings,
    load_training_data,
    build_csv_to_table_id_mapping,
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> "faiss.Index":
    """Build FAISS index for fast retrieval."""
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is required for this script. Install with: pip install faiss-cpu")

    dim = embeddings.shape[1]

    # Use inner product (dot product) index
    index = faiss.IndexFlatIP(dim)

    # Normalize embeddings for cosine similarity via dot product
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings_normalized.astype(np.float32))

    return index, embeddings_normalized


def check_answer_in_table(table: Dict, answers: List[str]) -> bool:
    """Check if any answer appears in the table content."""
    if not answers:
        return False

    # Concatenate all table content
    table_text = table.get('title', '').lower()

    for header in table.get('header', []):
        table_text += ' ' + str(header).lower()

    for row in table.get('rows', []):
        for cell in row:
            table_text += ' ' + str(cell).lower()

    # Check if any answer appears in the table
    for answer in answers:
        if answer.lower() in table_text:
            return True

    return False


def curate_training_data(
    table_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    questions: List[Dict],
    tables: Dict[str, Dict],  # table_id -> table dict
    id2table: Dict[int, str],
    table2id: Dict[str, int],
    top_k: int = 100,
    num_hard_negatives: int = 5,
    num_other_negatives: int = 10,
    use_gpu: bool = False,
) -> List[Dict]:
    """
    Curate training data with hard negatives from retrieval.

    For each question:
    1. Retrieve top-k tables from full corpus
    2. Categorize into positive, hard_negative, negative

    Returns:
        List of curated samples in NQT-Retrieval format
    """
    print(f"Building FAISS index for {table_embeddings.shape[0]} tables...")
    index, table_embs_normalized = build_faiss_index(table_embeddings, use_gpu)

    # Normalize query embeddings
    query_embs_normalized = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )

    curated_samples = []
    stats = {
        'total': 0,
        'has_positive_in_retrieved': 0,
        'avg_positives': 0,
        'avg_hard_negatives': 0,
    }

    print(f"Curating {len(questions)} training samples...")
    for idx, question in enumerate(tqdm(questions)):
        q_id = question['question_id']
        q_text = question['question']
        answers = question.get('answers', [])
        gold_table_id = question['table_id']

        # Get query embedding
        q_emb = query_embs_normalized[idx:idx+1].astype(np.float32)

        # Retrieve top-k tables
        scores, indices = index.search(q_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        # Categorize retrieved tables
        positive_ctxs = []
        hard_negative_ctxs = []
        negative_ctxs = []

        # Always include the gold table as positive
        if gold_table_id in tables:
            gold_table = tables[gold_table_id]
            positive_ctxs.append({
                'table_id': gold_table_id,
                'title': gold_table.get('title', ''),
                'score': 1.0,
                'has_answer': True,
                'is_gold': True,
            })

        for rank, (score, table_idx) in enumerate(zip(scores, indices)):
            table_id = id2table.get(int(table_idx))
            if table_id is None:
                continue

            # Skip if it's the gold table (already added)
            if table_id == gold_table_id:
                continue

            table = tables.get(table_id)
            if table is None:
                continue

            has_answer = check_answer_in_table(table, answers)

            ctx = {
                'table_id': table_id,
                'title': table.get('title', ''),
                'score': float(score),
                'has_answer': has_answer,
                'rank': rank,
            }

            if has_answer:
                positive_ctxs.append(ctx)
            elif rank < num_hard_negatives + len([p for p in positive_ctxs if p.get('rank') is not None]):
                # Top-ranked non-positive tables are hard negatives
                hard_negative_ctxs.append(ctx)
            else:
                negative_ctxs.append(ctx)

        # Limit hard negatives and other negatives
        hard_negative_ctxs = hard_negative_ctxs[:num_hard_negatives]
        negative_ctxs = negative_ctxs[:num_other_negatives]

        # Create curated sample
        curated_sample = {
            'question_id': q_id,
            'question': q_text,
            'answers': answers,
            'table_id': gold_table_id,  # Keep original gold table reference
            'positive_ctxs': positive_ctxs,
            'hard_negative_ctxs': hard_negative_ctxs,
            'negative_ctxs': negative_ctxs,
        }
        curated_samples.append(curated_sample)

        # Update stats
        stats['total'] += 1
        if len(positive_ctxs) > 0:
            stats['has_positive_in_retrieved'] += 1
        stats['avg_positives'] += len(positive_ctxs)
        stats['avg_hard_negatives'] += len(hard_negative_ctxs)

    # Finalize stats
    stats['avg_positives'] /= max(stats['total'], 1)
    stats['avg_hard_negatives'] /= max(stats['total'], 1)

    print(f"\nCuration Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Samples with positive in top-{top_k}: {stats['has_positive_in_retrieved']} ({100*stats['has_positive_in_retrieved']/stats['total']:.1f}%)")
    print(f"  Avg positives per sample: {stats['avg_positives']:.2f}")
    print(f"  Avg hard negatives per sample: {stats['avg_hard_negatives']:.2f}")

    return curated_samples


def main():
    parser = argparse.ArgumentParser(
        description='Curate training data with hard negatives from corpus retrieval'
    )

    # Input paths
    parser.add_argument('--table_embeddings', type=str, required=True,
                        help='Path to table embeddings .pkl file')
    parser.add_argument('--table_id_mapping', type=str, default=None,
                        help='Path to table_id_to_csv.json for remapping CSV-based table_ids')
    parser.add_argument('--query_embeddings', type=str, required=True,
                        help='Path to query embeddings .pkl file')
    parser.add_argument('--questions', type=str, required=True,
                        help='Path to questions JSON file')
    parser.add_argument('--tables_json', type=str, default=None,
                        help='Path to tables.json (for answer checking). If not provided, inferred from questions path.')

    # Output
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save curated training data')

    # Retrieval parameters
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of tables to retrieve per question')
    parser.add_argument('--num_hard_negatives', type=int, default=5,
                        help='Number of hard negatives per sample')
    parser.add_argument('--num_other_negatives', type=int, default=10,
                        help='Number of other negatives per sample')

    # GPU
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for FAISS')

    parser.add_argument("--embedding_type", type=str, default="column_mean",
                        choices=["column_mean", "cls_embedding", "table_embedding", "token_mean"],
                        help="Embedding type to extract from pickle (default: column_mean)")

    args = parser.parse_args()

    # Load embeddings
    print("Loading table embeddings...")
    tid_mapping = build_csv_to_table_id_mapping(args.table_id_mapping) if args.table_id_mapping else None
    table_emb_dict = load_table_embeddings(args.table_embeddings, embedding_type=args.embedding_type, table_id_mapping=tid_mapping)
    table_ids = sorted(table_emb_dict.keys())
    table_embeddings = np.stack([table_emb_dict[tid] for tid in table_ids])
    id2table = {i: tid for i, tid in enumerate(table_ids)}
    table2id = {tid: i for i, tid in enumerate(table_ids)}
    print(f"  Loaded {table_embeddings.shape[0]} table embeddings")

    print("Loading query embeddings...")
    query_emb_dict = _load_query_embeddings(args.query_embeddings)
    questions = load_training_data(args.questions)
    # Build query array aligned with questions list
    query_embeddings = np.stack([query_emb_dict[q['question_id']] for q in questions])
    print(f"  Loaded {query_embeddings.shape[0]} query embeddings")

    # Load tables for answer checking
    tables_json_path = args.tables_json
    if tables_json_path is None:
        # Infer from questions path
        questions_dir = Path(args.questions).parent
        tables_json_path = questions_dir / 'tables.json'

    print(f"Loading tables from {tables_json_path}...")
    with open(tables_json_path, 'r') as f:
        tables_list = json.load(f)
    tables = {t['table_id']: t for t in tables_list}
    print(f"  Loaded {len(tables)} tables")

    # Curate training data
    curated_samples = curate_training_data(
        table_embeddings=table_embeddings,
        query_embeddings=query_embeddings,
        questions=questions,
        tables=tables,
        id2table=id2table,
        table2id=table2id,
        top_k=args.top_k,
        num_hard_negatives=args.num_hard_negatives,
        num_other_negatives=args.num_other_negatives,
        use_gpu=args.gpu,
    )

    # Save curated data
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(curated_samples, f, indent=2)

    print(f"\nSaved curated training data to {output_path}")
    print(f"Total samples: {len(curated_samples)}")


if __name__ == '__main__':
    main()
