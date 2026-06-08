"""
Step 8b: Re-rank Stage 1 FAISS candidates using Hungarian column matching.

Takes the top-100 FAISS candidates per query and re-scores them using
bipartite column matching (Hungarian algorithm on column-level cosine
similarities). This isolates the effect of the scoring function from the
embedding model.

Usage:
    python downstream_tasks/dlte/scripts/step8b_hungarian_rerank.py
    python downstream_tasks/dlte/scripts/step8b_hungarian_rerank.py --models bert starmie --topk 10 50 100
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = COL_EMB_ROOT = DATASET_ROOT = RESULTS_ROOT = None


def resolve_paths(args):
    global PROJECT_ROOT, COL_EMB_ROOT, DATASET_ROOT, RESULTS_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    COL_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "column"
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
    RESULTS_ROOT = output_root / "stage1"

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
DEFAULT_TOPK = [10, 50, 100]
SIMILARITY_THRESHOLD = 0.6  # Starmie's default


def load_query_tasks():
    """Load ground truth query tasks."""
    tasks = []
    with open(DATASET_ROOT / "ground_truth" / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def compute_metrics(query_tasks, results, k):
    """Compute Recall@K and MRR broken down by split, tier, and relation."""
    metrics_by_split = {}
    for split in ["dev", "test", "train"]:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue

        recall_any = []
        recall_union = []
        recall_join = []
        mrr_any = []
        tier_recall_any = defaultdict(list)
        tier_recall_union = defaultdict(list)
        tier_recall_join = defaultdict(list)

        for qt in split_tasks:
            qid = qt["query_table_id"]
            tier = qt["noise_tier"]
            if qid not in results:
                continue

            candidates = results[qid]
            candidate_set = set(candidates)

            gt_union = [r["table_id"] for r in qt["relevant"] if r["relation"] == "union"]
            gt_join = [r["table_id"] for r in qt["relevant"] if r["relation"] == "join"]
            gt_all = set(gt_union + gt_join)

            found_any = len(gt_all & candidate_set)
            found_union = len(set(gt_union) & candidate_set)
            found_join = len(set(gt_join) & candidate_set)

            recall_any.append(found_any / len(gt_all) if gt_all else 0)
            recall_union.append(found_union / len(gt_union) if gt_union else 0)
            recall_join.append(found_join / len(gt_join) if gt_join else 0)

            tier_recall_any[tier].append(recall_any[-1])
            tier_recall_union[tier].append(recall_union[-1])
            tier_recall_join[tier].append(recall_join[-1])

            rr = 0.0
            for rank, cid in enumerate(candidates, 1):
                if cid in gt_all:
                    rr = 1.0 / rank
                    break
            mrr_any.append(rr)

        metrics_by_split[split] = {
            "k": k,
            "n_queries": len(split_tasks),
            "recall_any": float(np.mean(recall_any)) if recall_any else 0,
            "recall_union": float(np.mean(recall_union)) if recall_union else 0,
            "recall_join": float(np.mean(recall_join)) if recall_join else 0,
            "mrr_any": float(np.mean(mrr_any)) if mrr_any else 0,
            "per_tier": {
                tier: {
                    "recall_any": float(np.mean(tier_recall_any[tier])),
                    "recall_union": float(np.mean(tier_recall_union[tier])),
                    "recall_join": float(np.mean(tier_recall_join[tier])),
                    "n_queries": len(tier_recall_any[tier]),
                }
                for tier in sorted(tier_recall_any.keys())
            },
        }
    return metrics_by_split


def hungarian_score(query_cols, candidate_cols, threshold=SIMILARITY_THRESHOLD):
    """Compute Hungarian matching score between two tables' column embeddings.

    Args:
        query_cols: dict of col_idx -> embedding vector
        candidate_cols: dict of col_idx -> embedding vector
        threshold: minimum cosine similarity to consider a match

    Returns:
        Normalized matching score (sum of matched similarities / min(n_q, n_c)).
    """
    q_vecs = np.array([query_cols[k] for k in sorted(query_cols.keys())], dtype=np.float32)
    c_vecs = np.array([candidate_cols[k] for k in sorted(candidate_cols.keys())], dtype=np.float32)

    # L2-normalize for cosine similarity
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    c_norms[c_norms == 0] = 1.0
    q_normed = q_vecs / q_norms
    c_normed = c_vecs / c_norms

    # Pairwise cosine similarity matrix
    sim_matrix = q_normed @ c_normed.T  # (n_q, n_c)

    # Threshold: zero out similarities below threshold
    sim_matrix[sim_matrix < threshold] = 0.0

    # Convert to cost matrix for minimization
    cost = sim_matrix.max() - sim_matrix if sim_matrix.max() > 0 else -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    # Score = sum of matched similarities, normalized
    score = sim_matrix[row_ind, col_ind].sum()
    n_min = min(len(q_vecs), len(c_vecs))
    return float(score / n_min) if n_min > 0 else 0.0


def process_model(model_name, topk_values, query_tasks, threshold=SIMILARITY_THRESHOLD):
    """Re-rank FAISS candidates using Hungarian matching for one model."""
    print(f"\n  Model: {model_name}")
    t0 = time.time()

    # Load column embeddings
    print("    Loading column embeddings...")
    t_load = time.time()
    with open(COL_EMB_ROOT / model_name / "dlte_v1_queries.pkl", "rb") as f:
        queries_pkl = pickle.load(f)
    with open(COL_EMB_ROOT / model_name / "dlte_v1_targets.pkl", "rb") as f:
        targets_pkl = pickle.load(f)
    with open(COL_EMB_ROOT / model_name / "ckan_subset.pkl", "rb") as f:
        ckan_pkl = pickle.load(f)
    print(f"    Loaded in {time.time() - t_load:.1f}s")

    # Build lookups: table_id -> column_embeddings dict
    query_lookup = {e["table_id"]: e["column_embeddings"] for e in queries_pkl}
    lake_lookup = {e["table_id"]: e["column_embeddings"] for e in targets_pkl}
    lake_lookup.update({e["table_id"]: e["column_embeddings"] for e in ckan_pkl})
    print(f"    Lookups: {len(query_lookup)} queries, {len(lake_lookup)} lake")

    # Free pickle memory
    del queries_pkl, targets_pkl, ckan_pkl

    # Load top-100 FAISS results
    faiss_path = RESULTS_ROOT / model_name / "topk_100.jsonl"
    if not faiss_path.exists():
        print(f"    SKIP: {faiss_path} not found")
        return False

    faiss_entries = []
    with open(faiss_path) as f:
        for line in f:
            faiss_entries.append(json.loads(line.strip()))
    print(f"    Loaded {len(faiss_entries)} FAISS entries")

    # Re-score all candidates using Hungarian matching
    print("    Computing Hungarian scores...")
    t_score = time.time()
    reranked_entries = []
    n_missing = 0
    for i, entry in enumerate(faiss_entries):
        qid = entry["query_table_id"]
        q_cols = query_lookup.get(qid)
        if q_cols is None:
            n_missing += 1
            reranked_entries.append(entry)  # keep original if no embeddings
            continue

        scored_candidates = []
        for cand in entry["candidates"]:
            cid = cand["table_id"]
            c_cols = lake_lookup.get(cid)
            if c_cols is None:
                scored_candidates.append({"table_id": cid, "score": 0.0})
            else:
                score = hungarian_score(q_cols, c_cols, threshold=threshold)
                scored_candidates.append({"table_id": cid, "score": round(score, 6)})

        # Sort by Hungarian score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        reranked_entries.append({
            "query_table_id": qid,
            "candidates": scored_candidates,
        })

        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{len(faiss_entries)} queries scored...")

    elapsed_score = time.time() - t_score
    print(f"    Scored in {elapsed_score:.1f}s ({n_missing} queries missing embeddings)")

    # Save results and metrics for each K
    results_dir = RESULTS_ROOT / model_name
    for k in topk_values:
        results = {}
        topk_entries = []
        for entry in reranked_entries:
            qid = entry["query_table_id"]
            top_k_cands = entry["candidates"][:k]
            candidate_ids = [c["table_id"] for c in top_k_cands]
            results[qid] = candidate_ids
            topk_entries.append({
                "query_table_id": qid,
                "candidates": top_k_cands,
            })

        # Save re-ranked results
        with open(results_dir / f"topk_{k}_hungarian.jsonl", "w") as f:
            for e in topk_entries:
                f.write(json.dumps(e) + "\n")

        # Compute and save metrics
        metrics = compute_metrics(query_tasks, results, k)
        for split, split_metrics in metrics.items():
            with open(results_dir / f"metrics_{split}_topk_{k}_hungarian.json", "w") as f:
                json.dump(split_metrics, f, indent=2)

        # Print summary for dev split
        if "dev" in metrics:
            dev = metrics["dev"]
            print(f"    K={k:3d}: Recall@K_any={dev['recall_any']:.4f}  "
                  f"union={dev['recall_union']:.4f}  "
                  f"join={dev['recall_join']:.4f}  "
                  f"MRR={dev['mrr_any']:.4f}")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return True


def print_comparison(models):
    """Print side-by-side comparison of FAISS vs Hungarian results."""
    print(f"\n{'='*80}")
    print("COMPARISON: FAISS (inner product) vs Hungarian (column matching)")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Method':<10} {'R@10_any':>9} {'R@10_U':>7} {'R@10_J':>7} "
          f"{'R@100_any':>10} {'R@100_U':>8} {'R@100_J':>8} {'MRR':>6}")
    print("-" * 80)

    for model in models:
        for method, suffix in [("FAISS", ""), ("Hungarian", "_hungarian")]:
            path = RESULTS_ROOT / model / f"metrics_dev_topk_10{suffix}.json"
            path100 = RESULTS_ROOT / model / f"metrics_dev_topk_100{suffix}.json"
            if not path.exists() or not path100.exists():
                continue
            m10 = json.loads(path.read_text())
            m100 = json.loads(path100.read_text())
            print(f"{model:<12} {method:<10} "
                  f"{m10['recall_any']:>9.4f} {m10['recall_union']:>7.4f} {m10['recall_join']:>7.4f} "
                  f"{m100['recall_any']:>10.4f} {m100['recall_union']:>8.4f} {m100['recall_join']:>8.4f} "
                  f"{m10['mrr_any']:>6.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Re-rank Stage 1 FAISS candidates using Hungarian column matching")
    parser.add_argument("--models", nargs="+", default=COLUMN_MODELS)
    parser.add_argument("--topk", nargs="+", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Cosine similarity threshold for matching")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory (default: auto-detect)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output root for evaluation results (default: {project_root}/results/evaluation/dlte)")
    args = parser.parse_args()
    resolve_paths(args)

    print("Step 8b: Hungarian Re-ranking of Stage 1 Candidates")
    print("=" * 60)
    print(f"Threshold: {args.threshold}")

    query_tasks = load_query_tasks()
    print(f"Loaded {len(query_tasks)} query tasks")

    succeeded = 0
    for model in args.models:
        if process_model(model, args.topk, query_tasks, threshold=args.threshold):
            succeeded += 1

    print(f"\n{'='*60}")
    print(f"Processed {succeeded}/{len(args.models)} models")

    # Print comparison table
    print_comparison(args.models)

    return 0 if succeeded == len(args.models) else 1


if __name__ == "__main__":
    sys.exit(main())
