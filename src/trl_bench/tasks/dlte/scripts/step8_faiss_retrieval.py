"""
Step 8: Stage 1 — Table Retrieval via FAISS.

Build FAISS inner-product index on lake embeddings and retrieve top-K
candidates for each query table. Compute Recall@K and MRR metrics
broken down by split, noise tier, and relation type.

Reads table embeddings from embeddings/table/{model}/{dataset}.pkl,
extracting the specified variant (default: 'column_mean').
Vectors are L2-normalized before FAISS inner-product search so that
IP score = cosine similarity.

Usage:
    python downstream_tasks/dlte/scripts/step8_faiss_retrieval.py
    python downstream_tasks/dlte/scripts/step8_faiss_retrieval.py --models bert tabbie --topk 10 50 100
"""

import argparse
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
# Native table-level models: model_name -> default table_variant.
# TUTA is also a native table model (cls_embedding) but participates in
# DLTE as a row model (step10), not table retrieval — add it here if needed.
NATIVE_TABLE_MODELS = {
    "tapex": "table_embedding",
}
DEFAULT_TOPK = [10, 50, 100]

# Resolved at runtime by resolve_paths()
PROJECT_ROOT = TABLE_EMB_ROOT = DATASET_ROOT = INDICES_ROOT = RESULTS_ROOT = None


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, TABLE_EMB_ROOT, DATASET_ROOT, INDICES_ROOT, RESULTS_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    emb_base = Path(args.embeddings_root) if getattr(args, 'embeddings_root', None) else PROJECT_ROOT / "assets" / "embeddings"
    TABLE_EMB_ROOT = emb_base / "table"
    data_root = Path(args.data_root) if getattr(args, 'data_root', None) else PROJECT_ROOT
    DATASET_ROOT = data_root / "datasets" / "dlte_v1"
    INDICES_ROOT = output_root / "indices"
    RESULTS_ROOT = output_root / "stage1"


def load_query_tasks():
    """Load ground truth query tasks."""
    tasks = []
    with open(DATASET_ROOT / "ground_truth" / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def compute_metrics(query_tasks, lake_ids_list, results, k):
    """Compute Recall@K and MRR broken down by split, tier, and relation.

    Args:
        query_tasks: List of ground truth query task dicts.
        lake_ids_list: List of lake table IDs (index-aligned with FAISS).
        results: Dict of query_table_id -> list of candidate table_ids (top-K).
        k: The K value used.

    Returns:
        Dict of metric dicts keyed by split.
    """
    # Build query_table_id -> task lookup
    task_lookup = {qt["query_table_id"]: qt for qt in query_tasks}

    # Group by split
    metrics_by_split = {}
    for split in ["dev", "test", "train"]:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue

        # Accumulators
        recall_any = []
        recall_union = []
        recall_join = []
        mrr_any = []

        # Per-tier accumulators
        tier_recall_any = defaultdict(list)
        tier_recall_union = defaultdict(list)
        tier_recall_join = defaultdict(list)

        n_skipped = 0

        for qt in split_tasks:
            qid = qt["query_table_id"]
            tier = qt["noise_tier"]

            if qid not in results or not results[qid]:
                recall_any.append(0.0)
                recall_union.append(0.0)
                recall_join.append(0.0)
                mrr_any.append(0.0)
                tier_recall_any[tier].append(0.0)
                tier_recall_union[tier].append(0.0)
                tier_recall_join[tier].append(0.0)
                n_skipped += 1
                continue

            candidates = results[qid]
            candidate_set = set(candidates)

            # Ground truth relevant tables
            gt_union = [r["table_id"] for r in qt["relevant"] if r["relation"] == "union"]
            gt_join = [r["table_id"] for r in qt["relevant"] if r["relation"] == "join"]
            gt_all = set(gt_union + gt_join)

            # Recall@K: fraction of relevant found in top-K
            found_any = len(gt_all & candidate_set)
            found_union = len(set(gt_union) & candidate_set)
            found_join = len(set(gt_join) & candidate_set)

            recall_any.append(found_any / len(gt_all) if gt_all else 0)
            recall_union.append(found_union / len(gt_union) if gt_union else 0)
            recall_join.append(found_join / len(gt_join) if gt_join else 0)

            tier_recall_any[tier].append(recall_any[-1])
            tier_recall_union[tier].append(recall_union[-1])
            tier_recall_join[tier].append(recall_join[-1])

            # MRR: reciprocal rank of first relevant hit
            rr = 0.0
            for rank, cid in enumerate(candidates, 1):
                if cid in gt_all:
                    rr = 1.0 / rank
                    break
            mrr_any.append(rr)

        metrics_by_split[split] = {
            "k": k,
            "n_queries": len(split_tasks),
            "n_evaluated": len(split_tasks) - n_skipped,
            "n_skipped": n_skipped,
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


def _load_table_embeddings(pkl_path, variant="column_mean"):
    """Load table embeddings from a table embedding pkl.

    Args:
        pkl_path: Path to the table embedding pickle file.
        variant: Which table embedding variant to load (default: column_mean).

    Returns:
        Dict mapping table_id -> np.ndarray.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    result = {}
    skipped = 0
    for item in data:
        table_id = item.get("table_id", "")
        table_emb = item.get("table_embedding", {})
        if isinstance(table_emb, dict):
            emb = table_emb.get(variant)
            if emb is not None:
                result[table_id] = np.asarray(emb, dtype=np.float32)
            else:
                skipped += 1
    if skipped:
        logger.warning(
            "Skipped %d/%d tables missing variant '%s' in %s",
            skipped, skipped + len(result), variant, pkl_path,
        )
    return result


def _build_arrays_from_pkls(model_name, table_variant="column_mean"):
    """Build lake and query embedding arrays from table embedding pkls.

    Reads dlte_v1_queries.pkl, dlte_v1_targets.pkl, ckan_subset.pkl from
    the table embedding directory and constructs ordered arrays matching
    the lake manifest.

    Returns:
        (lake_emb, lake_ids, query_emb, query_ids)
    """
    emb_dir = TABLE_EMB_ROOT / model_name

    # Load table embeddings from pkls using the specified variant
    queries_pkl = emb_dir / "dlte_v1_queries.pkl"
    targets_pkl = emb_dir / "dlte_v1_targets.pkl"
    ckan_pkl = emb_dir / "ckan_subset.pkl"

    for p in [queries_pkl, targets_pkl, ckan_pkl]:
        if not p.exists():
            raise FileNotFoundError(f"Table embedding not found: {p}")

    q_lookup = _load_table_embeddings(queries_pkl, variant=table_variant)
    t_lookup = _load_table_embeddings(targets_pkl, variant=table_variant)
    c_lookup = _load_table_embeddings(ckan_pkl, variant=table_variant)

    if not q_lookup or not t_lookup or not c_lookup:
        empty = [name for name, lk in [("queries", q_lookup), ("targets", t_lookup), ("ckan", c_lookup)] if not lk]
        raise ValueError(
            f"Variant '{table_variant}' produced 0 embeddings for {', '.join(empty)} "
            f"(model={model_name}). Check that the variant exists in the pkl files."
        )

    print(f"    Loaded: queries={len(q_lookup)}, targets={len(t_lookup)}, ckan={len(c_lookup)}")

    # Build lake arrays ordered by manifest
    lake_manifest_path = DATASET_ROOT / "manifests" / "lake_manifest.jsonl"
    lake_ids = []
    lake_vecs = []
    missing = 0
    with open(lake_manifest_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            tid = entry["table_id"]
            vec = t_lookup.get(tid)
            if vec is None:
                vec = c_lookup.get(tid)
            if vec is not None:
                lake_ids.append(tid)
                lake_vecs.append(vec)
            else:
                missing += 1

    if missing > 0:
        print(f"    WARN: {missing} lake entries missing embeddings")

    # Build query arrays (ordered by pkl order)
    query_ids = list(q_lookup.keys())
    query_vecs = [q_lookup[tid] for tid in query_ids]

    lake_emb = np.ascontiguousarray(np.stack(lake_vecs), dtype=np.float32)
    query_emb = np.ascontiguousarray(np.stack(query_vecs), dtype=np.float32)

    return lake_emb, lake_ids, query_emb, query_ids


def process_model(model_name, topk_values, query_tasks, table_variant="column_mean"):
    """Build FAISS index and retrieve for one model."""
    print(f"\n  Model: {model_name}")
    t0 = time.time()

    lake_emb, lake_ids, query_emb, query_ids = _build_arrays_from_pkls(model_name, table_variant=table_variant)

    dim = lake_emb.shape[1]
    print(f"    lake: {lake_emb.shape}, queries: {query_emb.shape}, dim={dim}")

    # L2-normalize so inner product = cosine similarity
    faiss.normalize_L2(lake_emb)
    faiss.normalize_L2(query_emb)

    # Build FAISS index (inner product on unit vectors = cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(lake_emb)
    print(f"    FAISS index built: {index.ntotal} vectors")

    # Save index
    index_dir = INDICES_ROOT / model_name
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "dlte_v1_lake.faiss"))

    # Search for max K
    max_k = max(topk_values)
    distances, indices = index.search(query_emb, max_k)

    # Save results and metrics for each K
    results_dir = RESULTS_ROOT / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for k in topk_values:
        # Build results dict
        results = {}
        topk_entries = []
        for q_idx, qid in enumerate(query_ids):
            candidates = []
            candidate_ids = []
            for rank in range(k):
                lake_idx = int(indices[q_idx, rank])
                if lake_idx < 0:
                    break
                score = float(distances[q_idx, rank])
                cid = lake_ids[lake_idx]
                candidates.append({"table_id": cid, "score": round(score, 6)})
                candidate_ids.append(cid)
            results[qid] = candidate_ids
            topk_entries.append({
                "query_table_id": qid,
                "candidates": candidates,
            })

        # Ensure every query task has an entry (missing embedding → empty candidates)
        for qt in query_tasks:
            qid = qt["query_table_id"]
            if qid not in results:
                results[qid] = []
                topk_entries.append({"query_table_id": qid, "candidates": []})

        # Save top-K results
        with open(results_dir / f"topk_{k}.jsonl", "w") as f:
            for entry in topk_entries:
                f.write(json.dumps(entry) + "\n")

        # Compute and save metrics
        metrics = compute_metrics(query_tasks, lake_ids, results, k)
        for split, split_metrics in metrics.items():
            with open(results_dir / f"metrics_{split}_topk_{k}.json", "w") as f:
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


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: FAISS table retrieval for DLTE")
    parser.add_argument("--models", nargs="+",
                        default=COLUMN_MODELS + list(NATIVE_TABLE_MODELS))
    parser.add_argument("--topk", nargs="+", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--table_variant", type=str, default="column_mean",
                        help="Table embedding variant to use (default: column_mean)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root for DLTE outputs (default: {project_root}/results/evaluation/dlte)")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root (default: derived from script location)")
    parser.add_argument("--embeddings_root", type=str, default=None,
                        help="Embeddings root (default: {project_root}/embeddings)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Data root containing 'datasets/dlte_v1/' (default: {project_root})")
    args = parser.parse_args()

    resolve_paths(args)

    print("Step 8: Stage 1 — Table Retrieval (FAISS)")
    print("=" * 60)

    query_tasks = load_query_tasks()
    print(f"Loaded {len(query_tasks)} query tasks")

    succeeded = 0
    for model in args.models:
        variant = NATIVE_TABLE_MODELS.get(model, args.table_variant)
        if process_model(model, args.topk, query_tasks, table_variant=variant):
            succeeded += 1

    print(f"\n{'='*60}")
    print(f"Processed {succeeded}/{len(args.models)} models")
    print(f"Indices: {INDICES_ROOT}")
    print(f"Results: {RESULTS_ROOT}")
    print(f"{'='*60}")

    return 0 if succeeded == len(args.models) else 1


if __name__ == "__main__":
    sys.exit(main())
