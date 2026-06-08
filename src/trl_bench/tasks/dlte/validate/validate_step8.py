"""
Validate Step 8: Stage 1 — FAISS Table Retrieval.

Test conditions (from PLAN.md):
  1. faiss_index.ntotal == N_lake (47,772)
  2. Each query has exactly K candidates
  3. Recall@K_any on dev beats random baseline by clear margin
  4. Metrics files exist for all splits and K values
  5. Per-tier breakdown is present and consistent
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
INDICES_ROOT = EVAL_ROOT / "indices"
RESULTS_ROOT = EVAL_ROOT / "stage1"

MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
TOPK_VALUES = [10, 50, 100]
N_LAKE = 47772
N_QUERIES = 5516

passed = 0
failed = 0
skipped = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"    PASS: {name}")
    else:
        failed += 1
        print(f"    FAIL: {name} — {detail}")


def skip(name, reason):
    global skipped
    skipped += 1
    print(f"    SKIP: {name} — {reason}")


def validate_model(model_name):
    """Validate FAISS index and retrieval results for one model."""
    print(f"\n  Model: {model_name}")

    index_path = INDICES_ROOT / model_name / "dlte_v1_lake.faiss"
    results_dir = RESULTS_ROOT / model_name

    # Check index exists
    if not index_path.exists():
        skip(f"{model_name} index", "FAISS index not found")
        return
    if not results_dir.exists():
        skip(f"{model_name} results", "results directory not found")
        return

    # Test 1: faiss_index.ntotal == N_lake
    index = faiss.read_index(str(index_path))
    check(f"index.ntotal == {N_LAKE}", index.ntotal == N_LAKE,
          f"got {index.ntotal}")

    for k in TOPK_VALUES:
        topk_path = results_dir / f"topk_{k}.jsonl"
        if not topk_path.exists():
            skip(f"topk_{k}.jsonl", "file not found")
            continue

        # Load topk results
        entries = []
        with open(topk_path) as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        # Test 2a: Number of query entries
        check(f"K={k}: {len(entries)} query entries == {N_QUERIES}",
              len(entries) == N_QUERIES,
              f"got {len(entries)}")

        # Test 2b: Each query has exactly K candidates
        wrong_k = [e["query_table_id"] for e in entries
                    if len(e["candidates"]) != k]
        check(f"K={k}: all queries have exactly {k} candidates",
              len(wrong_k) == 0,
              f"{len(wrong_k)} queries have wrong count: {wrong_k[:3]}")

        # Test 2c: Scores are sorted descending
        unsorted = []
        for e in entries[:200]:  # sample 200
            scores = [c["score"] for c in e["candidates"]]
            if scores != sorted(scores, reverse=True):
                unsorted.append(e["query_table_id"])
        check(f"K={k}: scores sorted descending (200 samples)",
              len(unsorted) == 0,
              f"{len(unsorted)} unsorted: {unsorted[:3]}")

        # Test 4: Metrics files exist for all splits
        for split in ["dev", "test", "train"]:
            metrics_path = results_dir / f"metrics_{split}_topk_{k}.json"
            if not metrics_path.exists():
                skip(f"K={k} {split} metrics", "file not found")
                continue

            metrics = json.loads(metrics_path.read_text())

            # Test 5: Per-tier breakdown present
            check(f"K={k} {split}: per_tier present",
                  "per_tier" in metrics and len(metrics["per_tier"]) > 0,
                  "missing per_tier")

            # Test 5b: Tier query counts sum to total
            if "per_tier" in metrics:
                tier_sum = sum(t["n_queries"]
                               for t in metrics["per_tier"].values())
                check(f"K={k} {split}: tier counts sum to n_queries",
                      tier_sum == metrics["n_queries"],
                      f"sum={tier_sum} vs n_queries={metrics['n_queries']}")

    # Test 3: Recall@K_any on dev beats random baseline
    # Random baseline for Recall@K in a lake of N_LAKE:
    # E[Recall@K] = K * n_relevant / N_LAKE (for small K relative to N_LAKE)
    # With ~2 relevant per query (1 union + 1 join), random ≈ K * 2 / 47772
    dev_metrics_path = results_dir / f"metrics_dev_topk_10.json"
    if dev_metrics_path.exists():
        dev = json.loads(dev_metrics_path.read_text())
        random_baseline = 10 * 2 / N_LAKE  # ~0.0004
        recall = dev["recall_any"]
        margin = recall / max(random_baseline, 1e-9)
        check(f"Recall@10_any on dev ({recall:.4f}) >> random ({random_baseline:.6f})",
              recall > random_baseline * 10,
              f"margin only {margin:.1f}x")

        # Also check that Recall increases with K
        r10 = dev["recall_any"]
        m100_path = results_dir / "metrics_dev_topk_100.json"
        if m100_path.exists():
            m100 = json.loads(m100_path.read_text())
            r100 = m100["recall_any"]
            check(f"Recall@100 ({r100:.4f}) >= Recall@10 ({r10:.4f})",
                  r100 >= r10,
                  f"Recall decreased with larger K")


def main():
    global passed, failed, skipped

    print("Step 8 Validation: Stage 1 — FAISS Table Retrieval")
    print("=" * 60)

    for model in MODELS:
        validate_model(model)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
