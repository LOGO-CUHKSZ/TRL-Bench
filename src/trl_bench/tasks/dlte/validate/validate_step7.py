"""
Validate Step 7: Table-level embeddings via aggregation.

Test conditions (from PLAN.md):
  1. len(lake_table_ids) == N_lake == count in lake_manifest (47,772)
  2. len(query_table_ids) == N_queries (5,516)
  3. For 1000 random vectors: abs(||v||_2 - 1.0) < 1e-3
  4. Every ID resolvable back to source embeddings
  5. dtype == float32, contiguous arrays (FAISS requirement)
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
TABLE_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "table" / "column"
COL_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "column"

MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
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
    """Validate table embeddings for one column model."""
    print(f"\n  Model: {model_name}")

    out_dir = TABLE_EMB_ROOT / model_name
    lake_emb_path = out_dir / "dlte_v1_lake_embeddings.npy"
    lake_ids_path = out_dir / "dlte_v1_lake_table_ids.txt"
    query_emb_path = out_dir / "dlte_v1_query_embeddings.npy"
    query_ids_path = out_dir / "dlte_v1_query_table_ids.txt"
    config_path = out_dir / "dlte_v1_aggregation.json"

    # Check files exist
    for p in [lake_emb_path, lake_ids_path, query_emb_path, query_ids_path, config_path]:
        if not p.exists():
            skip(f"{model_name}", f"{p.name} not found")
            return

    # Load
    lake_emb = np.load(lake_emb_path)
    query_emb = np.load(query_emb_path)
    lake_ids = lake_ids_path.read_text().strip().split("\n")
    query_ids = query_ids_path.read_text().strip().split("\n")

    # Test 1: Lake count matches manifest
    check(f"lake count == {N_LAKE}", len(lake_ids) == N_LAKE,
          f"got {len(lake_ids)}")
    check(f"lake embeddings shape[0] == {N_LAKE}",
          lake_emb.shape[0] == N_LAKE,
          f"got {lake_emb.shape[0]}")

    # Test 2: Query count
    check(f"query count == {N_QUERIES}", len(query_ids) == N_QUERIES,
          f"got {len(query_ids)}")
    check(f"query embeddings shape[0] == {N_QUERIES}",
          query_emb.shape[0] == N_QUERIES,
          f"got {query_emb.shape[0]}")

    # Test 3: L2 norm ≈ 1.0 for 1000 random vectors
    rng = np.random.RandomState(42)
    sample_lake = rng.choice(lake_emb.shape[0], min(1000, lake_emb.shape[0]), replace=False)
    norms = np.linalg.norm(lake_emb[sample_lake], axis=1)
    max_deviation = float(np.max(np.abs(norms - 1.0)))
    check(f"L2 norm ≈ 1.0 (max deviation={max_deviation:.6f})",
          max_deviation < 1e-3,
          f"max deviation {max_deviation:.6f} >= 1e-3")

    # Also check query norms
    q_norms = np.linalg.norm(query_emb, axis=1)
    q_max_dev = float(np.max(np.abs(q_norms - 1.0)))
    check(f"query L2 norm ≈ 1.0 (max deviation={q_max_dev:.6f})",
          q_max_dev < 1e-3,
          f"max deviation {q_max_dev:.6f} >= 1e-3")

    # Test 4: Every ID resolvable back to source embeddings
    # Load source pickle IDs for spot check
    import pickle
    with open(COL_EMB_ROOT / model_name / "dlte_v1_queries.pkl", "rb") as f:
        q_pkl = pickle.load(f)
    q_pkl_ids = {e["table_id"] for e in q_pkl}

    with open(COL_EMB_ROOT / model_name / "dlte_v1_targets.pkl", "rb") as f:
        t_pkl = pickle.load(f)
    t_pkl_ids = {e["table_id"] for e in t_pkl}

    with open(COL_EMB_ROOT / model_name / "ckan_subset.pkl", "rb") as f:
        c_pkl = pickle.load(f)
    c_pkl_ids = {e["table_id"] for e in c_pkl}

    all_source_ids = q_pkl_ids | t_pkl_ids | c_pkl_ids

    unresolvable_query = [tid for tid in query_ids if tid not in q_pkl_ids]
    unresolvable_lake = [tid for tid in lake_ids if tid not in all_source_ids]

    check("all query IDs resolvable to source",
          len(unresolvable_query) == 0,
          f"{len(unresolvable_query)} unresolvable: {unresolvable_query[:3]}")
    check("all lake IDs resolvable to source",
          len(unresolvable_lake) == 0,
          f"{len(unresolvable_lake)} unresolvable: {unresolvable_lake[:3]}")

    # Test 5: dtype and contiguity
    check("lake dtype == float32", lake_emb.dtype == np.float32,
          f"got {lake_emb.dtype}")
    check("query dtype == float32", query_emb.dtype == np.float32,
          f"got {query_emb.dtype}")
    check("lake array is C-contiguous", lake_emb.flags["C_CONTIGUOUS"],
          "not contiguous")
    check("query array is C-contiguous", query_emb.flags["C_CONTIGUOUS"],
          "not contiguous")


def main():
    global passed, failed, skipped

    print("Step 7 Validation: Table-level Embeddings")
    print("=" * 60)

    for model in MODELS:
        validate_model(model)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
