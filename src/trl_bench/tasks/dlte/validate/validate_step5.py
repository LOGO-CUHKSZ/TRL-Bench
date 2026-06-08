"""
Validate Step 5: Column embeddings for DLTE fragments.

Test conditions (from PLAN.md):
  1. #entries in queries.pkl == #query CSVs (5,516)
  2. #entries in targets.pkl == #target CSVs (11,032)
  3. For 100 random tables: n_cols matches column_embeddings shape
  4. dtype == float32, no NaN/Inf
  5. Constant dim within each file
  6. Sanity check: Tier 0 dev cosine(seed_col, matching_union_col) >
     cosine(seed_col, random_CKAN_col) (report median gap)

Run after all 14 SLURM jobs complete.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "column"

MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
N_QUERIES = 5516
N_TARGETS = 11032

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


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_model(model_name):
    """Validate embeddings for one model."""
    print(f"\n  Model: {model_name}")

    queries_path = EMB_ROOT / model_name / "dlte_v1_queries.pkl"
    targets_path = EMB_ROOT / model_name / "dlte_v1_targets.pkl"

    # Check files exist
    if not queries_path.exists():
        skip(f"{model_name} queries.pkl", "file not found")
        return
    if not targets_path.exists():
        skip(f"{model_name} targets.pkl", "file not found")
        return

    queries = load_pkl(queries_path)
    targets = load_pkl(targets_path)

    # Test 1-2: Entry counts
    check(f"queries count == {N_QUERIES}", len(queries) == N_QUERIES,
          f"got {len(queries)}")
    check(f"targets count == {N_TARGETS}", len(targets) == N_TARGETS,
          f"got {len(targets)}")

    # Test 3: Column count matches (sample 100)
    rng = np.random.RandomState(42)
    all_entries = queries + targets
    sample_idx = rng.choice(len(all_entries), min(100, len(all_entries)), replace=False)
    col_mismatches = []
    for idx in sample_idx:
        entry = all_entries[idx]
        col_embs = entry.get("column_embeddings", {})
        col_names = entry.get("column_names", [])
        n_emb_cols = len(col_embs)
        n_name_cols = len(col_names)
        if n_emb_cols != n_name_cols:
            col_mismatches.append(
                f"{entry.get('table_id', '?')}: {n_emb_cols} emb cols vs {n_name_cols} names")

    check("column_embeddings count matches column_names (100 samples)",
          len(col_mismatches) == 0,
          f"{len(col_mismatches)} mismatches: {col_mismatches[:3]}")

    # Test 4: dtype and NaN/Inf check (sample)
    bad_dtype = []
    has_nan_inf = []
    for idx in sample_idx[:50]:
        entry = all_entries[idx]
        for col_idx, emb in entry.get("column_embeddings", {}).items():
            arr = np.asarray(emb)
            if arr.dtype != np.float32:
                bad_dtype.append(f"{entry.get('table_id', '?')}:col{col_idx}={arr.dtype}")
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                has_nan_inf.append(f"{entry.get('table_id', '?')}:col{col_idx}")

    check("dtype is float32 (50 samples)",
          len(bad_dtype) == 0,
          f"{len(bad_dtype)} violations: {bad_dtype[:3]}")
    check("no NaN/Inf (50 samples)",
          len(has_nan_inf) == 0,
          f"{len(has_nan_inf)} violations: {has_nan_inf[:3]}")

    # Test 5: Constant dim
    dims = set()
    for entry in all_entries[:200]:
        for col_idx, emb in entry.get("column_embeddings", {}).items():
            dims.add(np.asarray(emb).shape[-1])
            break  # one col per entry is enough

    check(f"constant embedding dim (found: {dims})",
          len(dims) == 1,
          f"multiple dims: {dims}")

    return queries, targets


def sanity_check(model_name, queries, targets):
    """Test 6: Tier 0 dev cosine similarity sanity check."""
    print(f"\n  Sanity check: {model_name}")

    # Load fragment manifest and query tasks for Tier 0 dev
    fragments = []
    with open(DATASET_ROOT / "manifests" / "fragments_manifest.jsonl") as f:
        for line in f:
            fragments.append(json.loads(line.strip()))

    # Build lookup: table_id → fragment info
    frag_lookup = {f["table_id"]: f for f in fragments}

    # Build embedding lookups
    q_emb = {e["table_id"]: e for e in queries}
    t_emb = {e["table_id"]: e for e in targets}

    # Load CKAN embeddings for random baseline
    ckan_path = EMB_ROOT / model_name / "ckan_subset.pkl"
    if not ckan_path.exists():
        skip(f"{model_name} sanity check", "ckan_subset.pkl not found")
        return

    ckan_data = load_pkl(ckan_path)
    ckan_emb = {e["table_id"]: e for e in ckan_data}

    # Get Tier 0 dev seed-union pairs
    query_tasks = []
    with open(DATASET_ROOT / "ground_truth" / "query_tasks.jsonl") as f:
        for line in f:
            qt = json.loads(line.strip())
            if qt["noise_tier"] == 0 and qt["split"] == "dev":
                query_tasks.append(qt)

    rng = np.random.RandomState(42)
    n_check = min(100, len(query_tasks))
    sample_tasks = rng.choice(len(query_tasks), n_check, replace=False)

    ckan_ids = list(ckan_emb.keys())
    matched_sims = []
    random_sims = []

    for idx in sample_tasks:
        qt = query_tasks[idx]
        seed_id = qt["query_table_id"]

        # Find union target
        union_rel = [r for r in qt["relevant"] if r["relation"] == "union"]
        if not union_rel:
            continue
        union_id = union_rel[0]["table_id"]

        if seed_id not in q_emb or union_id not in t_emb:
            continue

        seed_entry = q_emb[seed_id]
        union_entry = t_emb[union_id]

        # Get ground truth column mapping
        gt_dir = DATASET_ROOT / "ground_truth" / "table_maps"
        seed_gt = np.load(gt_dir / f"{seed_id}.npz")
        union_gt = np.load(gt_dir / f"{union_id}.npz")

        seed_col_parent = seed_gt["col_parent_idx"]
        union_col_parent = union_gt["col_parent_idx"]

        # Match columns by parent index
        seed_cols = seed_entry.get("column_embeddings", {})
        union_cols = union_entry.get("column_embeddings", {})

        for s_pos, s_parent_idx in enumerate(seed_col_parent):
            if s_pos not in seed_cols:
                continue
            s_vec = np.asarray(seed_cols[s_pos], dtype=np.float32)
            if np.linalg.norm(s_vec) == 0:
                continue

            # Find matching union column (same parent col idx)
            for u_pos, u_parent_idx in enumerate(union_col_parent):
                if u_parent_idx == s_parent_idx and u_pos in union_cols:
                    u_vec = np.asarray(union_cols[u_pos], dtype=np.float32)
                    if np.linalg.norm(u_vec) > 0:
                        sim = np.dot(s_vec, u_vec) / (
                            np.linalg.norm(s_vec) * np.linalg.norm(u_vec))
                        matched_sims.append(sim)
                    break

            # Random CKAN column
            rand_ckan_id = ckan_ids[rng.randint(0, len(ckan_ids))]
            rand_entry = ckan_emb[rand_ckan_id]
            rand_cols = rand_entry.get("column_embeddings", {})
            if rand_cols:
                rand_col_idx = list(rand_cols.keys())[0]
                r_vec = np.asarray(rand_cols[rand_col_idx], dtype=np.float32)
                if np.linalg.norm(r_vec) > 0:
                    sim = np.dot(s_vec, r_vec) / (
                        np.linalg.norm(s_vec) * np.linalg.norm(r_vec))
                    random_sims.append(sim)

    if matched_sims and random_sims:
        med_matched = np.median(matched_sims)
        med_random = np.median(random_sims)
        gap = med_matched - med_random
        print(f"    INFO: median_matched={med_matched:.4f}, "
              f"median_random={med_random:.4f}, gap={gap:.4f}")
        check(f"matched > random (gap={gap:.4f})",
              med_matched > med_random,
              f"median matched ({med_matched:.4f}) <= random ({med_random:.4f})")
    else:
        skip("sanity check", f"insufficient pairs: {len(matched_sims)} matched, {len(random_sims)} random")


def main():
    global passed, failed, skipped

    print("Step 5 Validation: Column Embeddings for DLTE")
    print("=" * 60)

    results = {}
    for model in MODELS:
        result = validate_model(model)
        if result:
            results[model] = result

    # Run sanity checks for models that passed basic validation
    for model in results:
        queries, targets = results[model]
        sanity_check(model, queries, targets)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if skipped > 0:
        print(f"\nNote: {skipped} checks skipped (likely embeddings not yet generated).")
        print("Re-run after all SLURM jobs complete.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
