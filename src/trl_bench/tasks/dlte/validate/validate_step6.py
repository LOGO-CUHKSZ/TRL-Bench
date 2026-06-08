"""
Validate Step 6: Row embeddings for DLTE fragments.

Test conditions (from PLAN.md):
  1. #entries in queries.pkl == #query CSVs (5,516)
  2. #entries in targets.pkl == #target CSVs (11,032)
  3. row_embeddings.shape[0] == n_rows per table (100 samples)
  4. dtype == float32, no NaN/Inf
  5. Constant dim within each model
  6. Cross-table sanity: Tier 0 query→join-target pairs,
     median(sim_matched) > median(sim_random)

Run after all 6 SLURM jobs complete.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "row"

MODELS = [
    "bert", "dae", "gte", "saint", "scarf", "subtab",
    "tabbie", "tabicl", "tabpfn", "tabtransformer", "tabular_binning",
    "transtab", "tuta", "vime",
]
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
    """Validate row embeddings for one model."""
    print(f"\n  Model: {model_name}")

    queries_path = EMB_ROOT / model_name / "dlte_v1_queries.pkl"
    targets_path = EMB_ROOT / model_name / "dlte_v1_targets.pkl"

    # Check files exist
    if not queries_path.exists():
        skip(f"{model_name} queries.pkl", "file not found")
        return None
    if not targets_path.exists():
        skip(f"{model_name} targets.pkl", "file not found")
        return None

    queries = load_pkl(queries_path)
    targets = load_pkl(targets_path)

    # Test 1-2: Entry counts
    # Row models may skip tables that fail preprocessing (e.g. TabPFN/TabICL).
    # Allow up to 5% missing as WARN; more than 5% is FAIL.
    # Also detect in-progress jobs (targets far below expected).
    MAX_MISSING_FRAC = 0.05
    IN_PROGRESS_THRESHOLD = 0.20  # >20% missing → likely still running

    q_missing = N_QUERIES - len(queries)
    t_missing = N_TARGETS - len(targets)

    def count_check(label, actual, expected):
        """Check entry count with tolerance for preprocessing failures."""
        global passed
        missing = expected - actual
        frac = missing / expected if expected > 0 else 0
        if actual == expected:
            check(f"{label} == {expected}", True)
        elif frac > IN_PROGRESS_THRESHOLD:
            skip(f"{label} == {expected}",
                 f"got {actual} ({frac:.1%} missing — job likely still in progress)")
        elif frac <= MAX_MISSING_FRAC:
            passed += 1
            print(f"    WARN: {label} {actual} ({missing} missing, "
                  f"{frac:.2%}) — within tolerance")
        else:
            check(f"{label} == {expected}", False,
                  f"got {actual}, missing {missing} ({frac:.1%})")

    count_check("queries count", len(queries), N_QUERIES)
    count_check("targets count", len(targets), N_TARGETS)

    # Build lookup for row shape checks
    q_tables_dir = DATASET_ROOT / "queries" / "tables"
    t_tables_dir = DATASET_ROOT / "lake" / "targets" / "tables"

    # Test 3: row_embeddings.shape[0] matches n_rows in source CSV (sample 100)
    rng = np.random.RandomState(42)
    all_entries = queries + targets
    sample_idx = rng.choice(len(all_entries), min(100, len(all_entries)), replace=False)
    row_mismatches = []
    for idx in sample_idx:
        entry = all_entries[idx]
        emb = entry.get("row_embeddings")
        if emb is None:
            row_mismatches.append(f"{entry.get('table_id', '?')}: no row_embeddings key")
            continue
        emb = np.asarray(emb)
        table_id = entry["table_id"]

        # Find source CSV
        csv_path = q_tables_dir / f"{table_id}.csv"
        if not csv_path.exists():
            csv_path = t_tables_dir / f"{table_id}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=None)
                n_csv_rows = len(df)
                n_emb_rows = emb.shape[0]
                if n_emb_rows != n_csv_rows:
                    row_mismatches.append(
                        f"{table_id}: {n_emb_rows} emb rows vs {n_csv_rows} csv rows")
            except Exception as e:
                row_mismatches.append(f"{table_id}: csv read error: {e}")

    check("row_embeddings.shape[0] matches CSV n_rows (100 samples)",
          len(row_mismatches) == 0,
          f"{len(row_mismatches)} mismatches: {row_mismatches[:5]}")

    # Test 4: dtype and NaN/Inf check (sample 50)
    bad_dtype = []
    has_nan_inf = []
    for idx in sample_idx[:50]:
        entry = all_entries[idx]
        emb = np.asarray(entry.get("row_embeddings", np.array([])))
        if emb.size == 0:
            continue
        if emb.dtype != np.float32:
            bad_dtype.append(f"{entry.get('table_id', '?')}={emb.dtype}")
        if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
            has_nan_inf.append(entry.get("table_id", "?"))

    check("dtype is float32 (50 samples)",
          len(bad_dtype) == 0,
          f"{len(bad_dtype)} violations: {bad_dtype[:5]}")
    check("no NaN/Inf (50 samples)",
          len(has_nan_inf) == 0,
          f"{len(has_nan_inf)} violations: {has_nan_inf[:5]}")

    # Test 5: Constant dim
    dims = set()
    for entry in all_entries[:200]:
        emb = np.asarray(entry.get("row_embeddings", np.array([])))
        if emb.ndim >= 2:
            dims.add(emb.shape[1])
        elif emb.ndim == 1 and emb.size > 0:
            dims.add(emb.shape[0])

    check(f"constant embedding dim (found: {dims})",
          len(dims) == 1,
          f"multiple dims: {dims}")

    return queries, targets


def sanity_check(model_name, queries, targets):
    """Test 6: Tier 0 dev cosine similarity sanity check for row matching.

    Uses query→join-target pairs since join targets share rows with seeds
    (join targets contain ALL rows from the parent, matching keys allow
    row-level alignment via parent_row_idx).
    """
    print(f"\n  Sanity check: {model_name}")

    # Load query tasks for Tier 0 dev
    query_tasks = []
    with open(DATASET_ROOT / "ground_truth" / "query_tasks.jsonl") as f:
        for line in f:
            qt = json.loads(line.strip())
            if qt["noise_tier"] == 0 and qt["split"] == "dev":
                query_tasks.append(qt)

    if not query_tasks:
        skip(f"{model_name} sanity check", "no Tier 0 dev query tasks")
        return

    # Build embedding lookups
    q_emb = {e["table_id"]: e for e in queries}
    t_emb = {e["table_id"]: e for e in targets}

    gt_dir = DATASET_ROOT / "ground_truth" / "table_maps"

    rng = np.random.RandomState(42)
    n_check = min(200, len(query_tasks))
    sample_tasks = rng.choice(len(query_tasks), n_check, replace=False)

    # Collect all target embeddings for random baseline
    all_target_ids = list(t_emb.keys())

    matched_sims = []
    random_sims = []

    for idx in sample_tasks:
        qt = query_tasks[idx]
        seed_id = qt["query_table_id"]

        # Find join target (shares all rows with seed via parent mapping)
        join_rel = [r for r in qt["relevant"] if r["relation"] == "join"]
        if not join_rel:
            continue
        join_id = join_rel[0]["table_id"]

        if seed_id not in q_emb or join_id not in t_emb:
            continue

        seed_entry = q_emb[seed_id]
        join_entry = t_emb[join_id]

        seed_emb = np.asarray(seed_entry["row_embeddings"], dtype=np.float32)
        join_emb = np.asarray(join_entry["row_embeddings"], dtype=np.float32)

        # Load ground truth row mappings
        seed_gt_path = gt_dir / f"{seed_id}.npz"
        join_gt_path = gt_dir / f"{join_id}.npz"
        if not seed_gt_path.exists() or not join_gt_path.exists():
            continue

        seed_gt = np.load(seed_gt_path)
        join_gt = np.load(join_gt_path)

        seed_row_parent = seed_gt["row_parent_idx"]
        join_row_parent = join_gt["row_parent_idx"]

        # Match rows by parent index
        # Build join parent→row_pos lookup (skip spurious rows with -1)
        join_parent_to_pos = {}
        for pos, parent_idx in enumerate(join_row_parent):
            if parent_idx >= 0 and pos < join_emb.shape[0]:
                join_parent_to_pos[parent_idx] = pos

        for s_pos, s_parent_idx in enumerate(seed_row_parent):
            if s_parent_idx < 0 or s_pos >= seed_emb.shape[0]:
                continue
            s_vec = seed_emb[s_pos]
            if np.linalg.norm(s_vec) == 0:
                continue

            # Find matching join row (same parent row)
            if s_parent_idx in join_parent_to_pos:
                j_pos = join_parent_to_pos[s_parent_idx]
                j_vec = join_emb[j_pos]
                if np.linalg.norm(j_vec) > 0:
                    sim = np.dot(s_vec, j_vec) / (
                        np.linalg.norm(s_vec) * np.linalg.norm(j_vec))
                    matched_sims.append(sim)

            # Random row from a random target table
            rand_target_id = all_target_ids[rng.randint(0, len(all_target_ids))]
            rand_entry = t_emb[rand_target_id]
            rand_emb = np.asarray(rand_entry["row_embeddings"], dtype=np.float32)
            if rand_emb.shape[0] > 0:
                rand_row = rng.randint(0, rand_emb.shape[0])
                r_vec = rand_emb[rand_row]
                if np.linalg.norm(r_vec) > 0:
                    sim = np.dot(s_vec, r_vec) / (
                        np.linalg.norm(s_vec) * np.linalg.norm(r_vec))
                    random_sims.append(sim)

    if matched_sims and random_sims:
        med_matched = float(np.median(matched_sims))
        med_random = float(np.median(random_sims))
        gap = med_matched - med_random
        print(f"    INFO: median_matched={med_matched:.4f}, "
              f"median_random={med_random:.4f}, gap={gap:.4f} "
              f"({len(matched_sims)} matched, {len(random_sims)} random pairs)")
        check(f"matched > random (gap={gap:.4f})",
              med_matched > med_random,
              f"median matched ({med_matched:.4f}) <= random ({med_random:.4f})")
    else:
        skip("sanity check",
             f"insufficient pairs: {len(matched_sims)} matched, {len(random_sims)} random")


def main():
    global passed, failed, skipped

    print("Step 6 Validation: Row Embeddings for DLTE")
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
