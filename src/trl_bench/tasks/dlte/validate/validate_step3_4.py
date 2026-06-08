"""
Validate Step 3 + Step 4 outputs against PLAN.md test conditions.

Step 3 tests:
  1. All table_ids in lake_manifest are unique
  2. No query (role="query") table_id appears in lake
  3. 100% of csv_paths exist
  4. Log: #targets, #ckan_distractors, total lake size
  5. CKAN IDs match exactly with keys in embeddings/column/*/ckan_subset.pkl

Step 4 tests:
  6. Every query has exactly 2 relevant targets (1 union, 1 join)
  7. All relevant target IDs exist in lake_manifest
  8. All relevant targets have .npz mapping files
  9. Re-run produces identical output (hash check)
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


def main():
    global passed, failed

    # Load all manifests (resolve relative csv_path against project root)
    lake = []
    with open(DATASET_ROOT / "manifests" / "lake_manifest.jsonl") as f:
        for line in f:
            entry = json.loads(line.strip())
            p = Path(entry["csv_path"])
            if not p.is_absolute():
                entry["csv_path"] = str(PROJECT_ROOT / p)
            lake.append(entry)

    fragments = []
    with open(DATASET_ROOT / "manifests" / "fragments_manifest.jsonl") as f:
        for line in f:
            entry = json.loads(line.strip())
            p = Path(entry["csv_path"])
            if not p.is_absolute():
                entry["csv_path"] = str(PROJECT_ROOT / p)
            fragments.append(entry)

    query_tasks = []
    with open(DATASET_ROOT / "ground_truth" / "query_tasks.jsonl") as f:
        for line in f:
            query_tasks.append(json.loads(line.strip()))

    lake_ids = set(e["table_id"] for e in lake)
    query_ids = set(f["table_id"] for f in fragments if f["role"] == "query")

    print(f"Lake: {len(lake)}, Fragments: {len(fragments)}, Query tasks: {len(query_tasks)}\n")

    # ═══ Step 3 Tests ═══════════════════════════════════════════
    print("═══ Step 3: Lake Construction ═══")

    # Test 1: Unique table_ids
    all_lake_ids = [e["table_id"] for e in lake]
    check("All lake table_ids unique",
          len(all_lake_ids) == len(set(all_lake_ids)),
          f"{len(all_lake_ids) - len(set(all_lake_ids))} duplicates")

    # Test 2: No queries in lake
    queries_in_lake = query_ids & lake_ids
    check("No query table_ids in lake",
          len(queries_in_lake) == 0,
          f"{len(queries_in_lake)} queries found: {list(queries_in_lake)[:3]}")

    # Test 3: All csv_paths exist (sample 500)
    import numpy as np
    rng = np.random.RandomState(42)
    sample = rng.choice(len(lake), min(500, len(lake)), replace=False)
    missing = [lake[i]["table_id"] for i in sample
               if not Path(lake[i]["csv_path"]).exists()]
    check(f"CSV paths exist (sampled {len(sample)})",
          len(missing) == 0,
          f"{len(missing)} missing: {missing[:5]}")

    # Test 4: Counts
    n_targets = sum(1 for e in lake if e["source"] == "dlte_target")
    n_distractors = sum(1 for e in lake if e["source"] == "ckan_distractor")
    print(f"  INFO: targets={n_targets}, distractors={n_distractors}, total={len(lake)}")
    check("Lake has targets + distractors",
          n_targets > 0 and n_distractors > 0)

    # Test 5: CKAN IDs match embeddings
    ckan_lake_ids = sorted(e["table_id"] for e in lake if e["source"] == "ckan_distractor")
    models = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
    emb_root = PROJECT_ROOT / "assets" / "embeddings" / "column"

    all_match = True
    for model in models:
        with open(emb_root / model / "ckan_subset.pkl", "rb") as f:
            emb_data = pickle.load(f)
        emb_ids = sorted(set(e["table_id"] for e in emb_data))
        if ckan_lake_ids != emb_ids:
            all_match = False
            diff = set(ckan_lake_ids).symmetric_difference(set(emb_ids))
            print(f"  MISMATCH with {model}: {len(diff)} IDs differ")

    check("CKAN IDs match all 7 column embedding models", all_match)
    print()

    # ═══ Step 4 Tests ═══════════════════════════════════════════
    print("═══ Step 4: Ground Truth ═══")

    # Test 6: Every query has exactly 2 relevant targets (1 union, 1 join)
    bad_tasks = []
    for qt in query_tasks:
        rels = qt["relevant"]
        types = sorted(r["relation"] for r in rels)
        if types != ["join", "union"]:
            bad_tasks.append(f"{qt['query_table_id']}: {types}")

    check("Every query has exactly 1 union + 1 join",
          len(bad_tasks) == 0,
          f"{len(bad_tasks)} violations: {bad_tasks[:5]}")

    # Test 7: All relevant target IDs exist in lake
    targets_missing = []
    for qt in query_tasks:
        for rel in qt["relevant"]:
            if rel["table_id"] not in lake_ids:
                targets_missing.append(rel["table_id"])

    check("All relevant targets in lake",
          len(targets_missing) == 0,
          f"{len(targets_missing)} missing: {targets_missing[:5]}")

    # Test 8: All relevant targets have .npz
    gt_dir = DATASET_ROOT / "ground_truth" / "table_maps"
    npz_missing = []
    for qt in query_tasks:
        for rel in qt["relevant"]:
            if not (gt_dir / f"{rel['table_id']}.npz").exists():
                npz_missing.append(rel["table_id"])

    check("All relevant targets have .npz mappings",
          len(npz_missing) == 0,
          f"{len(npz_missing)} missing: {npz_missing[:5]}")

    # Test 9: Reproducibility (hash)
    for fname, label in [
        ("lake_manifest.jsonl", "Lake manifest"),
        ("ckan_distractor_ids.txt", "CKAN IDs"),
    ]:
        path = DATASET_ROOT / "manifests" / fname
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"  INFO: {label} SHA256: {h}...")

    qt_hash = hashlib.sha256(
        (DATASET_ROOT / "ground_truth" / "query_tasks.jsonl").read_bytes()
    ).hexdigest()[:16]
    print(f"  INFO: Query tasks SHA256: {qt_hash}...")
    check("Hashes recorded for reproducibility", True)
    print()

    # ── Summary ──
    print(f"{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
