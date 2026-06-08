"""
Step 5b: Split unified Starmie embeddings into per-split pickles.

After Starmie is pretrained on the combined corpus (dlte_v1_all),
this script splits the resulting pickle into the three files expected
by the downstream pipeline (steps 7, 8):
  - dlte_v1_queries.pkl  (query seed tables)
  - dlte_v1_targets.pkl  (union+join target fragments)
  - ckan_subset.pkl      (CKAN distractors)

The split is determined by table_id prefix:
  - dlte_v1__*__seed__*  → queries
  - dlte_v1__*__union__* or dlte_v1__*__join__*  → targets
  - everything else      → CKAN distractors

Usage:
    python downstream_tasks/dlte/scripts/step5b_split_starmie_embeddings.py
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = COL_EMB_ROOT = None
COMBINED_PATH = QUERIES_PATH = TARGETS_PATH = CKAN_PATH = None


def resolve_paths(args):
    global PROJECT_ROOT, COL_EMB_ROOT, COMBINED_PATH, QUERIES_PATH, TARGETS_PATH, CKAN_PATH
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    COL_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "column" / "starmie"
    COMBINED_PATH = COL_EMB_ROOT / "dlte_v1_all.pkl"
    QUERIES_PATH = COL_EMB_ROOT / "dlte_v1_queries.pkl"
    TARGETS_PATH = COL_EMB_ROOT / "dlte_v1_targets.pkl"
    CKAN_PATH = COL_EMB_ROOT / "ckan_subset.pkl"

N_EXPECTED_QUERIES = 5516
N_EXPECTED_TARGETS = 11032
N_EXPECTED_CKAN = 36740


def classify_entry(entry):
    """Classify a table entry as query, target, or CKAN based on table_id."""
    tid = entry.get("table_id", "")
    if "__seed__" in tid:
        return "query"
    elif "__union__" in tid or "__join__" in tid:
        return "target"
    else:
        return "ckan"


def main():
    parser = argparse.ArgumentParser(description="Step 5b: Split unified Starmie embeddings")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory (default: auto-detect)")
    args = parser.parse_args()
    resolve_paths(args)

    print("Step 5b: Split Unified Starmie Embeddings")
    print("=" * 60)

    if not COMBINED_PATH.exists():
        print(f"ERROR: Combined pickle not found: {COMBINED_PATH}")
        print("Run the starmie_dlte_v1_all SLURM job first.")
        return 1

    print(f"Loading {COMBINED_PATH}...")
    t0 = time.time()
    with open(COMBINED_PATH, "rb") as f:
        all_entries = pickle.load(f)
    print(f"Loaded {len(all_entries)} entries in {time.time() - t0:.1f}s")

    # Split
    queries = []
    targets = []
    ckan = []
    for entry in all_entries:
        cat = classify_entry(entry)
        if cat == "query":
            queries.append(entry)
        elif cat == "target":
            targets.append(entry)
        else:
            ckan.append(entry)

    print(f"\nSplit results:")
    print(f"  Queries: {len(queries)} (expected {N_EXPECTED_QUERIES})")
    print(f"  Targets: {len(targets)} (expected {N_EXPECTED_TARGETS})")
    print(f"  CKAN:    {len(ckan)} (expected {N_EXPECTED_CKAN})")
    print(f"  Total:   {len(queries) + len(targets) + len(ckan)}")

    # Validate counts
    ok = True
    if len(queries) != N_EXPECTED_QUERIES:
        print(f"  WARN: queries count mismatch ({len(queries)} vs {N_EXPECTED_QUERIES})")
        ok = False
    if len(targets) != N_EXPECTED_TARGETS:
        print(f"  WARN: targets count mismatch ({len(targets)} vs {N_EXPECTED_TARGETS})")
        ok = False
    if len(ckan) != N_EXPECTED_CKAN:
        print(f"  WARN: CKAN count mismatch ({len(ckan)} vs {N_EXPECTED_CKAN})")
        ok = False

    if ok:
        print("  All counts match!")

    # Backup existing files if present
    for path, name in [(QUERIES_PATH, "queries"), (TARGETS_PATH, "targets"), (CKAN_PATH, "ckan")]:
        if path.exists():
            backup = path.with_suffix(".pkl.bak_separate_pretrain")
            print(f"\n  Backing up existing {name}: {path.name} → {backup.name}")
            path.rename(backup)

    # Save split pickles
    print(f"\nSaving split pickles...")
    for entries, path, name in [
        (queries, QUERIES_PATH, "queries"),
        (targets, TARGETS_PATH, "targets"),
        (ckan, CKAN_PATH, "ckan"),
    ]:
        with open(path, "wb") as f:
            pickle.dump(entries, f)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: {path.name} ({len(entries)} entries, {size_mb:.1f} MB)")

    print(f"\n{'='*60}")
    print("Done. Re-run step 7 and step 8 for starmie to update results.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
