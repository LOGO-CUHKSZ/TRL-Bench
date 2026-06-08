"""
Step 3: Lake Construction for DLTE benchmark.

Combines DLTE target fragments (union + join) with CKAN distractor tables
into a single lake manifest.

Inputs:
  - datasets/dlte_v1/manifests/fragments_manifest.jsonl
  - embeddings/column/bert/ckan_subset.pkl  (for CKAN table IDs)
  - datasets/ckan_subset/tables/*.csv

Outputs:
  - datasets/dlte_v1/manifests/ckan_distractor_ids.txt
  - datasets/dlte_v1/manifests/lake_manifest.jsonl
"""

import argparse
import json
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Step 3: Build DLTE data lake")
    parser.add_argument("--project-root", type=str,
                        default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    dataset_root = project_root / "datasets" / "dlte_v1"
    manifests_dir = dataset_root / "manifests"

    # ── Load fragments manifest ──
    fragments = []
    with open(manifests_dir / "fragments_manifest.jsonl") as f:
        for line in f:
            fragments.append(json.loads(line.strip()))

    # Filter to lake entries (union + join targets)
    targets = [f for f in fragments if f["role"] == "lake"]
    print(f"Loaded {len(fragments)} fragments, {len(targets)} lake targets")

    # ── Get CKAN table IDs from embeddings ──
    emb_path = project_root / "assets" / "embeddings" / "column" / "bert" / "ckan_subset.pkl"
    with open(emb_path, "rb") as f:
        ckan_emb = pickle.load(f)
    ckan_ids = sorted(set(e["table_id"] for e in ckan_emb))
    print(f"CKAN embedding IDs: {len(ckan_ids)}")

    # Verify CSV files exist
    ckan_tables_dir = project_root / "datasets" / "ckan_subset" / "tables"
    missing = [tid for tid in ckan_ids
               if not (ckan_tables_dir / f"{tid}.csv").exists()]
    if missing:
        print(f"WARNING: {len(missing)} CKAN IDs missing CSV files: {missing[:5]}")
    else:
        print(f"All {len(ckan_ids)} CKAN CSVs verified")

    # ── Build lake manifest ──
    lake_entries = []

    # Add target fragments
    for t in targets:
        lake_entries.append({
            "table_id": t["table_id"],
            "source": "dlte_target",
            "parent_id": t["parent_id"],
            "fragment_type": t["fragment_type"],
            "noise_tier": t["noise_tier"],
            "replicate_id": t["replicate_id"],
            "split": t["split"],
            "csv_path": t["csv_path"],
            "n_rows": t["n_rows"],
            "n_cols": t["n_cols"],
        })

    # Add CKAN distractors
    for tid in ckan_ids:
        csv_path = str((ckan_tables_dir / f"{tid}.csv").relative_to(project_root))
        lake_entries.append({
            "table_id": tid,
            "source": "ckan_distractor",
            "parent_id": None,
            "fragment_type": None,
            "noise_tier": None,
            "replicate_id": None,
            "split": None,
            "csv_path": csv_path,
            "n_rows": None,  # not pre-computed for distractors
            "n_cols": None,
        })

    # ── Write outputs ──
    # CKAN distractor IDs
    ids_path = manifests_dir / "ckan_distractor_ids.txt"
    with open(ids_path, "w") as f:
        for tid in ckan_ids:
            f.write(tid + "\n")

    # Lake manifest
    lake_path = manifests_dir / "lake_manifest.jsonl"
    with open(lake_path, "w") as f:
        for entry in lake_entries:
            f.write(json.dumps(entry) + "\n")

    # ── Summary ──
    n_targets = sum(1 for e in lake_entries if e["source"] == "dlte_target")
    n_distractors = sum(1 for e in lake_entries if e["source"] == "ckan_distractor")
    print(f"\n{'='*60}")
    print(f"LAKE CONSTRUCTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total lake size: {len(lake_entries)}")
    print(f"  DLTE targets:     {n_targets}")
    print(f"    union: {sum(1 for e in lake_entries if e.get('fragment_type') == 'union')}")
    print(f"    join:  {sum(1 for e in lake_entries if e.get('fragment_type') == 'join')}")
    print(f"  CKAN distractors: {n_distractors}")
    print(f"\nWritten:")
    print(f"  {ids_path}")
    print(f"  {lake_path}")

    # Uniqueness check
    all_ids = [e["table_id"] for e in lake_entries]
    unique_ids = set(all_ids)
    if len(all_ids) != len(unique_ids):
        dupes = len(all_ids) - len(unique_ids)
        print(f"\nWARNING: {dupes} duplicate table_ids in lake!")
    else:
        print(f"\nAll {len(all_ids)} table_ids are unique")


if __name__ == "__main__":
    main()
