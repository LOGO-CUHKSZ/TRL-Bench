"""
Step 4: Ground Truth Generation for DLTE benchmark.

Creates query→relevant-target mappings for evaluation.

Inputs:
  - datasets/dlte_v1/manifests/fragments_manifest.jsonl
  - datasets/dlte_v1/manifests/lake_manifest.jsonl

Outputs:
  - datasets/dlte_v1/ground_truth/query_tasks.jsonl
  - datasets/dlte_v1/ground_truth/gt_validation.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Step 4: Generate query tasks GT")
    parser.add_argument("--project-root", type=str,
                        default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    dataset_root = project_root / "datasets" / "dlte_v1"

    # ── Load fragments manifest ──
    fragments = []
    with open(dataset_root / "manifests" / "fragments_manifest.jsonl") as f:
        for line in f:
            fragments.append(json.loads(line.strip()))

    # ── Load lake manifest (for validation) ──
    lake_ids = set()
    with open(dataset_root / "manifests" / "lake_manifest.jsonl") as f:
        for line in f:
            entry = json.loads(line.strip())
            lake_ids.add(entry["table_id"])

    print(f"Loaded {len(fragments)} fragments, {len(lake_ids)} lake table IDs")

    # ── Group fragments by (parent_id, tier, rep) ──
    groups = defaultdict(dict)
    for f in fragments:
        key = (f["parent_id"], f["noise_tier"], f["replicate_id"])
        groups[key][f["fragment_type"]] = f

    # ── Build query tasks ──
    query_tasks = []
    errors = []

    for (parent_id, tier, rep), role_map in sorted(groups.items()):
        if "seed" not in role_map:
            errors.append(f"Missing seed for {parent_id}, t{tier}, r{rep}")
            continue
        if "union" not in role_map:
            errors.append(f"Missing union for {parent_id}, t{tier}, r{rep}")
            continue
        if "join" not in role_map:
            errors.append(f"Missing join for {parent_id}, t{tier}, r{rep}")
            continue

        seed = role_map["seed"]
        union = role_map["union"]
        join = role_map["join"]

        query_tasks.append({
            "query_table_id": seed["table_id"],
            "parent_id": parent_id,
            "split": seed["split"],
            "noise_tier": tier,
            "replicate_id": rep,
            "relevant": [
                {"table_id": union["table_id"], "relation": "union"},
                {"table_id": join["table_id"], "relation": "join"},
            ],
        })

    if errors:
        print(f"ERRORS: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")

    # ── Validate against lake ──
    targets_not_in_lake = []
    for qt in query_tasks:
        for rel in qt["relevant"]:
            if rel["table_id"] not in lake_ids:
                targets_not_in_lake.append(rel["table_id"])

    queries_in_lake = [qt for qt in query_tasks if qt["query_table_id"] in lake_ids]

    # ── Write query_tasks.jsonl ──
    gt_dir = dataset_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    qt_path = gt_dir / "query_tasks.jsonl"
    with open(qt_path, "w") as f:
        for qt in query_tasks:
            f.write(json.dumps(qt) + "\n")

    # ── Write validation summary ──
    validation = {
        "n_query_tasks": len(query_tasks),
        "n_per_split": {},
        "n_per_tier": {},
        "n_errors": len(errors),
        "n_targets_not_in_lake": len(targets_not_in_lake),
        "n_queries_in_lake": len(queries_in_lake),
    }

    for split in ["train", "dev", "test"]:
        validation["n_per_split"][split] = sum(
            1 for qt in query_tasks if qt["split"] == split)
    for tier in range(4):
        validation["n_per_tier"][str(tier)] = sum(
            1 for qt in query_tasks if qt["noise_tier"] == tier)

    val_path = gt_dir / "gt_validation.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"GROUND TRUTH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total query tasks: {len(query_tasks)}")
    print(f"  Per split: {validation['n_per_split']}")
    print(f"  Per tier:  {validation['n_per_tier']}")
    print(f"  Targets missing from lake: {len(targets_not_in_lake)}")
    print(f"  Queries accidentally in lake: {len(queries_in_lake)}")
    print(f"\nWritten:")
    print(f"  {qt_path}")
    print(f"  {val_path}")


if __name__ == "__main__":
    main()
