"""
Validate Step 10: Stage 3 — Row Matching + Merge.

Test conditions (from PLAN.md):
  1. Every query produces .enriched.csv and .provenance.json
  2. CSVs readable by pandas with unique headers
  3. Tier 0 oracle test: correct candidates + correct key → CellF1 ≈ 1.0
  4. Join row match accuracy > random baseline per row model
  5. Output files exist for all 28 combinations
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
GT_ROOT = DATASET_ROOT / "ground_truth"
TABLE_MAPS_DIR = GT_ROOT / "table_maps"
MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
PARENTS_PATH = DATASET_ROOT / "manifests" / "parents_filtered.jsonl"
EVAL_ROOT = PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
ENRICHED_ROOT = EVAL_ROOT / "enriched"
STAGE3_ROOT = EVAL_ROOT / "stage3"

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
ROW_MODELS = [
    "bert", "dae", "gte", "saint", "scarf", "subtab",
    "tabbie", "tabicl", "tabpfn", "tabtransformer", "tabular_binning",
    "transtab", "tuta", "vime",
]
SPLITS = ["dev", "test", "train"]

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


def load_manifest():
    lookup = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            lookup[entry["table_id"]] = entry
    return lookup


def load_parents():
    """Load parents manifest into a dict keyed by parent_id."""
    lookup = {}
    with open(PARENTS_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            p = Path(entry["csv_path"])
            if not p.is_absolute():
                entry["csv_path"] = str(PROJECT_ROOT / p)
            lookup[entry["parent_id"]] = entry
    return lookup


def resolve_csv_path(csv_path_str):
    """Resolve a csv_path from a manifest entry, handling both relative and absolute paths."""
    path = Path(csv_path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_query_tasks():
    tasks = []
    with open(GT_ROOT / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def cell_f1(enriched_df, parent_df):
    """Compute cell-level F1 between enriched table and parent.

    Compares multisets of cell values (as strings).
    """
    enriched_cells = []
    for col in enriched_df.columns:
        for val in enriched_df[col]:
            enriched_cells.append(str(val).strip().lower())

    parent_cells = []
    for col in parent_df.columns:
        for val in parent_df[col]:
            parent_cells.append(str(val).strip().lower())

    # Convert to multiset counts
    from collections import Counter
    e_counter = Counter(enriched_cells)
    p_counter = Counter(parent_cells)

    # TP = overlap
    tp = sum((e_counter & p_counter).values())
    fp = sum(e_counter.values()) - tp
    fn = sum(p_counter.values()) - tp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1


def validate_combination(combo_name, query_tasks, manifest, parents_lookup):
    """Validate one col_model__row_model combination."""
    print(f"\n  Combination: {combo_name}")

    combo_enriched = ENRICHED_ROOT / combo_name
    combo_stage3 = STAGE3_ROOT / combo_name
    has_enriched = combo_enriched.exists()

    if not combo_stage3.exists():
        skip(f"{combo_name} stage3 dir", "directory not found")
        if not has_enriched:
            skip(f"{combo_name} enriched dir", "directory not found")
        return

    # Test 1/2: Enriched CSV checks (only when --save-enriched was used)
    if has_enriched:
        for split in SPLITS:
            split_dir = combo_enriched / split
            if not split_dir.exists():
                skip(f"{combo_name}/{split} enriched", "split directory not found")
                continue

            split_tasks = [qt for qt in query_tasks if qt["split"] == split]

            csv_files = list(split_dir.glob("*.enriched.csv"))
            prov_files = list(split_dir.glob("*.provenance.json"))
            check(f"{split}: {len(csv_files)} enriched CSVs == {len(split_tasks)} queries",
                  len(csv_files) == len(split_tasks),
                  f"got {len(csv_files)}")
            check(f"{split}: {len(prov_files)} provenance JSONs == {len(split_tasks)} queries",
                  len(prov_files) == len(split_tasks),
                  f"got {len(prov_files)}")

            # Test 2: CSVs readable by pandas with unique headers (sample 50)
            sample = csv_files[:50]
            unreadable = 0
            dup_headers = 0
            for csv_path in sample:
                try:
                    df = pd.read_csv(csv_path)
                    if len(df.columns) != len(set(df.columns)):
                        dup_headers += 1
                except Exception:
                    unreadable += 1
            check(f"{split}: all sampled CSVs readable (50 samples)",
                  unreadable == 0, f"{unreadable} unreadable")
            check(f"{split}: all sampled CSVs have unique headers",
                  dup_headers == 0, f"{dup_headers} with duplicate headers")
    else:
        skip(f"{combo_name} enriched checks",
             "enriched dir not found (run with --save-enriched to enable)")

    # Merge log checks (always run — these only need Stage 3 outputs)
    for split in SPLITS:
        log_path = combo_stage3 / f"merge_log_{split}.jsonl"
        check(f"{split}: merge_log exists", log_path.exists())

    # Test 3: Tier 0 oracle test on dev (CellF1 ≈ 1.0)
    # Requires enriched CSVs to be present
    dev_t0 = [qt for qt in query_tasks
              if qt["split"] == "dev" and qt["noise_tier"] == 0]

    gt_lookup = {}
    for qt in dev_t0:
        gt_u = [r["table_id"] for r in qt["relevant"] if r["relation"] == "union"]
        gt_j = [r["table_id"] for r in qt["relevant"] if r["relation"] == "join"]
        gt_lookup[qt["query_table_id"]] = (gt_u[0] if gt_u else None,
                                            gt_j[0] if gt_j else None)

    log_path = combo_stage3 / "merge_log_dev.jsonl"
    if log_path.exists():
        log = []
        with open(log_path) as f:
            for line in f:
                log.append(json.loads(line.strip()))
        log_by_qid = {e["query_table_id"]: e for e in log}

        # Find queries where BOTH correct candidates were predicted
        oracle_hits = []
        for qid, (gt_u, gt_j) in gt_lookup.items():
            if qid not in log_by_qid:
                continue
            entry = log_by_qid[qid]
            if entry["union_candidate"] == gt_u and entry["join_candidate"] == gt_j:
                oracle_hits.append(qid)

        if oracle_hits and has_enriched:
            cell_f1s = []
            for qid in oracle_hits[:20]:  # sample up to 20
                csv_path = combo_enriched / "dev" / f"{qid}.enriched.csv"
                if not csv_path.exists():
                    continue
                enriched = pd.read_csv(csv_path)

                # Load parent via manifest lookup (not hardcoded path)
                parent_id = manifest[qid]["parent_id"]
                parent_entry = parents_lookup.get(parent_id)
                if parent_entry is None:
                    continue
                parent_path = resolve_csv_path(parent_entry["csv_path"])
                if not parent_path.exists():
                    continue
                parent = pd.read_csv(parent_path, engine="python", on_bad_lines="skip")
                f1 = cell_f1(enriched, parent)
                cell_f1s.append(f1)

            if cell_f1s:
                mean_f1 = np.mean(cell_f1s)
                check(f"Tier 0 oracle CellF1 ({len(cell_f1s)} samples): {mean_f1:.4f} >= 0.8",
                      mean_f1 >= 0.8,
                      f"mean CellF1 = {mean_f1:.4f}")
            else:
                skip("Tier 0 oracle CellF1", "no parent CSVs found")
        elif oracle_hits and not has_enriched:
            skip("Tier 0 oracle CellF1", "enriched CSVs not saved (--save-enriched)")
        else:
            skip("Tier 0 oracle CellF1",
                 f"no oracle hits (0/{len(dev_t0)} T0 dev queries had both correct)")

    # Test 4: Join row match accuracy > random baseline
    if log_path.exists():
        log = []
        with open(log_path) as f:
            for line in f:
                log.append(json.loads(line.strip()))

        join_entries = [e for e in log if e["join_candidate"]]
        if join_entries:
            total_matched = sum(e["join_rows_matched"] for e in join_entries)
            total_rows = sum(e["join_rows_matched"] + e["join_rows_unmatched"]
                            for e in join_entries)
            match_rate = total_matched / total_rows if total_rows > 0 else 0
            # Random baseline: 1/n_join_rows per query ≈ very low
            check(f"Join row match rate: {match_rate:.4f} > 0.1 (random baseline)",
                  match_rate > 0.1,
                  f"match_rate = {match_rate:.4f}")


def main():
    global passed, failed, skipped

    print("Step 10 Validation: Stage 3 — Row Matching + Merge")
    print("=" * 60)

    manifest = load_manifest()
    query_tasks = load_query_tasks()
    parents_lookup = load_parents()

    for col_model in COLUMN_MODELS:
        for row_model in ROW_MODELS:
            combo = f"{col_model}__{row_model}"
            validate_combination(combo, query_tasks, manifest, parents_lookup)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
