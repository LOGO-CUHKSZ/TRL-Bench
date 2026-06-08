"""
Step 11: Evaluation Harness — Unified metrics across all stages.

Computes CellF1 (headline metric), region decomposition, ParentRowRecall,
ParentColRecall, and consolidates Stage 1/2/3 metrics into a single summary.

Usage:
    python downstream_tasks/dlte/scripts/step11_evaluation.py
    python downstream_tasks/dlte/scripts/step11_evaluation.py --col_models bert --row_models tabicl --splits dev
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths (resolved at runtime by resolve_paths()) ───────────────

PROJECT_ROOT = DATA_ROOT = DATASET_ROOT = GT_ROOT = TABLE_MAPS_DIR = None
MANIFEST_PATH = PARENTS_PATH = None
STAGE1_ROOT = STAGE2_ROOT = STAGE3_ROOT = ENRICHED_ROOT = METRICS_ROOT = None

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
ROW_MODELS = [
    "bert", "dae", "gte", "saint", "scarf", "subtab",
    "tabbie", "tabicl", "tabpfn", "tabtransformer", "tabular_binning",
    "transtab", "tuta", "vime",
]


def derive_stage2_key(table_model, col_model):
    """Derive the Stage 2 directory key from table and column model names."""
    if table_model and table_model != col_model:
        return f"{table_model}__{col_model}"
    return col_model


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, DATA_ROOT, DATASET_ROOT, GT_ROOT, TABLE_MAPS_DIR
    global MANIFEST_PATH, PARENTS_PATH
    global STAGE1_ROOT, STAGE2_ROOT, STAGE3_ROOT, ENRICHED_ROOT, METRICS_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    DATA_ROOT = Path(args.data_root) if getattr(args, 'data_root', None) else PROJECT_ROOT
    DATASET_ROOT = DATA_ROOT / "datasets" / "dlte_v1"
    GT_ROOT = DATASET_ROOT / "ground_truth"
    TABLE_MAPS_DIR = GT_ROOT / "table_maps"
    MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
    PARENTS_PATH = DATASET_ROOT / "manifests" / "parents_filtered.jsonl"
    STAGE1_ROOT = output_root / "stage1"
    STAGE2_ROOT = output_root / "stage2"
    STAGE3_ROOT = output_root / "stage3"
    ENRICHED_ROOT = output_root / "enriched"
    METRICS_ROOT = output_root / "metrics"


# ── Data Loading ───────────────────────────────────────────────────

def load_query_tasks():
    tasks = []
    with open(GT_ROOT / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def _resolve_csv_path(entry):
    """Resolve relative csv_path entries against DATA_ROOT (in-place)."""
    p = Path(entry["csv_path"])
    if not p.is_absolute():
        entry["csv_path"] = str(DATA_ROOT / p)
    return entry


def load_manifest():
    lookup = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = _resolve_csv_path(json.loads(line.strip()))
            lookup[entry["table_id"]] = entry
    return lookup


def load_parents():
    """Load parent table lookup: parent_id -> entry with csv_path."""
    lookup = {}
    with open(PARENTS_PATH) as f:
        for line in f:
            entry = _resolve_csv_path(json.loads(line.strip()))
            lookup[entry["parent_id"]] = entry
    return lookup


def load_parent_csv(parent_entry):
    """Load a parent CSV, handling both relative and absolute paths."""
    csv_path = Path(parent_entry["csv_path"])
    if not csv_path.is_absolute():
        csv_path = DATA_ROOT / csv_path
    return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")


# ── CellF1 ─────────────────────────────────────────────────────────

def normalize_cell(val):
    """Normalize a cell value for comparison."""
    s = str(val).strip().lower()
    if s == "nan" or s == "none" or s == "":
        return None
    return s


def cell_f1(enriched_df, parent_df):
    """Multiset F1 over normalized cell values."""
    e_cells = Counter()
    for col in enriched_df.columns:
        for val in enriched_df[col]:
            nv = normalize_cell(val)
            if nv is not None:
                e_cells[nv] += 1

    p_cells = Counter()
    for col in parent_df.columns:
        for val in parent_df[col]:
            nv = normalize_cell(val)
            if nv is not None:
                p_cells[nv] += 1

    tp = sum((e_cells & p_cells).values())
    n_enriched = sum(e_cells.values())
    n_parent = sum(p_cells.values())

    prec = tp / n_enriched if n_enriched > 0 else 0
    rec = tp / n_parent if n_parent > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp,
            "n_enriched": n_enriched, "n_parent": n_parent}


# ── Region Decomposition ──────────────────────────────────────────

def region_recall(enriched_df, parent_df, seed_npz, gt_union_id, gt_join_id):
    """Compute recall in each of the 4 CellF1 regions.

    Regions are defined by which parent cells fall in which quadrant:
      - core_core: parent[seed_rows, seed_cols] — already in seed
      - union_region: parent[union_rows, seed_cols] — new rows, same cols
      - join_region: parent[seed_rows, join_cols] — same rows, new cols
      - hard_region: parent[union_rows, join_cols] — new rows AND new cols

    Note: The four region recalls are **independent** metrics, not additive
    components of total recall.  The enriched-cell multiset (``e_cells``) is
    shared across regions without deduction, so the same enriched cell can
    satisfy multiple regions.  This is intentional — each region recall answers
    "what fraction of *this region's* cells appear in the enriched table?"
    independently.
    """
    # Core rows/cols from seed
    seed_rows = set(int(x) for x in seed_npz["row_parent_idx"] if x >= 0)
    seed_cols = set(int(x) for x in seed_npz["col_parent_idx"] if x >= 0)

    # Union rows (missing from seed)
    union_rows = set()
    if gt_union_id:
        union_npz_path = TABLE_MAPS_DIR / f"{gt_union_id}.npz"
        if union_npz_path.exists():
            union_npz = np.load(union_npz_path)
            union_rows = set(int(x) for x in union_npz["row_parent_idx"] if x >= 0)

    # Join cols (missing from seed)
    join_cols = set()
    if gt_join_id:
        join_npz_path = TABLE_MAPS_DIR / f"{gt_join_id}.npz"
        if join_npz_path.exists():
            join_npz = np.load(join_npz_path)
            join_cols = set(int(x) for x in join_npz["col_parent_idx"] if x >= 0) - seed_cols

    # Build parent cell multisets per region
    p_cols = list(parent_df.columns)
    all_parent_cols = set(range(len(p_cols)))

    regions = {
        "core_core": [],
        "union_region": [],
        "join_region": [],
        "hard_region": [],
    }

    for ri in range(len(parent_df)):
        for ci in range(len(p_cols)):
            val = normalize_cell(parent_df.iloc[ri, ci])
            if val is None:
                continue
            in_seed_row = ri in seed_rows
            in_union_row = ri in union_rows
            in_seed_col = ci in seed_cols
            in_join_col = ci in join_cols

            if in_seed_row and in_seed_col:
                regions["core_core"].append(val)
            elif in_union_row and in_seed_col:
                regions["union_region"].append(val)
            elif in_seed_row and in_join_col:
                regions["join_region"].append(val)
            elif in_union_row and in_join_col:
                regions["hard_region"].append(val)
            # else: rows/cols from neither seed nor union/join (shouldn't happen at Tier 0)

    # Compare with enriched table's multiset
    e_cells = Counter()
    for col in enriched_df.columns:
        for val in enriched_df[col]:
            nv = normalize_cell(val)
            if nv is not None:
                e_cells[nv] += 1

    # Compute recall per region: what fraction of region cells appear in enriched?
    result = {}
    for region_name, region_cells in regions.items():
        if not region_cells:
            result[region_name] = {"recall": None, "n_cells": 0}
            continue
        region_counter = Counter(region_cells)
        # Recall: how many region cells are recovered in enriched
        recovered = sum((region_counter & e_cells).values())
        total = sum(region_counter.values())
        result[region_name] = {
            "recall": recovered / total if total > 0 else 0,
            "n_cells": total,
            "n_recovered": recovered,
        }

    return result


# ── Parent Row/Col Recall ─────────────────────────────────────────

def parent_row_recall(enriched_df, parent_df, parent_entry):
    """Fraction of parent rows whose key value appears in the enriched table."""
    key_col = parent_entry.get("key_col")
    if not key_col or key_col not in parent_df.columns:
        return None

    parent_keys = set()
    for v in parent_df[key_col]:
        nv = normalize_cell(v)
        if nv is not None:
            parent_keys.add(nv)

    if not parent_keys:
        return None

    # Check enriched table's key column only (not all columns — checking all
    # columns inflates recall when key values coincidentally appear elsewhere)
    if key_col not in enriched_df.columns:
        return None
    enriched_vals = set()
    for v in enriched_df[key_col]:
        nv = normalize_cell(v)
        if nv is not None:
            enriched_vals.add(nv)

    recovered = len(parent_keys & enriched_vals)
    return recovered / len(parent_keys)


def parent_col_recall(enriched_df, parent_df):
    """Fraction of parent columns whose name appears in the enriched table (normalized)."""
    parent_cols = {c.strip().lower() for c in parent_df.columns}
    enriched_cols = {c.strip().lower() for c in enriched_df.columns}
    if not parent_cols:
        return None
    return len(parent_cols & enriched_cols) / len(parent_cols)


# ── Per-Query Evaluation ──────────────────────────────────────────

def evaluate_query(qid, qt, enriched_path, parent_df, parent_entry, manifest):
    """Evaluate one query: CellF1, region decomposition, row/col recall."""
    if not enriched_path.exists():
        return None

    enriched_df = pd.read_csv(enriched_path)
    result = {}

    # CellF1
    cf1 = cell_f1(enriched_df, parent_df)
    result["cell_f1"] = cf1["f1"]
    result["cell_precision"] = cf1["precision"]
    result["cell_recall"] = cf1["recall"]

    # Parent row/col recall
    result["parent_row_recall"] = parent_row_recall(enriched_df, parent_df, parent_entry)
    result["parent_col_recall"] = parent_col_recall(enriched_df, parent_df)

    # Region decomposition
    seed_npz_path = TABLE_MAPS_DIR / f"{qid}.npz"
    if seed_npz_path.exists():
        seed_npz = np.load(seed_npz_path)
        gt_union = None
        gt_join = None
        for rel in qt.get("relevant", []):
            if rel["relation"] == "union":
                gt_union = rel["table_id"]
            elif rel["relation"] == "join":
                gt_join = rel["table_id"]

        regions = region_recall(enriched_df, parent_df, seed_npz, gt_union, gt_join)
        result["region_recall"] = {
            k: v["recall"] for k, v in regions.items()
        }
        result["region_cells"] = {
            k: v["n_cells"] for k, v in regions.items()
        }
    else:
        result["region_recall"] = None

    # Enriched shape
    result["enriched_shape"] = list(enriched_df.shape)
    result["parent_shape"] = list(parent_df.shape)

    return result


# ── Per-Combination Evaluation ────────────────────────────────────

def evaluate_combination(col_model, row_model, query_tasks, parents, manifest, splits,
                         table_model=None):
    """Evaluate one col_model x row_model combination across splits."""
    stage2_key = derive_stage2_key(table_model, col_model)
    combo = f"{stage2_key}__{row_model}"
    s1_model = table_model if table_model else col_model
    print(f"\n  Combination: {combo}")
    t0 = time.time()

    # Build query task lookup
    qt_by_qid = {qt["query_table_id"]: qt for qt in query_tasks}

    # Output directory
    metrics_dir = METRICS_ROOT / combo
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Consolidate Stage 1 metrics ──
    stage1_data = {}
    for split in splits:
        for k in [10, 50, 100]:
            path = STAGE1_ROOT / s1_model / f"metrics_{split}_topk_{k}.json"
            if path.exists():
                stage1_data[f"{split}_topk_{k}"] = json.loads(path.read_text())
    with open(metrics_dir / "stage1.json", "w") as f:
        json.dump({"table_model": s1_model, "col_model": col_model,
                    "metrics": stage1_data}, f, indent=2)

    # ── Consolidate Stage 2 metrics ──
    stage2_data = {}
    for split in splits:
        path = STAGE2_ROOT / stage2_key / f"metrics_{split}_topk_100.json"
        if path.exists():
            stage2_data[split] = json.loads(path.read_text())
    cal_path = STAGE2_ROOT / stage2_key / "calibration_dev.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    with open(metrics_dir / "stage2.json", "w") as f:
        json.dump({"col_model": col_model, "calibration": cal, "metrics": stage2_data}, f, indent=2)

    # ── End-to-end evaluation ──
    end_to_end = {"col_model": col_model, "row_model": row_model,
                  "table_model": s1_model, "splits": {}}

    for split in splits:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue

        print(f"    Split: {split} ({len(split_tasks)} queries)")
        enriched_dir = ENRICHED_ROOT / combo / split

        # Per-query results
        all_results = []
        tier_results = defaultdict(list)
        n_processed = 0

        for qt in split_tasks:
            qid = qt["query_table_id"]
            parent_id = qt["parent_id"]
            tier = qt["noise_tier"]

            parent_entry = parents.get(parent_id)
            if parent_entry is None:
                continue

            parent_csv_path = Path(parent_entry["csv_path"])
            if not parent_csv_path.exists():
                continue

            parent_df = load_parent_csv(parent_entry)
            enriched_path = enriched_dir / f"{qid}.enriched.csv"

            result = evaluate_query(qid, qt, enriched_path, parent_df, parent_entry, manifest)
            if result is None:
                continue

            result["tier"] = tier
            all_results.append(result)
            tier_results[tier].append(result)
            n_processed += 1

            if n_processed % 500 == 0:
                print(f"      {n_processed}/{len(split_tasks)} queries evaluated...")

        if not all_results:
            continue

        # Aggregate
        def agg(results, field):
            vals = [r[field] for r in results if r.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        def agg_regions(results):
            region_names = ["core_core", "union_region", "join_region", "hard_region"]
            out = {}
            for rn in region_names:
                vals = [r["region_recall"][rn] for r in results
                        if r.get("region_recall") and r["region_recall"].get(rn) is not None]
                out[rn] = float(np.mean(vals)) if vals else None
            return out

        split_metrics = {
            "n_queries": len(split_tasks),
            "n_evaluated": n_processed,
            "cell_f1": agg(all_results, "cell_f1"),
            "cell_precision": agg(all_results, "cell_precision"),
            "cell_recall": agg(all_results, "cell_recall"),
            "parent_row_recall": agg(all_results, "parent_row_recall"),
            "parent_col_recall": agg(all_results, "parent_col_recall"),
            "region_recall": agg_regions(all_results),
            "per_tier": {},
        }

        for tier in sorted(tier_results.keys()):
            t_results = tier_results[tier]
            split_metrics["per_tier"][tier] = {
                "n_queries": len(t_results),
                "cell_f1": agg(t_results, "cell_f1"),
                "cell_precision": agg(t_results, "cell_precision"),
                "cell_recall": agg(t_results, "cell_recall"),
                "parent_row_recall": agg(t_results, "parent_row_recall"),
                "parent_col_recall": agg(t_results, "parent_col_recall"),
                "region_recall": agg_regions(t_results),
            }

        end_to_end["splits"][split] = split_metrics

        # Print summary
        _f = lambda v: f"{v:.4f}" if v is not None else "N/A"
        print(f"      CellF1={_f(split_metrics['cell_f1'])}  "
              f"RowRecall={_f(split_metrics['parent_row_recall'])}  "
              f"ColRecall={_f(split_metrics['parent_col_recall'])}")
        rr = split_metrics["region_recall"]
        print(f"      Regions: core={rr.get('core_core','N/A'):.4f}  "
              f"union={rr.get('union_region','N/A'):.4f}  "
              f"join={rr.get('join_region','N/A'):.4f}  "
              f"hard={rr.get('hard_region','N/A'):.4f}"
              if all(rr.get(k) is not None for k in ["core_core", "union_region", "join_region", "hard_region"])
              else f"      Regions: {rr}")
        for tier in sorted(split_metrics["per_tier"].keys()):
            tm = split_metrics["per_tier"][tier]
            print(f"      Tier {tier}: CellF1={_f(tm['cell_f1'])}  "
                  f"RowR={_f(tm['parent_row_recall'])}  "
                  f"ColR={_f(tm['parent_col_recall'])}")

    with open(metrics_dir / "end_to_end.json", "w") as f:
        json.dump(end_to_end, f, indent=2)

    # ── Summary CSV (one row) ──
    summary = {"col_model": col_model, "row_model": row_model, "table_model": s1_model}
    for split in splits:
        sm = end_to_end["splits"].get(split, {})
        summary[f"cell_f1_{split}"] = sm.get("cell_f1")
        summary[f"parent_row_recall_{split}"] = sm.get("parent_row_recall")
        summary[f"parent_col_recall_{split}"] = sm.get("parent_col_recall")

    # Add Stage 1 headline
    s1_dev = stage1_data.get("dev_topk_10", {})
    summary["recall_any_10_dev"] = s1_dev.get("recall_any")
    summary["recall_any_100_dev"] = stage1_data.get("dev_topk_100", {}).get("recall_any")

    # Add Stage 2 headline
    s2_dev = stage2_data.get("dev", {})
    summary["relation_acc_dev"] = s2_dev.get("relation_acc")

    pd.DataFrame([summary]).to_csv(metrics_dir / "summary.csv", index=False)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return summary


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 11: Evaluation Harness")
    parser.add_argument("--col_models", nargs="+", default=COLUMN_MODELS)
    parser.add_argument("--row_models", nargs="+", default=ROW_MODELS)
    parser.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root for DLTE outputs (default: {project_root}/results/evaluation/dlte)")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root (default: derived from script location)")
    parser.add_argument("--table_model", type=str, default=None,
                        help="Table model for Stage 1 retrieval (default: same as --col_models)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Data root containing 'datasets/dlte_v1/' (default: {project_root})")
    args = parser.parse_args()

    resolve_paths(args)

    print("Step 11: Evaluation Harness")
    print("=" * 60)
    print(f"Column models: {args.col_models}")
    print(f"Row models: {args.row_models}")
    print(f"Splits: {args.splits}")

    # Load shared data
    print("\nLoading shared data...")
    t_load = time.time()
    query_tasks = load_query_tasks()
    manifest = load_manifest()
    parents = load_parents()
    print(f"  {len(query_tasks)} query tasks, {len(manifest)} manifest entries, "
          f"{len(parents)} parents in {time.time() - t_load:.1f}s")

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for col_model in args.col_models:
        for row_model in args.row_models:
            try:
                summary = evaluate_combination(
                    col_model, row_model, query_tasks, parents, manifest, args.splits,
                    table_model=args.table_model)
                if summary:
                    all_summaries.append(summary)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Global summary CSV
    if all_summaries:
        global_df = pd.DataFrame(all_summaries)
        global_path = METRICS_ROOT / "all_combinations_summary.csv"
        global_df.to_csv(global_path, index=False)
        print(f"\nGlobal summary: {global_path}")

        # Print CellF1 leaderboard
        _fmt = lambda v: f"{v:>8.4f}" if pd.notna(v) else "     N/A"
        if "cell_f1_dev" in global_df.columns:
            has_table_model = "table_model" in global_df.columns and (global_df["table_model"] != global_df["col_model"]).any()
            print(f"\n{'='*70}")
            print("CellF1 Leaderboard (dev split)")
            print(f"{'='*70}")
            if has_table_model:
                print(f"{'Table Model':<14} {'Col Model':<14} {'Row Model':<10} {'CellF1':>8} {'RowR':>8} {'ColR':>8}")
                print("-" * 64)
            else:
                print(f"{'Col Model':<14} {'Row Model':<10} {'CellF1':>8} {'RowR':>8} {'ColR':>8}")
                print("-" * 50)
            for _, row in global_df.sort_values("cell_f1_dev", ascending=False, na_position='last').iterrows():
                if has_table_model:
                    print(f"{row['table_model']:<14} {row['col_model']:<14} {row['row_model']:<10} "
                          f"{_fmt(row['cell_f1_dev'])} "
                          f"{_fmt(row['parent_row_recall_dev'])} "
                          f"{_fmt(row['parent_col_recall_dev'])}")
                else:
                    print(f"{row['col_model']:<14} {row['row_model']:<10} "
                          f"{_fmt(row['cell_f1_dev'])} "
                          f"{_fmt(row['parent_row_recall_dev'])} "
                          f"{_fmt(row['parent_col_recall_dev'])}")

    print(f"\n{'='*60}")
    print(f"Processed {len(all_summaries)}/{len(args.col_models) * len(args.row_models)} combinations")
    print(f"Metrics: {METRICS_ROOT}")
    print(f"{'='*60}")

    return 0 if len(all_summaries) == len(args.col_models) * len(args.row_models) else 1


if __name__ == "__main__":
    sys.exit(main())
