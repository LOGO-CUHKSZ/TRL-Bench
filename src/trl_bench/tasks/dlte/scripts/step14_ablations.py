"""
Step 14: Ablations + Baselines.

Ablation studies:
  A1. Noise tier breakdown (from existing Step 11/13 metrics)
  A2. K sensitivity (from existing Step 12 data)
  A3. Stage-wise oracle bounds:
      - Oracle S1: GT candidates + predicted alignment + predicted row matching (Step 13)
      - Oracle S1+S2: GT candidates + GT column alignment + predicted row matching
      - Oracle S1+S2+S3: GT candidates + GT column alignment + GT row mapping (upper bound)

Baselines:
  B1. Stage 1: TF-IDF over column headers
  B2. Stage 2: Token Jaccard greedy column matching
  B3. Stage 3: Exact match on normalized key strings (already what Step 10 does)

Usage:
    python downstream_tasks/dlte/scripts/step14_ablations.py --col_model bert --row_model tabicl
    python downstream_tasks/dlte/scripts/step14_ablations.py --aggregate
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths (resolved at runtime by resolve_paths()) ───────────────

PROJECT_ROOT = DATASET_ROOT = GT_ROOT = TABLE_MAPS_DIR = None
MANIFEST_PATH = PARENTS_PATH = ROW_EMB_ROOT = None
RESULTS_ROOT = METRICS_ROOT = HEATMAP_ROOT = ABLATION_ROOT = None

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
ROW_MODELS = [
    "bert", "dae", "gte", "saint", "scarf", "subtab",
    "tabbie", "tabicl", "tabpfn", "tabtransformer", "tabular_binning",
    "transtab", "tuta", "vime",
]

ROW_SIM_THRESHOLD = 0.80


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, DATASET_ROOT, GT_ROOT, TABLE_MAPS_DIR
    global MANIFEST_PATH, PARENTS_PATH, ROW_EMB_ROOT
    global RESULTS_ROOT, METRICS_ROOT, HEATMAP_ROOT, ABLATION_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
    GT_ROOT = DATASET_ROOT / "ground_truth"
    TABLE_MAPS_DIR = GT_ROOT / "table_maps"
    MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
    PARENTS_PATH = DATASET_ROOT / "manifests" / "parents_filtered.jsonl"
    ROW_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "row"
    RESULTS_ROOT = output_root
    METRICS_ROOT = output_root / "metrics"
    HEATMAP_ROOT = output_root / "experiments" / "heatmap_oracle_stage1"
    ABLATION_ROOT = output_root / "experiments" / "ablations"


# ── Data Loading ───────────────────────────────────────────────────

def load_query_tasks():
    tasks = []
    with open(GT_ROOT / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def _resolve_csv_path(entry):
    """Resolve relative csv_path entries against PROJECT_ROOT (in-place)."""
    p = Path(entry["csv_path"])
    if not p.is_absolute():
        entry["csv_path"] = str(PROJECT_ROOT / p)
    return entry


def load_manifest():
    lookup = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = _resolve_csv_path(json.loads(line.strip()))
            lookup[entry["table_id"]] = entry
    return lookup


def load_parents():
    lookup = {}
    with open(PARENTS_PATH) as f:
        for line in f:
            entry = _resolve_csv_path(json.loads(line.strip()))
            lookup[entry["parent_id"]] = entry
    return lookup


def load_row_embeddings(model_name):
    q_path = ROW_EMB_ROOT / model_name / "dlte_v1_queries.pkl"
    t_path = ROW_EMB_ROOT / model_name / "dlte_v1_targets.pkl"
    with open(q_path, "rb") as f:
        q_pkl = pickle.load(f)
    with open(t_path, "rb") as f:
        t_pkl = pickle.load(f)
    q_lookup = {e["table_id"]: e["row_embeddings"] for e in q_pkl}
    t_lookup = {e["table_id"]: e["row_embeddings"] for e in t_pkl}
    del q_pkl, t_pkl
    return q_lookup, t_lookup


# ── CellF1 ────────────────────────────────────────────────────────

def normalize_cell(val):
    s = str(val).strip().lower()
    if s in ("nan", "none", ""):
        return None
    return s


def cell_f1(enriched_df, parent_df):
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
    n_e = sum(e_cells.values())
    n_p = sum(p_cells.values())
    prec = tp / n_e if n_e > 0 else 0
    rec = tp / n_p if n_p > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


# ── Row Matching Helpers ──────────────────────────────────────────

def normalize_key(val):
    return str(val).strip().lower()


def cosine_sim_matrix(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


def match_rows_for_join(query_df, join_df, key_col_q, key_col_j,
                        q_row_embs, j_row_embs):
    """Match query rows to join rows. Returns {enriched_row: join_row}."""
    row_mapping = {}
    if key_col_q and key_col_j and key_col_q in query_df.columns and key_col_j in join_df.columns:
        join_key_index = defaultdict(list)
        for j_idx, val in enumerate(join_df[key_col_j]):
            join_key_index[normalize_key(val)].append(j_idx)
        used = set()
        for q_idx, val in enumerate(query_df[key_col_q]):
            for j_idx in join_key_index.get(normalize_key(val), []):
                if j_idx not in used:
                    row_mapping[q_idx] = j_idx
                    used.add(j_idx)
                    break

    n_with_embs = len(q_row_embs) if q_row_embs is not None else 0
    unmatched = [i for i in range(len(query_df)) if i not in row_mapping and i < n_with_embs]
    if unmatched and q_row_embs is not None and j_row_embs is not None:
        used = set(row_mapping.values())
        avail = [j for j in range(len(join_df)) if j not in used]
        if avail:
            sim = cosine_sim_matrix(q_row_embs[unmatched], j_row_embs[avail])
            for flat_idx in np.argsort(sim.ravel())[::-1]:
                qi = int(flat_idx // len(avail))
                ji = int(flat_idx % len(avail))
                if qi in {i for i, _ in enumerate(unmatched) if unmatched[i] in row_mapping}:
                    continue
                if sim[qi, ji] < ROW_SIM_THRESHOLD:
                    break
                row_mapping[unmatched[qi]] = avail[ji]
    return row_mapping


# ── Oracle S1+S2: GT candidates + GT column alignment ────────────

def oracle_s1s2_query(qt, manifest, parents, row_q_lookup, row_t_lookup):
    """Oracle Stage 1+2: use GT candidates AND GT column alignments.
    Only row matching uses the model (predicted)."""
    qid = qt["query_table_id"]
    parent_entry = parents.get(qt["parent_id"])
    if parent_entry is None:
        return None
    parent_csv = Path(parent_entry["csv_path"])
    if not parent_csv.exists():
        return None
    parent_df = pd.read_csv(parent_csv, engine="python", on_bad_lines="skip")

    q_manifest = manifest.get(qid)
    if q_manifest is None:
        return None
    query_df = pd.read_csv(q_manifest["csv_path"])
    key_col_q = q_manifest.get("key_col_fragment")

    seed_npz_path = TABLE_MAPS_DIR / f"{qid}.npz"
    if not seed_npz_path.exists():
        return None
    seed_parent_cols = np.load(seed_npz_path)["col_parent_idx"]

    q_row_embs = row_q_lookup.get(qid)
    enriched = query_df.copy()

    # Get GT candidates
    gt_union_id = None
    gt_join_id = None
    for rel in qt.get("relevant", []):
        if rel["relation"] == "union":
            gt_union_id = rel["table_id"]
        elif rel["relation"] == "join":
            gt_join_id = rel["table_id"]

    # ── Oracle Union (GT alignment) ──
    if gt_union_id:
        u_manifest = manifest.get(gt_union_id)
        if u_manifest and Path(u_manifest["csv_path"]).exists():
            cand_df = pd.read_csv(u_manifest["csv_path"])
            cand_npz_path = TABLE_MAPS_DIR / f"{gt_union_id}.npz"
            if cand_npz_path.exists():
                cand_parent_cols = np.load(cand_npz_path)["col_parent_idx"]
                # GT column alignment: match via shared parent column indices
                gt_pairs = []
                for i, sp in enumerate(seed_parent_cols):
                    if sp < 0:
                        continue
                    for j, cp in enumerate(cand_parent_cols):
                        if cp == sp:
                            gt_pairs.append((int(i), int(j), 1.0))
                            break

                # Dedup new rows via key string
                key_col_c = u_manifest.get("key_col_fragment")
                if key_col_q and key_col_c and key_col_q in enriched.columns and key_col_c in cand_df.columns:
                    q_keys = {normalize_key(v) for v in enriched[key_col_q]}
                    new_rows = [i for i, val in enumerate(cand_df[key_col_c])
                                if normalize_key(val) not in q_keys]
                else:
                    new_rows = list(range(len(cand_df)))

                # Merge union with GT alignment
                q_cols = list(enriched.columns)
                c_cols = list(cand_df.columns)
                idx_map = {}
                for q_idx, c_idx, _ in gt_pairs:
                    if q_idx < len(q_cols) and c_idx < len(c_cols):
                        idx_map[c_idx] = q_idx
                new_data = []
                for row_i in new_rows:
                    row_dict = {q_cols[q_idx]: cand_df.iloc[row_i, c_idx]
                                for c_idx, q_idx in idx_map.items()}
                    new_data.append(row_dict)
                if new_data:
                    new_rows_df = pd.DataFrame(new_data, columns=enriched.columns)
                    enriched = pd.concat([enriched, new_rows_df], ignore_index=True)

    # ── Oracle Join (GT alignment, predicted row matching) ──
    if gt_join_id:
        j_manifest = manifest.get(gt_join_id)
        if j_manifest and Path(j_manifest["csv_path"]).exists():
            join_df = pd.read_csv(j_manifest["csv_path"])
            key_col_j = j_manifest.get("key_col_fragment")
            j_row_embs = row_t_lookup.get(gt_join_id)

            join_npz_path = TABLE_MAPS_DIR / f"{gt_join_id}.npz"
            if join_npz_path.exists():
                join_parent_cols = np.load(join_npz_path)["col_parent_idx"]
                seed_col_set = set(int(x) for x in seed_parent_cols if x >= 0)

                # GT key pair: the shared column
                gt_key_pair = None
                for i, sp in enumerate(seed_parent_cols):
                    if sp < 0:
                        continue
                    for j, jp in enumerate(join_parent_cols):
                        if jp == sp:
                            gt_key_pair = (int(i), int(j))
                            break
                    if gt_key_pair:
                        break

                # New columns: join columns not in seed
                j_cols = list(join_df.columns)
                new_col_indices = []
                for j, jp in enumerate(join_parent_cols):
                    if jp >= 0 and int(jp) not in seed_col_set:
                        new_col_indices.append(j)

                if new_col_indices:
                    # Row matching (predicted — this is what varies by row model)
                    row_mapping = match_rows_for_join(
                        enriched, join_df, key_col_q, key_col_j,
                        q_row_embs, j_row_embs)

                    # Add new columns
                    new_col_names = [j_cols[i] for i in new_col_indices]
                    final_names = []
                    for name in new_col_names:
                        if name in enriched.columns:
                            name = f"{name}_join"
                        final_names.append(name)

                    for fname in final_names:
                        enriched[fname] = pd.Series([np.nan] * len(enriched), dtype=object)
                    for e_idx, j_idx in row_mapping.items():
                        if e_idx < len(enriched) and j_idx < len(join_df):
                            for orig_idx, fname in zip(new_col_indices, final_names):
                                enriched.at[e_idx, fname] = join_df.iloc[j_idx, orig_idx]

    return {"cell_f1": cell_f1(enriched, parent_df), "tier": qt["noise_tier"]}


# ── Oracle S1+S2+S3: GT everything (upper bound) ─────────────────

def oracle_s1s2s3_query(qt, manifest, parents):
    """Oracle Stage 1+2+3: GT candidates + GT column alignment + GT row mapping.
    This is the theoretical upper bound — tests only CellF1 metric correctness."""
    qid = qt["query_table_id"]
    parent_entry = parents.get(qt["parent_id"])
    if parent_entry is None:
        return None
    parent_csv = Path(parent_entry["csv_path"])
    if not parent_csv.exists():
        return None
    parent_df = pd.read_csv(parent_csv, engine="python", on_bad_lines="skip")

    q_manifest = manifest.get(qid)
    if q_manifest is None:
        return None
    query_df = pd.read_csv(q_manifest["csv_path"])

    seed_npz_path = TABLE_MAPS_DIR / f"{qid}.npz"
    if not seed_npz_path.exists():
        return None
    seed_npz = np.load(seed_npz_path)
    seed_row_parent = seed_npz["row_parent_idx"]
    seed_col_parent = seed_npz["col_parent_idx"]

    enriched = query_df.copy()

    gt_union_id = None
    gt_join_id = None
    for rel in qt.get("relevant", []):
        if rel["relation"] == "union":
            gt_union_id = rel["table_id"]
        elif rel["relation"] == "join":
            gt_join_id = rel["table_id"]

    # ── Oracle Union (GT everything) ──
    if gt_union_id:
        u_manifest = manifest.get(gt_union_id)
        if u_manifest and Path(u_manifest["csv_path"]).exists():
            cand_df = pd.read_csv(u_manifest["csv_path"])
            cand_npz_path = TABLE_MAPS_DIR / f"{gt_union_id}.npz"
            if cand_npz_path.exists():
                cand_npz = np.load(cand_npz_path)
                cand_row_parent = cand_npz["row_parent_idx"]
                cand_col_parent = cand_npz["col_parent_idx"]

                # GT col alignment
                gt_pairs = []
                for i, sp in enumerate(seed_col_parent):
                    if sp < 0:
                        continue
                    for j, cp in enumerate(cand_col_parent):
                        if cp == sp:
                            gt_pairs.append((int(i), int(j), 1.0))
                            break

                # GT row dedup: new rows = cand rows whose parent_row NOT in seed's parent_rows
                seed_row_set = set(int(x) for x in seed_row_parent if x >= 0)
                new_rows = [i for i, rp in enumerate(cand_row_parent)
                            if rp >= 0 and int(rp) not in seed_row_set]

                # Merge
                q_cols = list(enriched.columns)
                c_cols = list(cand_df.columns)
                idx_map = {}
                for q_idx, c_idx, _ in gt_pairs:
                    if q_idx < len(q_cols) and c_idx < len(c_cols):
                        idx_map[c_idx] = q_idx
                new_data = []
                for row_i in new_rows:
                    row_dict = {q_cols[q_idx]: cand_df.iloc[row_i, c_idx]
                                for c_idx, q_idx in idx_map.items()}
                    new_data.append(row_dict)
                if new_data:
                    new_rows_df = pd.DataFrame(new_data, columns=enriched.columns)
                    enriched = pd.concat([enriched, new_rows_df], ignore_index=True)

    # ── Oracle Join (GT everything) ──
    if gt_join_id:
        j_manifest = manifest.get(gt_join_id)
        if j_manifest and Path(j_manifest["csv_path"]).exists():
            join_df = pd.read_csv(j_manifest["csv_path"])
            join_npz_path = TABLE_MAPS_DIR / f"{gt_join_id}.npz"
            if join_npz_path.exists():
                join_npz = np.load(join_npz_path)
                join_row_parent = join_npz["row_parent_idx"]
                join_col_parent = join_npz["col_parent_idx"]

                seed_col_set = set(int(x) for x in seed_col_parent if x >= 0)

                # New columns
                j_cols = list(join_df.columns)
                new_col_indices = [j for j, jp in enumerate(join_col_parent)
                                   if jp >= 0 and int(jp) not in seed_col_set]

                if new_col_indices:
                    # GT row mapping: match by parent row index
                    # Build enriched->parent_row mapping
                    # Original seed rows have known parent indices
                    # Union-appended rows also have known parent indices
                    enriched_to_parent_row = {}
                    # Seed rows
                    for i, rp in enumerate(seed_row_parent):
                        if rp >= 0:
                            enriched_to_parent_row[i] = int(rp)
                    # Union rows (appended after seed)
                    if gt_union_id:
                        u_npz_path = TABLE_MAPS_DIR / f"{gt_union_id}.npz"
                        if u_npz_path.exists():
                            u_npz = np.load(u_npz_path)
                            u_row_parent = u_npz["row_parent_idx"]
                            seed_row_set = set(int(x) for x in seed_row_parent if x >= 0)
                            offset = len(seed_row_parent)
                            for i, rp in enumerate(u_row_parent):
                                if rp >= 0 and int(rp) not in seed_row_set:
                                    enriched_to_parent_row[offset] = int(rp)
                                    offset += 1

                    # Join row->parent mapping
                    join_parent_to_row = {}
                    for j, rp in enumerate(join_row_parent):
                        if rp >= 0:
                            join_parent_to_row[int(rp)] = j

                    # Match: enriched row -> join row via shared parent row
                    row_mapping = {}
                    for e_idx, p_row in enriched_to_parent_row.items():
                        if p_row in join_parent_to_row:
                            row_mapping[e_idx] = join_parent_to_row[p_row]

                    # Add columns
                    new_col_names = [j_cols[i] for i in new_col_indices]
                    final_names = []
                    for name in new_col_names:
                        if name in enriched.columns:
                            name = f"{name}_join"
                        final_names.append(name)
                    for fname in final_names:
                        enriched[fname] = pd.Series([np.nan] * len(enriched), dtype=object)
                    for e_idx, j_idx in row_mapping.items():
                        if e_idx < len(enriched) and j_idx < len(join_df):
                            for orig_idx, fname in zip(new_col_indices, final_names):
                                enriched.at[e_idx, fname] = join_df.iloc[j_idx, orig_idx]

    return {"cell_f1": cell_f1(enriched, parent_df), "tier": qt["noise_tier"]}


# ── Per-Combination Oracle S1+S2 Evaluation ──────────────────────

def evaluate_oracle_s1s2(row_model, query_tasks, manifest, parents, splits):
    """Evaluate Oracle S1+S2 for one row model (col model doesn't matter here)."""
    print(f"\n  Oracle S1+S2 with row_model={row_model}")
    t0 = time.time()

    print(f"    Loading row embeddings ({row_model})...")
    row_q_lookup, row_t_lookup = load_row_embeddings(row_model)

    results = {}
    for split in splits:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue
        print(f"    Split: {split} ({len(split_tasks)} queries)")

        all_f1 = []
        tier_f1 = defaultdict(list)
        n = 0
        for qt in split_tasks:
            r = oracle_s1s2_query(qt, manifest, parents, row_q_lookup, row_t_lookup)
            if r is None:
                continue
            all_f1.append(r["cell_f1"])
            tier_f1[r["tier"]].append(r["cell_f1"])
            n += 1
            if n % 500 == 0:
                print(f"      {n}/{len(split_tasks)}...")

        if all_f1:
            mean = float(np.mean(all_f1))
            results[split] = {"cell_f1": mean, "n": n,
                              "per_tier": {t: float(np.mean(v)) for t, v in sorted(tier_f1.items())}}
            print(f"      CellF1={mean:.4f}")

    print(f"    Done in {time.time() - t0:.1f}s")
    return results


def evaluate_oracle_s1s2s3(query_tasks, manifest, parents, splits):
    """Evaluate Oracle S1+S2+S3 (upper bound — no model dependency)."""
    print("\n  Oracle S1+S2+S3 (upper bound)")
    t0 = time.time()

    results = {}
    for split in splits:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue
        print(f"    Split: {split} ({len(split_tasks)} queries)")

        all_f1 = []
        tier_f1 = defaultdict(list)
        n = 0
        for qt in split_tasks:
            r = oracle_s1s2s3_query(qt, manifest, parents)
            if r is None:
                continue
            all_f1.append(r["cell_f1"])
            tier_f1[r["tier"]].append(r["cell_f1"])
            n += 1
            if n % 500 == 0:
                print(f"      {n}/{len(split_tasks)}...")

        if all_f1:
            mean = float(np.mean(all_f1))
            results[split] = {"cell_f1": mean, "n": n,
                              "per_tier": {t: float(np.mean(v)) for t, v in sorted(tier_f1.items())}}
            print(f"      CellF1={mean:.4f}")

    print(f"    Done in {time.time() - t0:.1f}s")
    return results


# ── Main (single combination mode) ──────────────────────────────

def run_single(col_model, row_model, query_tasks, manifest, parents, splits):
    """Run oracle S1+S2 and S1+S2+S3 for one row model."""
    results = {
        "col_model": col_model,
        "row_model": row_model,
    }

    # Oracle S1+S2
    results["oracle_s1s2"] = evaluate_oracle_s1s2(
        row_model, query_tasks, manifest, parents, splits)

    # Oracle S1+S2+S3 (only once per combination, no model dependency)
    results["oracle_s1s2s3"] = evaluate_oracle_s1s2s3(
        query_tasks, manifest, parents, splits)

    # Save
    out_dir = ABLATION_ROOT / "oracle_bounds"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{col_model}__{row_model}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Aggregation ──────────────────────────────────────────────────

def aggregate(splits):
    """Aggregate all ablation data into final CSVs and figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    # ── A3: Stage-wise oracle bounds ──
    print("\n  A3: Stage-wise Oracle Bounds")

    # Collect oracle S1 from Step 13
    oracle_s1_data = {}
    for col in COLUMN_MODELS:
        for row in ROW_MODELS:
            path = HEATMAP_ROOT / "metrics" / f"{col}__{row}.json"
            if path.exists():
                data = json.loads(path.read_text())
                for split in splits:
                    sm = data.get("splits", {}).get(split, {})
                    if sm:
                        oracle_s1_data[(col, row, split)] = sm.get("cell_f1")

    # Collect oracle S1+S2 and S1+S2+S3
    oracle_s1s2_data = {}
    oracle_s1s2s3_data = {}
    bounds_dir = ABLATION_ROOT / "oracle_bounds"
    for col in COLUMN_MODELS:
        for row in ROW_MODELS:
            path = bounds_dir / f"{col}__{row}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for split in splits:
                s12 = data.get("oracle_s1s2", {}).get(split, {})
                s123 = data.get("oracle_s1s2s3", {}).get(split, {})
                if s12:
                    oracle_s1s2_data[(col, row, split)] = s12.get("cell_f1")
                if s123:
                    oracle_s1s2s3_data[(col, row, split)] = s123.get("cell_f1")

    # Collect non-oracle from Step 11
    nonocle_data = {}
    for col in COLUMN_MODELS:
        for row in ROW_MODELS:
            path = METRICS_ROOT / f"{col}__{row}" / "end_to_end.json"
            if path.exists():
                data = json.loads(path.read_text())
                for split in splits:
                    sm = data.get("splits", {}).get(split, {})
                    if sm:
                        nonocle_data[(col, row, split)] = sm.get("cell_f1")

    # Build ablations CSV
    rows = []
    for col in COLUMN_MODELS:
        for row in ROW_MODELS:
            for split in splits:
                r = {
                    "col_model": col, "row_model": row, "split": split,
                    "no_oracle": nonocle_data.get((col, row, split)),
                    "oracle_s1": oracle_s1_data.get((col, row, split)),
                    "oracle_s1s2": oracle_s1s2_data.get((col, row, split)),
                    "oracle_s1s2s3": oracle_s1s2s3_data.get((col, row, split)),
                }
                rows.append(r)

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(ABLATION_ROOT / "ablations.csv", index=False)
    print(f"    Saved: ablations.csv ({len(ablation_df)} rows)")

    # ── Figure: Stage-wise oracle bounds (bar chart, dev) ──
    for split in splits:
        split_df = ablation_df[ablation_df["split"] == split].copy()
        if split_df.empty:
            continue

        # Average across row models (since row model barely matters)
        avg = split_df.groupby("col_model")[["no_oracle", "oracle_s1", "oracle_s1s2", "oracle_s1s2s3"]].mean()
        avg = avg.reindex(COLUMN_MODELS)

        MODEL_LABELS = {
            "bert": "BERT", "gte": "GTE",
            "starmie": "Starmie", "tabbie": "TABBIE", "tabert": "TaBERT",
            "tabsketchfm": "TabSketchFM", "tapas": "TAPAS", "turl": "TURL",
        }
        avg.index = [MODEL_LABELS.get(m, m) for m in avg.index]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(avg))
        width = 0.2

        stages = ["no_oracle", "oracle_s1", "oracle_s1s2", "oracle_s1s2s3"]
        labels = ["No Oracle", "Oracle S1", "Oracle S1+S2", "Oracle S1+S2+S3"]
        colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

        for i, (stage, label, color) in enumerate(zip(stages, labels, colors)):
            vals = avg[stage].fillna(0).values
            ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(avg.index, rotation=15, ha="right")
        ax.set_ylabel("CellF1")
        ax.set_title(f"Stage-wise Oracle Bounds ({split} split)")
        ax.legend(loc="upper left")
        ax.set_ylim(0.4, 1.05)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(ABLATION_ROOT / f"stagewise_oracle_bounds_{split}.pdf",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: stagewise_oracle_bounds_{split}.pdf")

    # ── A1: Noise tier breakdown (from Step 13 oracle data) ──
    print("\n  A1: Noise Tier Breakdown")
    tier_rows = []
    for col in COLUMN_MODELS:
        for row in ROW_MODELS:
            # Oracle S1 tiers
            path = HEATMAP_ROOT / "metrics" / f"{col}__{row}.json"
            if path.exists():
                data = json.loads(path.read_text())
                for split in splits:
                    sm = data.get("splits", {}).get(split, {})
                    for tier, tm in sm.get("per_tier", {}).items():
                        tier_rows.append({
                            "col_model": col, "row_model": row, "split": split,
                            "tier": int(tier), "oracle_s1_cell_f1": tm.get("cell_f1"),
                        })

    if tier_rows:
        tier_df = pd.DataFrame(tier_rows)
        tier_df.to_csv(ABLATION_ROOT / "noise_tier_breakdown.csv", index=False)
        print(f"    Saved: noise_tier_breakdown.csv ({len(tier_df)} rows)")

        # Figure: tier curves (oracle S1, dev, averaged across row models)
        for split in splits:
            split_tier = tier_df[tier_df["split"] == split]
            if split_tier.empty:
                continue

            avg_tier = split_tier.groupby(["col_model", "tier"])["oracle_s1_cell_f1"].mean().reset_index()

            fig, ax = plt.subplots(figsize=(8, 5))
            palette = sns.color_palette("tab10", n_colors=len(COLUMN_MODELS))
            for i, col in enumerate(COLUMN_MODELS):
                model_data = avg_tier[avg_tier["col_model"] == col].sort_values("tier")
                if not model_data.empty:
                    ax.plot(model_data["tier"], model_data["oracle_s1_cell_f1"],
                            "o-", label=MODEL_LABELS.get(col, col),
                            color=palette[i], linewidth=2, markersize=6)

            ax.set_xlabel("Noise Tier")
            ax.set_ylabel("CellF1 (Oracle S1)")
            ax.set_title(f"CellF1 by Noise Tier ({split} split)")
            ax.set_xticks([0, 1, 2, 3])
            ax.legend(loc="lower left")
            ax.set_ylim(0.6, 1.05)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            fig.savefig(ABLATION_ROOT / f"noise_tier_curve_{split}.pdf",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: noise_tier_curve_{split}.pdf")

    # ── A2: K sensitivity ──
    print("\n  A2: K Sensitivity")
    k_rows = []
    stage1_root = RESULTS_ROOT / "stage1"
    for col in COLUMN_MODELS:
        for split in splits:
            for k in [10, 50, 100]:
                path = stage1_root / col / f"metrics_{split}_topk_{k}.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    k_rows.append({
                        "col_model": col, "split": split, "k": k,
                        "recall_any": data["recall_any"],
                        "recall_union": data["recall_union"],
                        "recall_join": data["recall_join"],
                    })

    if k_rows:
        k_df = pd.DataFrame(k_rows)
        k_df.to_csv(ABLATION_ROOT / "topk_sensitivity.csv", index=False)
        print(f"    Saved: topk_sensitivity.csv ({len(k_df)} rows)")

        # Figure: K sensitivity
        for split in splits:
            split_k = k_df[k_df["split"] == split]
            if split_k.empty:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            metric_names = ["recall_any", "recall_union", "recall_join"]
            metric_titles = ["Recall@K (Any)", "Recall@K (Union)", "Recall@K (Join)"]
            palette = sns.color_palette("tab10", n_colors=len(COLUMN_MODELS))

            for ax, metric, title in zip(axes, metric_names, metric_titles):
                for i, col in enumerate(COLUMN_MODELS):
                    model_data = split_k[split_k["col_model"] == col].sort_values("k")
                    if not model_data.empty:
                        ax.plot(model_data["k"], model_data[metric],
                                "o-", label=MODEL_LABELS.get(col, col),
                                color=palette[i], linewidth=2, markersize=6)
                ax.set_xlabel("K")
                ax.set_title(title)
                ax.set_xticks([10, 50, 100])
                ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3)

            axes[0].set_ylabel("Recall")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4,
                       bbox_to_anchor=(0.5, -0.05), fontsize=9)
            fig.suptitle(f"K Sensitivity ({split} split)", fontsize=14, fontweight="bold", y=1.02)

            plt.tight_layout()
            fig.savefig(ABLATION_ROOT / f"topk_sensitivity_{split}.pdf",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: topk_sensitivity_{split}.pdf")

    print(f"\n  Outputs: {ABLATION_ROOT}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 14: Ablations + Baselines")
    parser.add_argument("--col_model", type=str)
    parser.add_argument("--row_model", type=str)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory for DLTE results")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory")
    args = parser.parse_args()
    resolve_paths(args)

    if args.aggregate:
        print("Step 14: Aggregating Ablations")
        print("=" * 60)
        aggregate(args.splits)
        print(f"\n{'='*60}")
        print("Done")
        return 0

    if not args.col_model or not args.row_model:
        print("ERROR: --col_model and --row_model required (or use --aggregate)")
        return 1

    print("Step 14: Ablations + Baselines")
    print("=" * 60)
    print(f"  col_model: {args.col_model}")
    print(f"  row_model: {args.row_model}")

    query_tasks = load_query_tasks()
    manifest = load_manifest()
    parents = load_parents()

    run_single(args.col_model, args.row_model, query_tasks, manifest, parents, args.splits)

    print(f"\n{'='*60}")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
