"""
Step 13: Oracle Stage 1 — 7×4 Heatmap.

Bypass Stage 1 retrieval by using GT union/join candidates directly.
Still runs full Stage 2 (column alignment + classification) and Stage 3
(row matching + merge), then evaluates CellF1.

This isolates the quality of column alignment + row matching models
from the retrieval step.

Usage:
    python downstream_tasks/dlte/scripts/step13_oracle_heatmap.py --col_model bert --row_model tabicl
    python downstream_tasks/dlte/scripts/step13_oracle_heatmap.py --aggregate
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
from scipy.optimize import linear_sum_assignment

# ── Paths (resolved at runtime by resolve_paths()) ───────────────

PROJECT_ROOT = DATASET_ROOT = GT_ROOT = TABLE_MAPS_DIR = None
MANIFEST_PATH = PARENTS_PATH = COL_EMB_ROOT = ROW_EMB_ROOT = None
STAGE2_ROOT = HEATMAP_ROOT = None

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
ROW_MODELS = [
    "bert", "dae", "gte", "saint", "scarf", "subtab",
    "tabbie", "tabicl", "tabpfn", "tabtransformer", "tabular_binning",
    "transtab", "tuta", "vime",
]

ROW_SIM_THRESHOLD = 0.80


# ── Data Loading ───────────────────────────────────────────────────

def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, DATASET_ROOT, GT_ROOT, TABLE_MAPS_DIR
    global MANIFEST_PATH, PARENTS_PATH, COL_EMB_ROOT, ROW_EMB_ROOT
    global STAGE2_ROOT, HEATMAP_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
    GT_ROOT = DATASET_ROOT / "ground_truth"
    TABLE_MAPS_DIR = GT_ROOT / "table_maps"
    MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
    PARENTS_PATH = DATASET_ROOT / "manifests" / "parents_filtered.jsonl"
    COL_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "column"
    ROW_EMB_ROOT = PROJECT_ROOT / "assets" / "embeddings" / "row"
    STAGE2_ROOT = output_root / "stage2"
    HEATMAP_ROOT = output_root / "experiments" / "heatmap_oracle_stage1"


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


def load_column_embeddings(model_name):
    """Load column embeddings -> (query_lookup, target_lookup).
    Each lookup: table_id -> dict of col_idx -> embedding vector."""
    with open(COL_EMB_ROOT / model_name / "dlte_v1_queries.pkl", "rb") as f:
        q_pkl = pickle.load(f)
    with open(COL_EMB_ROOT / model_name / "dlte_v1_targets.pkl", "rb") as f:
        t_pkl = pickle.load(f)

    q_lookup = {e["table_id"]: e["column_embeddings"] for e in q_pkl}
    t_lookup = {e["table_id"]: e["column_embeddings"] for e in t_pkl}
    del q_pkl, t_pkl
    return q_lookup, t_lookup


def load_row_embeddings(model_name):
    """Load row embeddings -> (query_lookup, target_lookup).
    Each lookup: table_id -> np.ndarray(n_rows, dim)."""
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


def load_calibrated_thresholds(col_model):
    """Load calibrated thresholds from Stage 2 calibration."""
    cal_path = STAGE2_ROOT / col_model / "calibration_dev.json"
    if cal_path.exists():
        return json.loads(cal_path.read_text()).get("best_thresholds")
    return None


# ── Column Alignment (from Step 9) ────────────────────────────────

def hungarian_align(query_cols, candidate_cols):
    """Hungarian matching on column embeddings. Returns alignment dict."""
    q_keys = sorted(query_cols.keys())
    c_keys = sorted(candidate_cols.keys())
    if not q_keys or not c_keys:
        return {"pairs": [], "n_query_cols": len(q_keys), "n_cand_cols": len(c_keys),
                "sims": np.array([], dtype=np.float32)}

    q_vecs = np.array([query_cols[k] for k in q_keys], dtype=np.float32)
    c_vecs = np.array([candidate_cols[k] for k in c_keys], dtype=np.float32)

    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    c_norms[c_norms == 0] = 1.0
    sim_matrix = (q_vecs / q_norms) @ (c_vecs / c_norms).T

    cost = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    sims = sim_matrix[row_ind, col_ind]

    pairs = []
    for r, c, s in zip(row_ind, col_ind, sims):
        pairs.append((int(q_keys[r]), int(c_keys[c]), float(s)))

    return {"pairs": pairs, "n_query_cols": len(q_keys), "n_cand_cols": len(c_keys),
            "sims": sims}


def classify_pair(align, thresholds):
    """Classify a (query, candidate) pair. Returns (relation, confidence, key_pair)."""
    tau = thresholds
    sims = align["sims"]
    n_q = align["n_query_cols"]
    if len(sims) == 0:
        return "none", 0.0, None

    matched_mask = sims >= tau["tau_match_floor"]
    n_matched = int(matched_mask.sum())
    match_ratio = n_matched / n_q if n_q > 0 else 0.0
    mean_sim = float(sims[matched_mask].mean()) if n_matched > 0 else 0.0
    max_sim = float(sims.max())

    if match_ratio >= tau["tau_union"] and mean_sim >= tau["tau_union_sim"]:
        return "union", float(min(match_ratio, mean_sim)), None

    if (match_ratio <= tau["tau_join_max"]
            and max_sim >= tau["tau_key_sim"]
            and 1 <= n_matched <= 3):
        best_idx = int(np.argmax(sims))
        key_pair = (align["pairs"][best_idx][0], align["pairs"][best_idx][1])
        return "join", float(max_sim), key_pair

    return "none", float(1.0 - max_sim), None


# ── Row Matching + Merge (from Step 10) ───────────────────────────

def normalize_key(val):
    return str(val).strip().lower()


def cosine_sim_matrix(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


def match_rows_for_union(query_df, cand_df, key_col_q, key_col_c,
                         q_row_embs, c_row_embs):
    key_matching_available = (key_col_q and key_col_q in query_df.columns
                              and key_col_c and key_col_c in cand_df.columns)
    if key_matching_available:
        q_keys = {normalize_key(v) for v in query_df[key_col_q]}
        new_rows = [i for i, val in enumerate(cand_df[key_col_c])
                    if normalize_key(val) not in q_keys]
        method = "key_string"
    elif q_row_embs is not None and c_row_embs is not None:
        sim = cosine_sim_matrix(c_row_embs, q_row_embs)
        max_sim = sim.max(axis=1)
        new_rows = [i for i in range(len(cand_df)) if max_sim[i] < ROW_SIM_THRESHOLD]
        method = "embedding_fallback"
    else:
        new_rows = list(range(len(cand_df)))
        method = "string_only"
    return new_rows, method


def match_rows_for_join(query_df, join_df, key_pair, key_col_q, key_col_j,
                        q_row_embs, j_row_embs):
    row_mapping = {}
    method = "key_string"
    if key_col_q and key_col_j and key_col_q in query_df.columns and key_col_j in join_df.columns:
        join_key_index = defaultdict(list)
        for j_idx, val in enumerate(join_df[key_col_j]):
            join_key_index[normalize_key(val)].append(j_idx)
        used_join_rows = set()
        for q_idx, val in enumerate(query_df[key_col_q]):
            nk = normalize_key(val)
            for j_idx in join_key_index.get(nk, []):
                if j_idx not in used_join_rows:
                    row_mapping[q_idx] = j_idx
                    used_join_rows.add(j_idx)
                    break

    n_with_embs = len(q_row_embs) if q_row_embs is not None else 0
    unmatched_q = [i for i in range(len(query_df)) if i not in row_mapping and i < n_with_embs]
    if unmatched_q and q_row_embs is not None and j_row_embs is not None:
        used_join_rows = set(row_mapping.values())
        available_j = [j for j in range(len(join_df)) if j not in used_join_rows]
        if available_j:
            q_embs = q_row_embs[unmatched_q]
            j_embs = j_row_embs[available_j]
            sim = cosine_sim_matrix(q_embs, j_embs)
            flat_indices = np.argsort(sim.ravel())[::-1]
            used_q = set()
            used_j_local = set()
            for flat_idx in flat_indices:
                qi = int(flat_idx // len(available_j))
                ji = int(flat_idx % len(available_j))
                if qi in used_q or ji in used_j_local:
                    continue
                if sim[qi, ji] < ROW_SIM_THRESHOLD:
                    break
                row_mapping[unmatched_q[qi]] = available_j[ji]
                used_q.add(qi)
                used_j_local.add(ji)
            if used_q:
                method = "embedding_fallback"
    return row_mapping, method


def merge_union(query_df, cand_df, col_alignment, new_row_indices):
    if not new_row_indices:
        return query_df.copy()
    q_cols = list(query_df.columns)
    c_cols = list(cand_df.columns)
    idx_map = {}
    for q_idx, c_idx, _ in col_alignment:
        if q_idx < len(q_cols) and c_idx < len(c_cols):
            idx_map[c_idx] = q_idx
    new_data = []
    for row_i in new_row_indices:
        row_dict = {}
        for c_idx, q_idx in idx_map.items():
            row_dict[q_cols[q_idx]] = cand_df.iloc[row_i, c_idx]
        new_data.append(row_dict)
    new_rows_df = pd.DataFrame(new_data, columns=query_df.columns)
    return pd.concat([query_df, new_rows_df], ignore_index=True)


def merge_join(enriched_df, join_df, col_alignment, key_pair, row_mapping):
    j_cols = list(join_df.columns)
    shared_j_idx = key_pair[1] if key_pair else None
    new_col_indices = [i for i in range(len(j_cols)) if i != shared_j_idx]
    new_col_names = [j_cols[i] for i in new_col_indices]
    if not new_col_names:
        return enriched_df.copy(), []
    final_names = []
    for name in new_col_names:
        if name in enriched_df.columns:
            name = f"{name}_join"
        final_names.append(name)
    result = enriched_df.copy()
    for fname in final_names:
        result[fname] = pd.Series([np.nan] * len(result), dtype=object)
    for e_idx, j_idx in row_mapping.items():
        if e_idx < len(result) and j_idx < len(join_df):
            for orig_col_idx, fname in zip(new_col_indices, final_names):
                result.at[e_idx, fname] = join_df.iloc[j_idx, orig_col_idx]
    return result, final_names


# ── CellF1 (from Step 11) ────────────────────────────────────────

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


# ── Oracle Pipeline ──────────────────────────────────────────────

def oracle_process_query(qt, manifest, parents, thresholds,
                         col_q_lookup, col_t_lookup,
                         row_q_lookup, row_t_lookup):
    """Process one query with oracle (GT) candidates.

    Returns dict with cell_f1, tier, and details, or None if parent missing.
    """
    qid = qt["query_table_id"]
    parent_id = qt["parent_id"]
    tier = qt["noise_tier"]

    # Load parent
    parent_entry = parents.get(parent_id)
    if parent_entry is None:
        return None
    parent_csv_path = Path(parent_entry["csv_path"])
    if not parent_csv_path.exists():
        return None
    parent_df = pd.read_csv(parent_csv_path, engine="python", on_bad_lines="skip")

    # Load query seed
    q_manifest = manifest.get(qid)
    if q_manifest is None:
        return None
    query_df = pd.read_csv(q_manifest["csv_path"])
    key_col_q = q_manifest.get("key_col_fragment")

    # Get GT candidates
    gt_union_id = None
    gt_join_id = None
    for rel in qt.get("relevant", []):
        if rel["relation"] == "union":
            gt_union_id = rel["table_id"]
        elif rel["relation"] == "join":
            gt_join_id = rel["table_id"]

    # Column embeddings for query
    q_col_embs = col_q_lookup.get(qid)
    q_row_embs = row_q_lookup.get(qid)

    enriched = query_df.copy()
    union_correct = False
    join_correct = False

    # ── Oracle Union ──
    if gt_union_id:
        u_manifest = manifest.get(gt_union_id)
        c_col_embs = col_t_lookup.get(gt_union_id)

        if u_manifest and Path(u_manifest["csv_path"]).exists() and q_col_embs and c_col_embs:
            cand_df = pd.read_csv(u_manifest["csv_path"])
            key_col_c = u_manifest.get("key_col_fragment")
            c_row_embs = row_t_lookup.get(gt_union_id)

            # Column alignment
            align = hungarian_align(q_col_embs, c_col_embs)

            # Classify (optional — we know it's union, but let Stage 2 decide)
            if thresholds:
                pred_rel, conf, _ = classify_pair(align, thresholds)
                union_correct = (pred_rel == "union")
            else:
                union_correct = True

            # Always merge (oracle gives us the right candidate)
            new_rows, _ = match_rows_for_union(
                enriched, cand_df, key_col_q, key_col_c, q_row_embs, c_row_embs)
            enriched = merge_union(enriched, cand_df, align["pairs"], new_rows)

    # ── Oracle Join ──
    if gt_join_id:
        j_manifest = manifest.get(gt_join_id)
        j_col_embs = col_t_lookup.get(gt_join_id)

        if j_manifest and Path(j_manifest["csv_path"]).exists() and q_col_embs and j_col_embs:
            join_df = pd.read_csv(j_manifest["csv_path"])
            key_col_j = j_manifest.get("key_col_fragment")
            j_row_embs = row_t_lookup.get(gt_join_id)

            # Column alignment
            align = hungarian_align(q_col_embs, j_col_embs)

            # Classify
            key_pair = None
            if thresholds:
                pred_rel, conf, key_pair = classify_pair(align, thresholds)
                join_correct = (pred_rel == "join")
            else:
                join_correct = True
                # Derive key_pair from highest similarity
                if len(align["sims"]) > 0:
                    best_idx = int(np.argmax(align["sims"]))
                    key_pair = (align["pairs"][best_idx][0], align["pairs"][best_idx][1])

            # If classifier didn't find key_pair, use best aligned column
            if key_pair is None and len(align["sims"]) > 0:
                best_idx = int(np.argmax(align["sims"]))
                key_pair = (align["pairs"][best_idx][0], align["pairs"][best_idx][1])

            # Match rows and merge
            row_mapping, _ = match_rows_for_join(
                enriched, join_df, key_pair, key_col_q, key_col_j,
                q_row_embs, j_row_embs)
            enriched, new_cols = merge_join(
                enriched, join_df, align["pairs"], key_pair, row_mapping)

    # Evaluate
    f1 = cell_f1(enriched, parent_df)

    return {
        "cell_f1": f1,
        "tier": tier,
        "union_correct": union_correct,
        "join_correct": join_correct,
        "enriched_shape": list(enriched.shape),
        "parent_shape": list(parent_df.shape),
    }


# ── Per-Combination Evaluation ────────────────────────────────────

def evaluate_combination(col_model, row_model, query_tasks, manifest, parents, splits):
    """Evaluate one col×row combination with oracle Stage 1."""
    combo = f"{col_model}__{row_model}"
    print(f"\n  Combination: {combo}")
    t0 = time.time()

    # Load embeddings
    print(f"    Loading column embeddings ({col_model})...")
    col_q_lookup, col_t_lookup = load_column_embeddings(col_model)
    print(f"    Loading row embeddings ({row_model})...")
    row_q_lookup, row_t_lookup = load_row_embeddings(row_model)

    # Load calibrated thresholds
    thresholds = load_calibrated_thresholds(col_model)
    if thresholds:
        print(f"    Using calibrated thresholds")
    else:
        print(f"    WARN: No calibrated thresholds, using oracle relation labels")

    results = {"col_model": col_model, "row_model": row_model, "splits": {}}

    for split in splits:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue

        # Deduplicate: since tiers produce identical enrichment (same seed),
        # only process tier 0 for each parent and copy results.
        # Actually — with oracle candidates, each tier has DIFFERENT GT candidates,
        # so we must process all queries.
        print(f"    Split: {split} ({len(split_tasks)} queries)")

        all_f1s = []
        tier_f1s = defaultdict(list)
        n_union_correct = 0
        n_join_correct = 0
        n_processed = 0

        for qt in split_tasks:
            result = oracle_process_query(
                qt, manifest, parents, thresholds,
                col_q_lookup, col_t_lookup,
                row_q_lookup, row_t_lookup)

            if result is None:
                continue

            all_f1s.append(result["cell_f1"])
            tier_f1s[result["tier"]].append(result["cell_f1"])
            if result["union_correct"]:
                n_union_correct += 1
            if result["join_correct"]:
                n_join_correct += 1
            n_processed += 1

            if n_processed % 500 == 0:
                print(f"      {n_processed}/{len(split_tasks)} queries...")

        if not all_f1s:
            continue

        mean_f1 = float(np.mean(all_f1s))
        split_result = {
            "n_queries": len(split_tasks),
            "n_evaluated": n_processed,
            "cell_f1": mean_f1,
            "union_classification_acc": n_union_correct / n_processed if n_processed > 0 else 0,
            "join_classification_acc": n_join_correct / n_processed if n_processed > 0 else 0,
            "per_tier": {},
        }

        for tier in sorted(tier_f1s.keys()):
            split_result["per_tier"][tier] = {
                "cell_f1": float(np.mean(tier_f1s[tier])),
                "n_queries": len(tier_f1s[tier]),
            }

        results["splits"][split] = split_result

        print(f"      CellF1={mean_f1:.4f}  UnionClassAcc={split_result['union_classification_acc']:.4f}  "
              f"JoinClassAcc={split_result['join_classification_acc']:.4f}")
        for tier in sorted(tier_f1s.keys()):
            t_f1 = float(np.mean(tier_f1s[tier]))
            print(f"      Tier {tier}: CellF1={t_f1:.4f}")

    # Save
    metrics_dir = HEATMAP_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{combo}.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    return results


# ── Aggregation ──────────────────────────────────────────────────

def aggregate():
    """Combine all per-combination metrics into heatmap matrix."""
    print("Aggregating oracle heatmap metrics...")
    metrics_dir = HEATMAP_ROOT / "metrics"

    rows = []
    for col in COLUMN_MODELS:
        for row in ROW_MODELS:
            combo = f"{col}__{row}"
            path = metrics_dir / f"{combo}.json"
            if not path.exists():
                print(f"  Missing: {combo}")
                continue
            data = json.loads(path.read_text())
            entry = {"col_model": col, "row_model": row}
            for split in ["dev", "test", "train"]:
                sm = data.get("splits", {}).get(split, {})
                entry[f"cell_f1_{split}"] = sm.get("cell_f1")
            rows.append(entry)

    if not rows:
        print("  No metrics found!")
        return

    df = pd.DataFrame(rows)

    # Save full summary
    df.to_csv(HEATMAP_ROOT / "heatmap_summary.csv", index=False)

    # Build 7×4 matrix for dev and test
    for split in ["dev", "test"]:
        col_name = f"cell_f1_{split}"
        if col_name not in df.columns:
            continue

        matrix = df.pivot(index="col_model", columns="row_model", values=col_name)
        matrix = matrix.reindex(index=COLUMN_MODELS, columns=ROW_MODELS)
        matrix.to_csv(HEATMAP_ROOT / f"heatmap_matrix_{split}.csv")
        print(f"\n  Heatmap matrix ({split}):")
        print(matrix.to_string(float_format=lambda x: f"{x:.4f}"))

    # Generate heatmap figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        for split in ["dev", "test"]:
            col_name = f"cell_f1_{split}"
            if col_name not in df.columns:
                continue

            matrix = df.pivot(index="col_model", columns="row_model", values=col_name)
            matrix = matrix.reindex(index=COLUMN_MODELS, columns=ROW_MODELS)

            MODEL_LABELS = {
                "bert": "BERT", "gte": "GTE",
                "starmie": "Starmie", "tabert": "TaBERT", "tabsketchfm": "TabSketchFM",
                "tapas": "TAPAS", "turl": "TURL",
                "dae": "DAE", "saint": "SAINT", "scarf": "SCARF",
                "subtab": "SubTab", "tabbie": "TABBIE",
                "tabicl": "TabICL", "tabpfn": "TabPFN", "tabtransformer": "TabTransformer",
                "tabular_binning": "TabularBinning", "transtab": "TransTab", "tuta": "TuTa",
                "vime": "VIME",
            }
            matrix.index = [MODEL_LABELS.get(m, m) for m in matrix.index]
            matrix.columns = [MODEL_LABELS.get(m, m) for m in matrix.columns]

            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                        vmin=0.5, vmax=0.9, linewidths=0.5,
                        cbar_kws={"label": "CellF1"}, ax=ax)
            ax.set_xlabel("Row Model")
            ax.set_ylabel("Column Model")
            ax.set_title(f"Oracle Stage 1: CellF1 ({split} split)")
            plt.tight_layout()
            fig.savefig(HEATMAP_ROOT / f"heatmap_7x4_{split}.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Saved: heatmap_7x4_{split}.pdf")

    except ImportError:
        print("  matplotlib/seaborn not available — skipping figure generation")

    print(f"\n  Outputs: {HEATMAP_ROOT}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 13: Oracle Stage 1 Heatmap")
    parser.add_argument("--col_model", type=str, help="Column model to evaluate")
    parser.add_argument("--row_model", type=str, help="Row model to evaluate")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate existing metrics into heatmap (no computation)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory for DLTE results")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory")
    args = parser.parse_args()
    resolve_paths(args)

    if args.aggregate:
        aggregate()
        return 0

    if not args.col_model or not args.row_model:
        print("ERROR: --col_model and --row_model required (or use --aggregate)")
        return 1

    print("Step 13: Oracle Stage 1 Heatmap")
    print("=" * 60)
    print(f"Column model: {args.col_model}")
    print(f"Row model: {args.row_model}")
    print(f"Splits: {args.splits}")

    print("\nLoading shared data...")
    t_load = time.time()
    query_tasks = load_query_tasks()
    manifest = load_manifest()
    parents = load_parents()
    print(f"  {len(query_tasks)} tasks, {len(manifest)} manifest, "
          f"{len(parents)} parents in {time.time() - t_load:.1f}s")

    HEATMAP_ROOT.mkdir(parents=True, exist_ok=True)

    evaluate_combination(
        args.col_model, args.row_model, query_tasks, manifest, parents, args.splits)

    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
