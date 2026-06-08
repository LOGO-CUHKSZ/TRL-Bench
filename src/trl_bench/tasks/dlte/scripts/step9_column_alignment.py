"""
Step 9: Stage 2 — Column Alignment + Join/Union Classification.

For each (query, candidate) pair from Stage 1, align columns using
Hungarian matching on column embeddings, classify the relationship
as union/join/none, and report metrics.

Usage:
    python downstream_tasks/dlte/scripts/step9_column_alignment.py
    python downstream_tasks/dlte/scripts/step9_column_alignment.py --models bert starmie
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Paths (resolved at runtime by resolve_paths()) ───────────────

PROJECT_ROOT = COL_EMB_ROOT = DATASET_ROOT = GT_ROOT = None
TABLE_MAPS_DIR = MANIFEST_PATH = STAGE1_ROOT = STAGE2_ROOT = None

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]


def derive_stage2_key(table_model, col_model):
    """Derive the Stage 2 directory key from table and column model names."""
    if table_model and table_model != col_model:
        return f"{table_model}__{col_model}"
    return col_model


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, COL_EMB_ROOT, DATASET_ROOT, GT_ROOT
    global TABLE_MAPS_DIR, MANIFEST_PATH, STAGE1_ROOT, STAGE2_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    emb_base = Path(args.embeddings_root) if getattr(args, 'embeddings_root', None) else PROJECT_ROOT / "assets" / "embeddings"
    COL_EMB_ROOT = emb_base / "column"
    data_root = Path(args.data_root) if getattr(args, 'data_root', None) else PROJECT_ROOT
    DATASET_ROOT = data_root / "datasets" / "dlte_v1"
    GT_ROOT = DATASET_ROOT / "ground_truth"
    TABLE_MAPS_DIR = GT_ROOT / "table_maps"
    MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
    STAGE1_ROOT = output_root / "stage1"
    STAGE2_ROOT = output_root / "stage2"

# ── Default threshold ranges for grid search ───────────────────────

GRID = {
    "tau_union":      np.arange(0.50, 1.01, 0.1),        # 6 values
    "tau_union_sim":  np.arange(0.70, 0.96, 0.05),       # 6 values (high: true union ~0.95)
    "tau_join_max":   np.arange(0.20, 0.61, 0.1),        # 5 values
    "tau_key_sim":    np.arange(0.75, 0.96, 0.05),       # 5 values (key col median sim ~0.88)
    "tau_match_floor": np.arange(0.70, 0.91, 0.05),      # 5 values (high: BERT baseline ~0.77)
}

DEFAULT_THRESHOLDS = {
    "tau_union": 0.80,
    "tau_union_sim": 0.85,
    "tau_join_max": 0.40,
    "tau_key_sim": 0.90,
    "tau_match_floor": 0.80,
}


# ── Data Loading ───────────────────────────────────────────────────

def load_query_tasks():
    tasks = []
    with open(GT_ROOT / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def load_manifest():
    """Load fragments manifest → dict of table_id → manifest entry."""
    lookup = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            lookup[entry["table_id"]] = entry
    return lookup


def build_gt(query_tasks):
    """Build ground truth relation labels and column alignments.

    Returns:
        gt_relations: dict[(qid, cid)] -> "union" | "join"
            (only for relevant pairs; all others are implicitly "none")
        gt_col_align: dict[(qid, cid)] -> list[(q_col, c_col)]
        gt_key_col_pair: dict[(qid, cid)] -> (q_col, c_col)  (join only)
    """
    gt_relations = {}
    gt_col_align = {}
    gt_key_col_pair = {}

    for qt in query_tasks:
        qid = qt["query_table_id"]
        seed_npz = TABLE_MAPS_DIR / f"{qid}.npz"
        if not seed_npz.exists():
            continue
        seed_parent = np.load(seed_npz)["col_parent_idx"]

        for rel in qt["relevant"]:
            cid = rel["table_id"]
            relation = rel["relation"]
            gt_relations[(qid, cid)] = relation

            cand_npz = TABLE_MAPS_DIR / f"{cid}.npz"
            if not cand_npz.exists():
                continue
            cand_parent = np.load(cand_npz)["col_parent_idx"]

            # Derive column alignment via shared parent indices
            pairs = []
            for i, sp in enumerate(seed_parent):
                if sp < 0:
                    continue
                for j, cp in enumerate(cand_parent):
                    if cp == sp:
                        pairs.append((int(i), int(j)))
                        break
            gt_col_align[(qid, cid)] = pairs

            # For join, the key is the single shared column
            if relation == "join" and len(pairs) >= 1:
                gt_key_col_pair[(qid, cid)] = pairs[0]

    return gt_relations, gt_col_align, gt_key_col_pair


# ── Core Alignment ─────────────────────────────────────────────────

def hungarian_align(query_cols, candidate_cols):
    """Perform Hungarian matching on column embeddings.

    Args:
        query_cols: dict of col_idx -> embedding vector
        candidate_cols: dict of col_idx -> embedding vector

    Returns:
        dict with pairs, n_query_cols, n_cand_cols, and raw similarities.
    """
    if not query_cols or not candidate_cols:
        return {
            "pairs": [],
            "n_query_cols": len(query_cols) if query_cols else 0,
            "n_cand_cols": len(candidate_cols) if candidate_cols else 0,
            "sims": np.array([], dtype=np.float32),
        }

    q_keys = sorted(query_cols.keys())
    c_keys = sorted(candidate_cols.keys())
    q_vecs = np.array([query_cols[k] for k in q_keys], dtype=np.float32)
    c_vecs = np.array([candidate_cols[k] for k in c_keys], dtype=np.float32)

    # L2-normalize
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c_vecs, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    c_norms[c_norms == 0] = 1.0
    q_normed = q_vecs / q_norms
    c_normed = c_vecs / c_norms

    # Pairwise cosine similarity
    sim_matrix = q_normed @ c_normed.T  # (n_q, n_c)

    # Hungarian assignment on cost = 1 - sim
    cost = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    sims = sim_matrix[row_ind, col_ind]

    # Build pairs with original column indices
    pairs = []
    for r, c, s in zip(row_ind, col_ind, sims):
        pairs.append((int(q_keys[r]), int(c_keys[c]), float(s)))

    return {
        "pairs": pairs,
        "n_query_cols": len(q_keys),
        "n_cand_cols": len(c_keys),
        "sims": sims,  # raw matched similarities (for vectorized tuning)
    }


# ── Classification ─────────────────────────────────────────────────

def classify_pair(align, thresholds):
    """Classify a (query, candidate) pair based on alignment statistics.

    Returns (relation, confidence, predicted_key_pair_or_None).
    """
    tau = thresholds
    sims = align["sims"]
    n_q = align["n_query_cols"]

    # Count matched pairs above floor
    matched_mask = sims >= tau["tau_match_floor"]
    n_matched = int(matched_mask.sum())
    match_ratio = n_matched / n_q if n_q > 0 else 0.0
    mean_sim = float(sims[matched_mask].mean()) if n_matched > 0 else 0.0
    max_sim = float(sims.max()) if len(sims) > 0 else 0.0

    # Union check
    if match_ratio >= tau["tau_union"] and mean_sim >= tau["tau_union_sim"]:
        return "union", float(min(match_ratio, mean_sim)), None

    # Join check
    if (match_ratio <= tau["tau_join_max"]
            and max_sim >= tau["tau_key_sim"]
            and 1 <= n_matched <= 3):
        # Key column = pair with highest similarity
        best_idx = int(np.argmax(sims))
        key_pair = (align["pairs"][best_idx][0], align["pairs"][best_idx][1])
        return "join", float(max_sim), key_pair

    # None
    return "none", float(1.0 - max_sim), None


# ── Vectorized Threshold Tuning ────────────────────────────────────

def tune_thresholds(dev_alignments, dev_gt_labels, dev_pair_keys):
    """Grid search over thresholds to maximize macro F1 on dev.

    We optimize macro F1 (average of per-class F1) rather than accuracy
    because "none" dominates (~98% of pairs). Accuracy optimization
    would just learn to never predict union/join.

    Args:
        dev_alignments: list of alignment dicts (one per pair)
        dev_gt_labels: np.array of int labels (0=none, 1=union, 2=join)
        dev_pair_keys: list of (qid, cid) for each pair

    Returns:
        best_thresholds dict, calibration report dict
    """
    N = len(dev_alignments)
    print(f"    Tuning thresholds on {N} dev pairs...")
    t0 = time.time()

    # Precompute per-pair statistics for each tau_match_floor
    n_q_cols = np.array([a["n_query_cols"] for a in dev_alignments], dtype=np.float32)
    all_max_sims = np.array([float(a["sims"].max()) if len(a["sims"]) > 0 else 0.0
                             for a in dev_alignments], dtype=np.float32)

    # For each floor value, precompute n_matched, match_ratio, mean_sim
    floor_data = {}
    for floor in GRID["tau_match_floor"]:
        n_matched = np.zeros(N, dtype=np.float32)
        mean_sims = np.zeros(N, dtype=np.float32)
        for i, a in enumerate(dev_alignments):
            mask = a["sims"] >= floor
            nm = mask.sum()
            n_matched[i] = nm
            mean_sims[i] = a["sims"][mask].mean() if nm > 0 else 0.0
        match_ratio = n_matched / np.maximum(n_q_cols, 1.0)
        floor_data[float(floor)] = {
            "n_matched": n_matched,
            "match_ratio": match_ratio,
            "mean_sim": mean_sims,
        }

    # Grid search — optimize macro F1
    best_score = -1.0
    best_thresholds = dict(DEFAULT_THRESHOLDS)
    n_combos = 0

    # Precompute GT masks
    gt_union = (dev_gt_labels == 1)
    gt_join = (dev_gt_labels == 2)
    gt_none = (dev_gt_labels == 0)
    n_gt_union = gt_union.sum()
    n_gt_join = gt_join.sum()
    n_gt_none = gt_none.sum()

    for floor in GRID["tau_match_floor"]:
        fd = floor_data[float(floor)]
        mr = fd["match_ratio"]
        ms = fd["mean_sim"]
        nm = fd["n_matched"]

        for tau_u in GRID["tau_union"]:
            for tau_us in GRID["tau_union_sim"]:
                union_mask = (mr >= tau_u) & (ms >= tau_us)

                for tau_jm in GRID["tau_join_max"]:
                    for tau_ks in GRID["tau_key_sim"]:
                        join_mask = ((mr <= tau_jm)
                                     & (all_max_sims >= tau_ks)
                                     & (nm >= 1) & (nm <= 3)
                                     & ~union_mask)
                        none_mask = ~union_mask & ~join_mask

                        # Per-class F1
                        f1s = []
                        for pred_mask, gt_mask, n_gt in [
                            (union_mask, gt_union, n_gt_union),
                            (join_mask, gt_join, n_gt_join),
                            (none_mask, gt_none, n_gt_none),
                        ]:
                            tp = (pred_mask & gt_mask).sum()
                            n_pred = pred_mask.sum()
                            prec = tp / n_pred if n_pred > 0 else 0
                            rec = tp / n_gt if n_gt > 0 else 0
                            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                            f1s.append(f1)

                        macro_f1 = np.mean(f1s)
                        n_combos += 1

                        if macro_f1 > best_score:
                            best_score = macro_f1
                            best_thresholds = {
                                "tau_union": float(tau_u),
                                "tau_union_sim": float(tau_us),
                                "tau_join_max": float(tau_jm),
                                "tau_key_sim": float(tau_ks),
                                "tau_match_floor": float(floor),
                            }

    elapsed = time.time() - t0
    print(f"    Grid search: {n_combos} combos in {elapsed:.1f}s, "
          f"best macro_F1={best_score:.4f}")

    calibration = {
        "best_thresholds": best_thresholds,
        "best_macro_f1": float(best_score),
        "n_combinations": n_combos,
        "elapsed_seconds": round(elapsed, 1),
    }
    return best_thresholds, calibration


# ── Metrics ────────────────────────────────────────────────────────

def compute_stage2_metrics(predictions, gt_relations, gt_col_align,
                           gt_key_col_pair, query_tasks, split,
                           topk=None, stage1_entries=None):
    """Compute Stage 2 metrics for one split.

    Args:
        predictions: dict[(qid, cid)] -> {relation_pred, key_pair, align_pairs}
        gt_relations: dict[(qid, cid)] -> "union"|"join" (missing = "none")
        gt_col_align: dict[(qid, cid)] -> list of (q_col, c_col)
        gt_key_col_pair: dict[(qid, cid)] -> (q_col, c_col)
        query_tasks: full list of query tasks
        split: "dev"|"test"|"train"
        topk: if set, only evaluate the top-k candidates per query
              (ranked by Stage 1 score). Requires stage1_entries.
        stage1_entries: list of Stage 1 result entries with ranked candidates.
              Required when topk is not None.
    """
    split_tasks = [qt for qt in query_tasks if qt["split"] == split]
    if not split_tasks:
        return None

    # Build the set of (qid, cid) pairs to evaluate.  When topk is specified,
    # only include the top-k candidates per query based on Stage 1 ranking.
    if topk is not None and stage1_entries is not None:
        eligible_pairs = set()
        for entry in stage1_entries:
            qid = entry["query_table_id"]
            for cand in entry["candidates"][:topk]:
                eligible_pairs.add((qid, cand["table_id"]))
    else:
        eligible_pairs = None  # no filtering

    # Accumulators
    relation_correct = 0
    relation_total = 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    key_col_correct = 0
    key_col_total = 0
    align_f1_sum = 0.0
    align_f1_count = 0

    # Per-tier accumulators
    tier_rel_correct = defaultdict(int)
    tier_rel_total = defaultdict(int)
    tier_key_correct = defaultdict(int)
    tier_key_total = defaultdict(int)
    tier_align_f1_sum = defaultdict(float)
    tier_align_f1_count = defaultdict(int)

    # Index predictions by qid for fast lookup
    preds_by_qid = defaultdict(list)
    for (qid, cid), pred in predictions.items():
        if eligible_pairs is not None and (qid, cid) not in eligible_pairs:
            continue
        preds_by_qid[qid].append((cid, pred))

    for qt in split_tasks:
        qid = qt["query_table_id"]
        tier = qt["noise_tier"]

        for cid, pred in preds_by_qid.get(qid, []):
            pred_rel = pred["relation_pred"]
            gt_rel = gt_relations.get((qid, cid), "none")

            # RelationAcc
            correct = pred_rel == gt_rel
            relation_total += 1
            tier_rel_total[tier] += 1
            if correct:
                relation_correct += 1
                tier_rel_correct[tier] += 1

            # Per-class P/R/F1
            per_class_tp[gt_rel] += int(correct)
            if not correct:
                per_class_fp[pred_rel] += 1
                per_class_fn[gt_rel] += 1

            # KeyColAcc (join pairs only, correctly classified)
            if gt_rel == "join" and pred_rel == "join":
                gt_key = gt_key_col_pair.get((qid, cid))
                pred_key = pred.get("key_pair")
                if gt_key is not None and pred_key is not None:
                    key_col_total += 1
                    tier_key_total[tier] += 1
                    if pred_key == gt_key:
                        key_col_correct += 1
                        tier_key_correct[tier] += 1

            # ColAlignF1 (union pairs only, correctly classified)
            if gt_rel == "union" and pred_rel == "union":
                gt_pairs_set = set(gt_col_align.get((qid, cid), []))
                pred_pairs_set = set(
                    (p[0], p[1]) for p in pred.get("align_pairs", [])
                    if p[2] >= pred.get("tau_match_floor", 0.2)
                )
                if gt_pairs_set and pred_pairs_set:
                    tp = len(gt_pairs_set & pred_pairs_set)
                    prec = tp / len(pred_pairs_set) if pred_pairs_set else 0
                    rec = tp / len(gt_pairs_set) if gt_pairs_set else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    align_f1_sum += f1
                    align_f1_count += 1
                    tier_align_f1_sum[tier] += f1
                    tier_align_f1_count[tier] += 1

    # Build per-class metrics
    per_class = {}
    for cls in ["union", "join", "none"]:
        tp = per_class_tp[cls]
        fp = per_class_fp[cls]
        fn = per_class_fn[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    metrics = {
        "split": split,
        "n_queries": len(split_tasks),
        "n_pairs": relation_total,
        "relation_acc": round(relation_correct / relation_total, 4) if relation_total > 0 else 0,
        "relation_per_class": per_class,
        "key_col_acc": round(key_col_correct / key_col_total, 4) if key_col_total > 0 else 0,
        "key_col_total": key_col_total,
        "col_align_f1_union": round(align_f1_sum / align_f1_count, 4) if align_f1_count > 0 else 0,
        "col_align_f1_count": align_f1_count,
        "per_tier": {},
    }

    for tier in sorted(tier_rel_total.keys()):
        t_total = tier_rel_total[tier]
        metrics["per_tier"][tier] = {
            "relation_acc": round(tier_rel_correct[tier] / t_total, 4) if t_total > 0 else 0,
            "key_col_acc": round(tier_key_correct[tier] / tier_key_total[tier], 4) if tier_key_total.get(tier, 0) > 0 else 0,
            "col_align_f1_union": round(tier_align_f1_sum[tier] / tier_align_f1_count[tier], 4) if tier_align_f1_count.get(tier, 0) > 0 else 0,
            "n_pairs": t_total,
        }

    return metrics


# ── Per-Model Processing ───────────────────────────────────────────

def process_model(model_name, query_tasks, gt_relations, gt_col_align,
                  gt_key_col_pair, topk_values, table_model=None):
    """Process one column model end-to-end."""
    print(f"\n  Model: {model_name}")
    t0 = time.time()

    # 1. Load column embeddings
    print("    Loading column embeddings...")
    t_load = time.time()
    with open(COL_EMB_ROOT / model_name / "dlte_v1_queries.pkl", "rb") as f:
        queries_pkl = pickle.load(f)
    with open(COL_EMB_ROOT / model_name / "dlte_v1_targets.pkl", "rb") as f:
        targets_pkl = pickle.load(f)
    with open(COL_EMB_ROOT / model_name / "ckan_subset.pkl", "rb") as f:
        ckan_pkl = pickle.load(f)
    print(f"    Loaded in {time.time() - t_load:.1f}s")

    query_lookup = {e["table_id"]: e["column_embeddings"] for e in queries_pkl}
    lake_lookup = {e["table_id"]: e["column_embeddings"] for e in targets_pkl}
    lake_lookup.update({e["table_id"]: e["column_embeddings"] for e in ckan_pkl})
    del queries_pkl, targets_pkl, ckan_pkl

    # 2. Load Stage 1 results (top-100)
    s1_model = table_model if table_model else model_name
    stage1_path = STAGE1_ROOT / s1_model / "topk_100.jsonl"
    stage1_entries = []
    with open(stage1_path) as f:
        for line in f:
            stage1_entries.append(json.loads(line.strip()))
    print(f"    Loaded {len(stage1_entries)} Stage 1 entries")

    # 3. Compute Hungarian alignment for ALL pairs
    print("    Computing Hungarian alignments...")
    t_align = time.time()
    # Store: pair_key -> alignment dict
    all_alignments = {}
    pair_keys_ordered = []
    n_missing = 0

    for i, entry in enumerate(stage1_entries):
        qid = entry["query_table_id"]
        q_cols = query_lookup.get(qid)

        for cand in entry["candidates"]:
            cid = cand["table_id"]
            pair_key = (qid, cid)
            pair_keys_ordered.append(pair_key)

            if q_cols is None:
                n_missing += 1
                all_alignments[pair_key] = {
                    "pairs": [], "n_query_cols": 0, "n_cand_cols": 0,
                    "sims": np.array([], dtype=np.float32),
                    "stage1_score": cand["score"],
                }
                continue

            c_cols = lake_lookup.get(cid)
            if c_cols is None:
                n_missing += 1
                all_alignments[pair_key] = {
                    "pairs": [], "n_query_cols": len(q_cols), "n_cand_cols": 0,
                    "sims": np.array([], dtype=np.float32),
                    "stage1_score": cand["score"],
                }
                continue

            align = hungarian_align(q_cols, c_cols)
            align["stage1_score"] = cand["score"]
            all_alignments[pair_key] = align

        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{len(stage1_entries)} queries aligned...")

    print(f"    Aligned {len(all_alignments)} pairs in {time.time() - t_align:.1f}s "
          f"({n_missing} missing embeddings)")

    # 4. Tune thresholds on dev
    dev_task_ids = {qt["query_table_id"] for qt in query_tasks if qt["split"] == "dev"}
    dev_pairs = [(k, all_alignments[k]) for k in pair_keys_ordered
                 if k[0] in dev_task_ids]
    dev_pair_keys = [p[0] for p in dev_pairs]
    dev_aligns = [p[1] for p in dev_pairs]

    # Build GT labels: 0=none, 1=union, 2=join
    label_map = {"none": 0, "union": 1, "join": 2}
    dev_gt_labels = np.array([
        label_map.get(gt_relations.get(k, "none"), 0)
        for k in dev_pair_keys
    ], dtype=np.int32)

    best_thresholds, calibration = tune_thresholds(dev_aligns, dev_gt_labels,
                                                    dev_pair_keys)

    # 5. Classify ALL pairs
    print("    Classifying all pairs...")
    predictions = {}
    for pair_key in pair_keys_ordered:
        align = all_alignments[pair_key]
        rel, conf, key_pair = classify_pair(align, best_thresholds)
        predictions[pair_key] = {
            "relation_pred": rel,
            "relation_conf": conf,
            "key_pair": key_pair,
            "align_pairs": align["pairs"],
            "tau_match_floor": best_thresholds["tau_match_floor"],
        }

    # Count predictions
    pred_counts = defaultdict(int)
    for p in predictions.values():
        pred_counts[p["relation_pred"]] += 1
    print(f"    Predictions: union={pred_counts['union']}, "
          f"join={pred_counts['join']}, none={pred_counts['none']}")

    # 6. Compute metrics
    stage2_key = derive_stage2_key(table_model, model_name)
    results_dir = STAGE2_ROOT / stage2_key
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save calibration
    calibration["model"] = model_name
    calibration["n_dev_queries"] = len(dev_task_ids)
    calibration["n_dev_pairs"] = len(dev_pairs)
    with open(results_dir / "calibration_dev.json", "w") as f:
        json.dump(calibration, f, indent=2)

    max_k = max(topk_values)
    for split in ["dev", "test", "train"]:
        for k in topk_values:
            metrics = compute_stage2_metrics(
                predictions, gt_relations, gt_col_align, gt_key_col_pair,
                query_tasks, split, topk=k, stage1_entries=stage1_entries)
            if metrics is None:
                continue

            with open(results_dir / f"metrics_{split}_topk_{k}.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Print summary for dev at max topk
            if split == "dev" and k == max_k:
                print(f"    Dev metrics (topk={k}): RelationAcc={metrics['relation_acc']:.4f}  "
                      f"KeyColAcc={metrics['key_col_acc']:.4f}  "
                      f"ColAlignF1_union={metrics['col_align_f1_union']:.4f}")
                for cls in ["union", "join", "none"]:
                    c = metrics["relation_per_class"][cls]
                    print(f"      {cls:>5}: P={c['precision']:.3f} R={c['recall']:.3f} "
                          f"F1={c['f1']:.3f} (n={c['support']})")

    # 7. Write aligned+classified output
    for k in topk_values:
        output_entries = []
        for entry in stage1_entries:
            qid = entry["query_table_id"]
            candidates = []
            for cand in entry["candidates"][:k]:
                cid = cand["table_id"]
                pair_key = (qid, cid)
                pred = predictions.get(pair_key, {})
                align = all_alignments.get(pair_key, {})

                # Format pairs for output (drop raw sims array)
                align_pairs = align.get("pairs", [])
                cand_out = {
                    "table_id": cid,
                    "stage1_score": cand["score"],
                    "alignment": {
                        "pairs": [[p[0], p[1], round(p[2], 4)] for p in align_pairs],
                        "n_query_cols": align.get("n_query_cols", 0),
                        "n_cand_cols": align.get("n_cand_cols", 0),
                    },
                    "relation_pred": pred.get("relation_pred", "none"),
                    "relation_conf": round(pred.get("relation_conf", 0), 4),
                }
                if pred.get("key_pair") is not None:
                    cand_out["key_pair"] = list(pred["key_pair"])
                candidates.append(cand_out)
            output_entries.append({
                "query_table_id": qid,
                "candidates": candidates,
            })

        with open(results_dir / f"aligned_classified_topk_{k}.jsonl", "w") as f:
            for e in output_entries:
                f.write(json.dumps(e) + "\n")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return True


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Column Alignment + Join/Union Classification")
    parser.add_argument("--models", nargs="+", default=COLUMN_MODELS)
    parser.add_argument("--topk", nargs="+", type=int, default=[100])
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root for DLTE outputs (default: {project_root}/results/evaluation/dlte)")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root (default: derived from script location)")
    parser.add_argument("--table_model", type=str, default=None,
                        help="Table model for Stage 1 retrieval (default: same as --models)")
    parser.add_argument("--embeddings_root", type=str, default=None,
                        help="Embeddings root (default: {project_root}/embeddings)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Data root containing 'datasets/dlte_v1/' (default: {project_root})")
    args = parser.parse_args()

    resolve_paths(args)

    print("Step 9: Stage 2 — Column Alignment + Join/Union Classification")
    print("=" * 60)

    # Load ground truth
    query_tasks = load_query_tasks()
    print(f"Loaded {len(query_tasks)} query tasks")

    print("Building ground truth alignments...")
    t_gt = time.time()
    gt_relations, gt_col_align, gt_key_col_pair = build_gt(query_tasks)
    print(f"  GT: {len(gt_relations)} relevant pairs, "
          f"{len(gt_col_align)} col alignments, "
          f"{len(gt_key_col_pair)} key col pairs "
          f"({time.time() - t_gt:.1f}s)")

    succeeded = 0
    for model in args.models:
        try:
            if process_model(model, query_tasks, gt_relations, gt_col_align,
                             gt_key_col_pair, args.topk,
                             table_model=args.table_model):
                succeeded += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Processed {succeeded}/{len(args.models)} models")
    print(f"Results: {STAGE2_ROOT}")
    print(f"{'='*60}")

    return 0 if succeeded == len(args.models) else 1


if __name__ == "__main__":
    sys.exit(main())
