"""
Step 10: Stage 3 — Row Matching + Merge + Evaluation.

For each query, picks the best union and best join candidate from Stage 2,
matches rows (CSLS-normalized reciprocal nearest-neighbor matching), and produces
enriched DataFrames with provenance metadata.  By default, evaluation metrics
(CellF1, region recall, etc.) are computed inline from in-memory DataFrames,
eliminating the need to write enriched CSVs to disk.

Usage:
    python downstream_tasks/dlte/scripts/step10_row_matching.py
    python downstream_tasks/dlte/scripts/step10_row_matching.py --col_models bert --row_models tabicl tuta
    python downstream_tasks/dlte/scripts/step10_row_matching.py --col_models bert --row_models tabicl --splits dev
    python downstream_tasks/dlte/scripts/step10_row_matching.py --save-enriched  # also write enriched CSVs
    python downstream_tasks/dlte/scripts/step10_row_matching.py --skip-evaluation  # merge only, no metrics
"""

import argparse
import json
import pickle
import sys
import time

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Paths (resolved at runtime by resolve_paths()) ───────────────

PROJECT_ROOT = DATA_ROOT = DATASET_ROOT = ROW_EMB_ROOT = MANIFEST_PATH = None
GT_ROOT = STAGE2_ROOT = STAGE3_ROOT = ENRICHED_ROOT = None
PARENTS_PATH = TABLE_MAPS_DIR = STAGE1_ROOT = METRICS_ROOT = None

COLUMN_MODELS = ["bert", "gte", "starmie", "tabbie", "tabert", "tabsketchfm", "tapas", "turl"]
ROW_MODELS = [
    "bert", "dae", "gte", "saint", "scarf", "subtab",
    "tabbie", "tabicl", "tabpfn", "tabtransformer", "tabular_binning",
    "transtab", "tuta", "vime",
]
EPS = 1e-12


def derive_stage2_key(table_model, col_model):
    """Derive the Stage 2 directory key from table and column model names."""
    if table_model and table_model != col_model:
        return f"{table_model}__{col_model}"
    return col_model


def resolve_paths(args):
    """Resolve project root and output paths from CLI args."""
    global PROJECT_ROOT, DATA_ROOT, DATASET_ROOT, ROW_EMB_ROOT, MANIFEST_PATH
    global GT_ROOT, STAGE2_ROOT, STAGE3_ROOT, ENRICHED_ROOT
    global PARENTS_PATH, TABLE_MAPS_DIR, STAGE1_ROOT, METRICS_ROOT
    PROJECT_ROOT = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"
    DATA_ROOT = Path(args.data_root) if getattr(args, 'data_root', None) else PROJECT_ROOT
    DATASET_ROOT = DATA_ROOT / "datasets" / "dlte_v1"
    emb_base = Path(args.embeddings_root) if getattr(args, 'embeddings_root', None) else PROJECT_ROOT / "assets" / "embeddings"
    ROW_EMB_ROOT = emb_base / "row"
    MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
    GT_ROOT = DATASET_ROOT / "ground_truth"
    STAGE1_ROOT = output_root / "stage1"
    STAGE2_ROOT = output_root / "stage2"
    STAGE3_ROOT = output_root / "stage3"
    ENRICHED_ROOT = output_root / "enriched"
    PARENTS_PATH = DATASET_ROOT / "manifests" / "parents_filtered.jsonl"
    TABLE_MAPS_DIR = GT_ROOT / "table_maps"
    METRICS_ROOT = output_root / "metrics"


# ── Data Loading ───────────────────────────────────────────────────

def _resolve_csv_path(entry):
    """Resolve relative csv_path entries against DATA_ROOT (in-place).

    Parents_filtered.jsonl stores csv_path as ``"datasets/<source>/tables/..."``
    which the DLTE stager materializes at ``<DATA_ROOT>/datasets/<source>/...``.
    """
    p = Path(entry["csv_path"])
    if not p.is_absolute():
        entry["csv_path"] = str(DATA_ROOT / p)
    return entry


def load_manifest():
    """Load fragments manifest -> dict of table_id -> manifest entry."""
    lookup = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = _resolve_csv_path(json.loads(line.strip()))
            lookup[entry["table_id"]] = entry
    return lookup


def load_query_tasks():
    tasks = []
    with open(GT_ROOT / "query_tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


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


def load_stage2_results(col_model, table_model=None):
    """Load Stage 2 aligned+classified results -> list of entries."""
    stage2_key = derive_stage2_key(table_model, col_model)
    path = STAGE2_ROOT / stage2_key / "aligned_classified_topk_100.jsonl"
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def load_row_embeddings(row_model):
    """Load row embeddings -> (query_lookup, target_lookup).

    Each lookup: table_id -> np.ndarray(n_rows, dim)
    """
    q_path = ROW_EMB_ROOT / row_model / "dlte_v1_queries.pkl"
    t_path = ROW_EMB_ROOT / row_model / "dlte_v1_targets.pkl"

    with open(q_path, "rb") as f:
        q_pkl = pickle.load(f)
    with open(t_path, "rb") as f:
        t_pkl = pickle.load(f)

    q_lookup = {e["table_id"]: e["row_embeddings"] for e in q_pkl}
    t_lookup = {e["table_id"]: e["row_embeddings"] for e in t_pkl}
    del q_pkl, t_pkl
    return q_lookup, t_lookup


# ── Candidate Selection ───────────────────────────────────────────

def pick_best_candidates(candidates):
    """Pick best union and best join candidate by relation_conf.

    Returns (best_union, best_join) — either may be None.
    """
    best_union = None
    best_join = None
    for cand in candidates:
        if cand["relation_pred"] == "union":
            if best_union is None or cand["relation_conf"] > best_union["relation_conf"]:
                best_union = cand
        elif cand["relation_pred"] == "join":
            if best_join is None or cand["relation_conf"] > best_join["relation_conf"]:
                best_join = cand
    return best_union, best_join


# ── Row Matching (CSLS + Reciprocal NN) ──────────────────────────

@dataclass(frozen=True)
class RowMatchConfig:
    # Similarity backend
    use_csls: bool = True
    csls_k: int = 5

    # Iterative reciprocal matching
    max_iter: int = 1        # 1 = mutual-top1 only, >1 = Itermax-like

    # Local confidence filters (all scale-relative, not raw cosine thresholds)
    min_best_z: float = 1.0
    min_margin_z: float = 0.15
    max_entropy: float = 0.99

    entropy_on: str = "raw"  # {"raw", "zscore"}

    # Optional absolute floor in normalized-score space (usually leave None)
    min_score: Optional[float] = None


# Conservative for dedup: precision-first
UNION_CFG = RowMatchConfig(
    use_csls=True,
    csls_k=5,
    max_iter=3,
    min_best_z=1.00,
    min_margin_z=0.25,
    max_entropy=1.0,      # disabled: raw entropy is ~1.0 for all models
    entropy_on="raw",
    min_score=None,
)

# Slightly looser for join: recall-first
JOIN_CFG = RowMatchConfig(
    use_csls=True,
    csls_k=5,
    max_iter=10,
    min_best_z=0.75,
    min_margin_z=0.10,
    max_entropy=1.0,      # disabled: raw entropy is ~1.0 for all models
    entropy_on="raw",
    min_score=None,
)


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 0:
            return arr.reshape(0, 0)
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def _truncate_embeddings(df_like: Sequence, row_embs: np.ndarray) -> Tuple[np.ndarray, int]:
    """Keep only rows for which both the dataframe row and embedding exist."""
    emb = _as_2d_float(row_embs)
    n = min(len(df_like), emb.shape[0])
    return emb[:n], n


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L2-normalized cosine similarity."""
    a = _as_2d_float(a)
    b = _as_2d_float(b)

    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: {a.shape[1]} vs {b.shape[1]}"
        )

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm < EPS] = 1.0
    b_norm[b_norm < EPS] = 1.0

    sim = (a / a_norm) @ (b / b_norm).T
    return np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)


def csls_sim_matrix(a: np.ndarray, b: np.ndarray, k: int = 5) -> np.ndarray:
    """CSLS-normalized similarity: 2*S(i,j) - r_i - c_j.

    Discounts hub-like rows whose neighborhood is dense, making it more
    likely that a nearest neighbor is also a reciprocal nearest neighbor.
    """
    sim = cosine_sim_matrix(a, b)
    if sim.size == 0:
        return sim

    k_row = max(1, min(k, sim.shape[1]))
    k_col = max(1, min(k, sim.shape[0]))

    # mean top-k for each row
    row_topk = np.partition(sim, kth=sim.shape[1] - k_row, axis=1)[:, -k_row:]
    r = row_topk.mean(axis=1)

    # mean top-k for each column
    col_topk = np.partition(sim, kth=sim.shape[0] - k_col, axis=0)[-k_col:, :]
    c = col_topk.mean(axis=0)

    csls = 2.0 * sim - r[:, None] - c[None, :]
    return np.nan_to_num(csls, nan=0.0, posinf=0.0, neginf=0.0)


def _row_entropy(scores: np.ndarray, mode: str = "raw") -> np.ndarray:
    """Normalized entropy per row in [0, 1].  Lower => sharper neighborhood."""
    if scores.ndim != 2:
        raise ValueError("scores must be 2D")
    n_rows, n_cols = scores.shape
    if n_rows == 0:
        return np.zeros(0, dtype=np.float32)
    if n_cols <= 1:
        return np.zeros(n_rows, dtype=np.float32)

    x = scores.astype(np.float32, copy=True)

    if mode == "zscore":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std[std < EPS] = 1.0
        x = (x - mean) / std
    elif mode != "raw":
        raise ValueError(f"Unsupported entropy mode: {mode}")

    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum[p_sum < EPS] = 1.0
    p = p / p_sum

    ent = -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)
    ent = ent / np.log(n_cols)
    return ent.astype(np.float32)


def _axis_diagnostics(scores: np.ndarray, entropy_on: str = "raw") -> Dict[str, np.ndarray]:
    """Row-wise diagnostics: best index, best z-score, margin, entropy."""
    if scores.ndim != 2:
        raise ValueError("scores must be 2D")

    n_rows, n_cols = scores.shape
    if n_rows == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float32)
        return {
            "best_idx": empty_i,
            "best_score": empty_f,
            "best_z": empty_f,
            "margin_z": empty_f,
            "entropy": empty_f,
        }

    best_idx = np.argmax(scores, axis=1)
    row_ids = np.arange(n_rows)
    best_score = scores[row_ids, best_idx]

    mean = scores.mean(axis=1)
    std = scores.std(axis=1)
    denom = std.copy()
    denom[denom < EPS] = 1.0

    best_z = (best_score - mean) / denom

    if n_cols > 1:
        top2 = np.partition(scores, kth=n_cols - 2, axis=1)[:, -2:]
        top2.sort(axis=1)
        second_score = top2[:, 0]
        margin_z = (best_score - second_score) / denom
    else:
        margin_z = np.full(n_rows, np.inf, dtype=np.float32)

    entropy = _row_entropy(scores, mode=entropy_on)

    return {
        "best_idx": best_idx.astype(np.int64),
        "best_score": best_score.astype(np.float32),
        "best_z": best_z.astype(np.float32),
        "margin_z": margin_z.astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def _score_matrix(left_embs: np.ndarray, right_embs: np.ndarray,
                  cfg: RowMatchConfig) -> np.ndarray:
    if cfg.use_csls:
        return csls_sim_matrix(left_embs, right_embs, k=cfg.csls_k)
    return cosine_sim_matrix(left_embs, right_embs)


def _accept_pair(
    li: int,
    rj: int,
    scores: np.ndarray,
    left_diag: Dict[str, np.ndarray],
    right_diag: Dict[str, np.ndarray],
    cfg: RowMatchConfig,
) -> bool:
    s = scores[li, rj]

    if cfg.min_score is not None and s < cfg.min_score:
        return False
    if left_diag["best_z"][li] < cfg.min_best_z:
        return False
    if right_diag["best_z"][rj] < cfg.min_best_z:
        return False
    if left_diag["margin_z"][li] < cfg.min_margin_z:
        return False
    if right_diag["margin_z"][rj] < cfg.min_margin_z:
        return False
    if left_diag["entropy"][li] > cfg.max_entropy:
        return False
    if right_diag["entropy"][rj] > cfg.max_entropy:
        return False

    return True


def _iterated_reciprocal_matches(
    left_embs: np.ndarray,
    right_embs: np.ndarray,
    cfg: RowMatchConfig,
) -> List[Tuple[int, int]]:
    """Iterative reciprocal nearest-neighbor matching.

    Each round:
      1) compute CSLS score matrix
      2) find mutual top-1 pairs
      3) keep only pairs passing local confidence tests
      4) remove matched rows/cols and repeat
    """
    left_embs = _as_2d_float(left_embs)
    right_embs = _as_2d_float(right_embs)

    n_left = left_embs.shape[0]
    n_right = right_embs.shape[0]
    if n_left == 0 or n_right == 0:
        return []

    left_remaining = list(range(n_left))
    right_remaining = list(range(n_right))
    matches: List[Tuple[int, int]] = []

    it = 0
    while left_remaining and right_remaining and it < cfg.max_iter:
        l_sub = left_embs[left_remaining]
        r_sub = right_embs[right_remaining]

        scores = _score_matrix(l_sub, r_sub, cfg)
        left_diag = _axis_diagnostics(scores, entropy_on=cfg.entropy_on)
        right_diag = _axis_diagnostics(scores.T, entropy_on=cfg.entropy_on)

        # Mutual top-1 pairs in the current subproblem
        proposed: List[Tuple[int, int]] = []
        left_best = left_diag["best_idx"]
        right_best = right_diag["best_idx"]

        for li_sub, rj_sub in enumerate(left_best):
            if right_best[rj_sub] != li_sub:
                continue
            if _accept_pair(li_sub, rj_sub, scores, left_diag, right_diag, cfg):
                proposed.append((li_sub, rj_sub))

        if not proposed:
            break

        proposed.sort(key=lambda ij: float(scores[ij[0], ij[1]]), reverse=True)

        used_l = set()
        used_r = set()
        for li_sub, rj_sub in proposed:
            if li_sub in used_l or rj_sub in used_r:
                continue
            matches.append((left_remaining[li_sub], right_remaining[rj_sub]))
            used_l.add(li_sub)
            used_r.add(rj_sub)

        if not used_l:
            break

        left_remaining = [g for pos, g in enumerate(left_remaining)
                          if pos not in used_l]
        right_remaining = [g for pos, g in enumerate(right_remaining)
                           if pos not in used_r]
        it += 1

    return matches


def match_rows_for_union(query_df, cand_df, q_row_embs, c_row_embs):
    """Embedding-only union dedup via reciprocal CSLS matching.

    A candidate row is treated as a duplicate iff it gets matched to a
    query row by reciprocal nearest-neighbor matching with local confidence
    checks.  All unmatched candidate rows are returned as NEW rows.
    """
    if (q_row_embs is not None and c_row_embs is not None
            and len(q_row_embs) > 0 and len(c_row_embs) > 0):
        q_embs, n_q = _truncate_embeddings(query_df, q_row_embs)
        c_embs, n_c = _truncate_embeddings(cand_df, c_row_embs)

        if n_q == 0 or n_c == 0:
            return list(range(len(cand_df))), "embedding"

        # left = candidate, right = query
        dup_pairs = _iterated_reciprocal_matches(c_embs, q_embs, UNION_CFG)
        duplicate_cand_rows = {ci for ci, _ in dup_pairs}

        new_rows = [i for i in range(len(cand_df))
                    if i >= n_c or i not in duplicate_cand_rows]
        method = "embedding"
    else:
        new_rows = list(range(len(cand_df)))
        method = "no_embeddings"

    return new_rows, method


def match_rows_for_join(query_df, join_df, q_row_embs, j_row_embs):
    """Embedding-only join row matching via iterative reciprocal CSLS.

    Returns dict: {query_row_idx: join_row_idx} and method string.
    Unmatched query rows are not in the dict.
    """
    row_mapping = {}
    method = "none"

    n_with_embs = len(q_row_embs) if q_row_embs is not None else 0
    eligible_q = [i for i in range(len(query_df)) if i < n_with_embs]

    if eligible_q and q_row_embs is not None and j_row_embs is not None:
        q_embs = np.asarray(q_row_embs[eligible_q], dtype=np.float32)
        j_embs = np.asarray(j_row_embs, dtype=np.float32)

        if q_embs.shape[0] > 0 and j_embs.shape[0] > 0:
            matches = _iterated_reciprocal_matches(q_embs, j_embs, JOIN_CFG)
            row_mapping = {eligible_q[qi]: ji for qi, ji in matches}

        method = "embedding"
    elif q_row_embs is None and j_row_embs is None:
        method = "no_embeddings"

    return row_mapping, method


# ── Merge Operations ──────────────────────────────────────────────

def merge_union(query_df, cand_df, col_alignment, new_row_indices):
    """Append new rows from union candidate to query table.

    col_alignment: list of [query_col_idx, cand_col_idx, sim] from Stage 2.
    """
    if not new_row_indices:
        return query_df.copy()

    q_cols = list(query_df.columns)
    c_cols = list(cand_df.columns)

    # Build column mapping: cand_col_idx -> query_col_idx
    idx_map = {}  # cand col idx -> query col idx
    for q_idx, c_idx, _ in col_alignment:
        if q_idx < len(q_cols) and c_idx < len(c_cols):
            idx_map[c_idx] = q_idx

    # Build new rows directly using query column order
    new_data = []
    for row_i in new_row_indices:
        row_dict = {}
        for c_idx, q_idx in idx_map.items():
            row_dict[q_cols[q_idx]] = cand_df.iloc[row_i, c_idx]
        new_data.append(row_dict)

    new_rows_df = pd.DataFrame(new_data, columns=query_df.columns)
    enriched = pd.concat([query_df, new_rows_df], ignore_index=True)
    return enriched


def merge_join(enriched_df, join_df, col_alignment, key_pair, row_mapping):
    """Add new columns from join candidate to enriched table.

    col_alignment: list of [query_col_idx, cand_col_idx, sim] from Stage 2.
    key_pair: [query_key_col_idx, join_key_col_idx]
    row_mapping: {enriched_row_idx: join_row_idx}
    """
    j_cols = list(join_df.columns)

    # For join: only the key column is shared; all other join columns are new.
    # Hungarian alignment matches all columns 1-1, but in a join only the key
    # column truly overlaps — the rest are new data to be added.
    shared_j_idx = key_pair[1] if key_pair else None
    new_col_indices = [i for i in range(len(j_cols)) if i != shared_j_idx]
    new_col_names = [j_cols[i] for i in new_col_indices]

    if not new_col_names:
        return enriched_df.copy(), []

    # Deduplicate column names if they clash with existing
    final_names = []
    for name in new_col_names:
        if name in enriched_df.columns:
            name = f"{name}_join"
        final_names.append(name)

    # Add new columns as object dtype (avoids FutureWarning with mixed types)
    result = enriched_df.copy()
    for fname in final_names:
        result[fname] = pd.Series([np.nan] * len(result), dtype=object)

    # Fill matched rows
    for e_idx, j_idx in row_mapping.items():
        if e_idx < len(result) and j_idx < len(join_df):
            for orig_col_idx, fname in zip(new_col_indices, final_names):
                result.at[e_idx, fname] = join_df.iloc[j_idx, orig_col_idx]

    return result, final_names


# ── Per-Query Processing ──────────────────────────────────────────

def process_query(qid, stage2_entry, manifest, q_row_lookup, t_row_lookup, split):
    """Process one query: pick candidates, match rows, merge, return outputs."""
    candidates = stage2_entry["candidates"]
    best_union, best_join = pick_best_candidates(candidates)

    # Load query CSV
    q_manifest = manifest.get(qid)
    if q_manifest is None:
        return None, None, None
    query_df = pd.read_csv(q_manifest["csv_path"])

    provenance = {
        "query_table_id": qid,
        "split": split,
        "union_source": None,
        "join_source": None,
    }
    log_entry = {
        "query_table_id": qid,
        "split": split,
        "union_candidate": None,
        "join_candidate": None,
        "union_rows_added": 0,
        "join_cols_added": 0,
        "join_rows_matched": 0,
        "join_rows_unmatched": 0,
        "row_match_method": "none",
        "enriched_shape": list(query_df.shape),
    }

    enriched = query_df.copy()
    q_embs = q_row_lookup.get(qid)

    # ── Union Merge ──
    if best_union is not None:
        uid = best_union["table_id"]
        u_manifest = manifest.get(uid)
        log_entry["union_candidate"] = uid

        if u_manifest is not None and Path(u_manifest["csv_path"]).exists():
            cand_df = pd.read_csv(u_manifest["csv_path"])
            c_embs = t_row_lookup.get(uid)

            new_rows, u_method = match_rows_for_union(
                enriched, cand_df, q_embs, c_embs)

            enriched = merge_union(enriched, cand_df, best_union["alignment"]["pairs"], new_rows)
            log_entry["union_rows_added"] = len(new_rows)
            log_entry["row_match_method"] = u_method

            provenance["union_source"] = {
                "table_id": uid,
                "relation_conf": best_union["relation_conf"],
                "rows_added": new_rows,
            }

    # ── Join Merge ──
    # Track union-appended row info for second-pass join
    union_appended_embs = None   # embeddings of union-appended rows
    union_appended_start = len(enriched)  # index where appended rows begin

    if best_union is not None and log_entry["union_rows_added"] > 0:
        uid = log_entry["union_candidate"]
        c_embs = t_row_lookup.get(uid)
        if c_embs is not None:
            new_rows = provenance["union_source"]["rows_added"]
            valid = [i for i in new_rows if i < len(c_embs)]
            if valid:
                union_appended_embs = np.array(
                    [c_embs[i] for i in valid], dtype=np.float32)

    if best_join is not None:
        jid = best_join["table_id"]
        j_manifest = manifest.get(jid)
        log_entry["join_candidate"] = jid

        if j_manifest is not None and Path(j_manifest["csv_path"]).exists():
            join_df = pd.read_csv(j_manifest["csv_path"])
            j_embs = t_row_lookup.get(jid)
            key_pair = best_join.get("key_pair")

            row_mapping, j_method = match_rows_for_join(
                enriched, join_df, q_embs, j_embs)

            enriched, new_cols = merge_join(
                enriched, join_df, best_join["alignment"]["pairs"],
                key_pair, row_mapping)

            # ── Second-pass join: match union-appended rows to join candidate ──
            second_pass_matched = 0
            if (union_appended_embs is not None and j_embs is not None
                    and len(new_cols) > 0):
                # Exclude join rows already matched in first pass
                already_matched_j = set(row_mapping.values())
                available_j = [j for j in range(len(j_embs))
                               if j not in already_matched_j]
                if available_j:
                    avail_j_embs = np.array(
                        [j_embs[j] for j in available_j], dtype=np.float32)
                    matches_2nd = _iterated_reciprocal_matches(
                        union_appended_embs, avail_j_embs, JOIN_CFG)

                    # Fill join columns for matched union-appended rows
                    shared_j_idx = key_pair[1] if key_pair else None
                    j_cols = list(join_df.columns)
                    new_col_indices = [i for i in range(len(j_cols))
                                      if i != shared_j_idx]
                    for ui, ji_local in matches_2nd:
                        e_idx = union_appended_start + ui
                        j_idx = available_j[ji_local]
                        if e_idx < len(enriched) and j_idx < len(join_df):
                            for orig_col_idx, fname in zip(new_col_indices, new_cols):
                                enriched.at[e_idx, fname] = join_df.iloc[j_idx, orig_col_idx]
                            second_pass_matched += 1

            total_matched = len(row_mapping) + second_pass_matched
            log_entry["join_cols_added"] = len(new_cols)
            log_entry["join_rows_matched"] = total_matched
            log_entry["join_rows_unmatched"] = len(enriched) - total_matched
            log_entry["join_rows_matched_2nd_pass"] = second_pass_matched
            if log_entry["row_match_method"] == "none":
                log_entry["row_match_method"] = j_method

            provenance["join_source"] = {
                "table_id": jid,
                "relation_conf": best_join["relation_conf"],
                "cols_added": new_cols,
                "row_mapping": {str(k): v for k, v in row_mapping.items()},
                "second_pass_matched": second_pass_matched,
            }

    log_entry["enriched_shape"] = list(enriched.shape)
    provenance["enriched_shape"] = list(enriched.shape)
    return enriched, provenance, log_entry


# ── Evaluation Functions (from step11_evaluation.py) ──────────────

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
        recovered = sum((region_counter & e_cells).values())
        total = sum(region_counter.values())
        result[region_name] = {
            "recall": recovered / total if total > 0 else 0,
            "n_cells": total,
            "n_recovered": recovered,
        }

    return result


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


def evaluate_query_inline(qid, qt, enriched_df, parent_df, parent_entry):
    """Evaluate one query from an in-memory DataFrame."""
    result = {}
    cf1 = cell_f1(enriched_df, parent_df)
    result["cell_f1"] = cf1["f1"]
    result["cell_precision"] = cf1["precision"]
    result["cell_recall"] = cf1["recall"]
    result["parent_row_recall"] = parent_row_recall(enriched_df, parent_df, parent_entry)
    result["parent_col_recall"] = parent_col_recall(enriched_df, parent_df)

    seed_npz_path = TABLE_MAPS_DIR / f"{qid}.npz"
    if seed_npz_path.exists():
        seed_npz = np.load(seed_npz_path)
        gt_union = gt_join = None
        for rel in qt.get("relevant", []):
            if rel["relation"] == "union":
                gt_union = rel["table_id"]
            elif rel["relation"] == "join":
                gt_join = rel["table_id"]
        regions = region_recall(enriched_df, parent_df, seed_npz, gt_union, gt_join)
        result["region_recall"] = {k: v["recall"] for k, v in regions.items()}
        result["region_cells"] = {k: v["n_cells"] for k, v in regions.items()}
    else:
        result["region_recall"] = None

    # UJ-H: harmonic mean of union and join region recall (per-query)
    rr = result.get("region_recall")
    if rr and rr.get("union_region") is not None and rr.get("join_region") is not None:
        u, j = rr["union_region"], rr["join_region"]
        result["uj_h"] = 2 * u * j / (u + j) if (u + j) > 0 else 0.0
    else:
        result["uj_h"] = None

    result["enriched_shape"] = list(enriched_df.shape)
    result["parent_shape"] = list(parent_df.shape)
    return result


def _aggregate_split_metrics(all_results, tier_results, n_evaluated, n_total):
    """Aggregate per-query evaluation results into split-level metrics."""
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

    metrics = {
        "n_queries": n_total, "n_evaluated": n_evaluated,
        "cell_f1": agg(all_results, "cell_f1"),
        "cell_precision": agg(all_results, "cell_precision"),
        "cell_recall": agg(all_results, "cell_recall"),
        "parent_row_recall": agg(all_results, "parent_row_recall"),
        "parent_col_recall": agg(all_results, "parent_col_recall"),
        "region_recall": agg_regions(all_results),
        "uj_h": agg(all_results, "uj_h"),
        "per_tier": {},
    }
    for tier in sorted(tier_results.keys()):
        t_results = tier_results[tier]
        metrics["per_tier"][tier] = {
            "n_queries": len(t_results),
            "cell_f1": agg(t_results, "cell_f1"),
            "cell_precision": agg(t_results, "cell_precision"),
            "cell_recall": agg(t_results, "cell_recall"),
            "parent_row_recall": agg(t_results, "parent_row_recall"),
            "parent_col_recall": agg(t_results, "parent_col_recall"),
            "region_recall": agg_regions(t_results),
            "uj_h": agg(t_results, "uj_h"),
        }
    return metrics


# ── Per-Combination Processing ────────────────────────────────────

def process_combination(col_model, row_model, query_tasks, manifest, splits,
                        table_model=None, parents=None,
                        save_enriched=False, skip_evaluation=False):
    """Process one col_model x row_model combination."""
    stage2_key = derive_stage2_key(table_model, col_model)
    combo_name = f"{stage2_key}__{row_model}"
    s1_model = table_model if table_model else col_model
    print(f"\n  Combination: {combo_name}")
    t0 = time.time()

    # Load Stage 2 results
    print("    Loading Stage 2 results...")
    stage2_entries = load_stage2_results(col_model, table_model=table_model)
    stage2_by_qid = {e["query_table_id"]: e for e in stage2_entries}
    print(f"    Loaded {len(stage2_entries)} entries")

    # Load row embeddings
    print(f"    Loading row embeddings ({row_model})...")
    t_load = time.time()
    q_row_lookup, t_row_lookup = load_row_embeddings(row_model)
    print(f"    Loaded {len(q_row_lookup)} query + {len(t_row_lookup)} target "
          f"row embeddings in {time.time() - t_load:.1f}s")

    # Process each split
    task_by_qid = {qt["query_table_id"]: qt for qt in query_tasks}
    total_queries = 0
    total_union = 0
    total_join = 0
    split_metrics_all = {}
    per_query_by_split = {}

    for split in splits:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue

        print(f"    Split: {split} ({len(split_tasks)} queries)")

        # Create output dirs
        if save_enriched:
            enriched_dir = ENRICHED_ROOT / combo_name / split
            enriched_dir.mkdir(parents=True, exist_ok=True)
        stage3_dir = STAGE3_ROOT / combo_name
        stage3_dir.mkdir(parents=True, exist_ok=True)

        merge_log = []
        n_processed = 0
        n_union = 0
        n_join = 0
        split_eval_results = []
        tier_eval_results = defaultdict(list)

        for qt in split_tasks:
            qid = qt["query_table_id"]
            stage2_entry = stage2_by_qid.get(qid)
            if stage2_entry is None:
                continue

            enriched, provenance, log_entry = process_query(
                qid, stage2_entry, manifest, q_row_lookup, t_row_lookup, split)

            if enriched is None:
                continue

            # Optionally write enriched CSV
            if save_enriched:
                csv_path = enriched_dir / f"{qid}.enriched.csv"
                enriched.to_csv(csv_path, index=False)
                prov_path = enriched_dir / f"{qid}.provenance.json"
                with open(prov_path, "w") as f:
                    json.dump(provenance, f, indent=2)

            # Evaluate in-memory
            if not skip_evaluation and parents is not None:
                qt_entry = task_by_qid.get(qid)
                if qt_entry:
                    parent_id = qt_entry["parent_id"]
                    parent_entry = parents.get(parent_id)
                    if parent_entry and Path(parent_entry["csv_path"]).exists():
                        parent_df = load_parent_csv(parent_entry)
                        eval_result = evaluate_query_inline(
                            qid, qt_entry, enriched, parent_df, parent_entry)
                        if eval_result is not None:
                            eval_result["tier"] = qt_entry["noise_tier"]
                            eval_result["qid"] = qid
                            eval_result["parent_id"] = parent_id
                            eval_result["source"] = parent_id.split("__", 1)[0]
                            split_eval_results.append(eval_result)
                            tier_eval_results[qt_entry["noise_tier"]].append(eval_result)

            merge_log.append(log_entry)
            n_processed += 1
            if log_entry["union_candidate"]:
                n_union += 1
            if log_entry["join_candidate"]:
                n_join += 1

            if n_processed % 500 == 0:
                print(f"      {n_processed}/{len(split_tasks)} queries processed...")

        # Write merge log
        log_path = stage3_dir / f"merge_log_{split}.jsonl"
        with open(log_path, "w") as f:
            for entry in merge_log:
                f.write(json.dumps(entry) + "\n")

        total_queries += n_processed
        total_union += n_union
        total_join += n_join

        # Print split summary
        avg_rows_added = np.mean([e["union_rows_added"] for e in merge_log]) if merge_log else 0
        avg_cols_added = np.mean([e["join_cols_added"] for e in merge_log]) if merge_log else 0
        avg_join_matched = np.mean([e["join_rows_matched"] for e in merge_log
                                     if e["join_candidate"]]) if n_join else 0
        print(f"      Processed: {n_processed}, union_merges: {n_union}, join_merges: {n_join}")
        print(f"      Avg rows added (union): {avg_rows_added:.1f}, "
              f"avg cols added (join): {avg_cols_added:.1f}, "
              f"avg rows matched (join): {avg_join_matched:.1f}")

        if not skip_evaluation and parents is not None and split_eval_results:
            split_metrics_all[split] = _aggregate_split_metrics(
                split_eval_results, tier_eval_results,
                len(split_eval_results), len(split_tasks))
            per_query_by_split[split] = split_eval_results
            sm = split_metrics_all[split]
            _f = lambda v: f"{v:.4f}" if v is not None else "N/A"
            rr = sm.get("region_recall", {})
            print(f"      CellF1={_f(sm['cell_f1'])}  UJ-H={_f(sm.get('uj_h'))}  "
                  f"union={_f(rr.get('union_region'))}  join={_f(rr.get('join_region'))}  "
                  f"hard={_f(rr.get('hard_region'))}")

    elapsed = time.time() - t0
    print(f"    Done: {total_queries} queries, "
          f"{total_union} unions, {total_join} joins in {elapsed:.1f}s")

    # Write metrics to disk
    if not skip_evaluation and parents is not None and split_metrics_all:
        metrics_dir = METRICS_ROOT / combo_name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Consolidate Stage 1 metrics
        stage1_data = {}
        for split in splits:
            for k in [10, 50, 100]:
                path = STAGE1_ROOT / s1_model / f"metrics_{split}_topk_{k}.json"
                if path.exists():
                    stage1_data[f"{split}_topk_{k}"] = json.loads(path.read_text())
        with open(metrics_dir / "stage1.json", "w") as f:
            json.dump({"table_model": s1_model, "col_model": col_model,
                        "metrics": stage1_data}, f, indent=2)

        # Consolidate Stage 2 metrics
        stage2_data = {}
        for split in splits:
            path = STAGE2_ROOT / stage2_key / f"metrics_{split}_topk_100.json"
            if path.exists():
                stage2_data[split] = json.loads(path.read_text())
        cal_path = STAGE2_ROOT / stage2_key / "calibration_dev.json"
        cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        with open(metrics_dir / "stage2.json", "w") as f:
            json.dump({"col_model": col_model, "calibration": cal,
                        "metrics": stage2_data}, f, indent=2)

        # End-to-end + summary CSV
        end_to_end = {"col_model": col_model, "row_model": row_model,
                      "table_model": s1_model, "splits": split_metrics_all}
        with open(metrics_dir / "end_to_end.json", "w") as f:
            json.dump(end_to_end, f, indent=2)

        # Per-query dump (for source splits and other post-hoc subgroupings)
        for split, results in per_query_by_split.items():
            per_query_path = metrics_dir / f"per_query_{split}.jsonl"
            with per_query_path.open("w") as f:
                for r in results:
                    rr = r.get("region_recall") or {}
                    f.write(json.dumps({
                        "qid": r.get("qid"),
                        "parent_id": r.get("parent_id"),
                        "source": r.get("source"),
                        "tier": r.get("tier"),
                        "uj_h": r.get("uj_h"),
                        "cell_f1": r.get("cell_f1"),
                        "union_recall": rr.get("union_region"),
                        "join_recall": rr.get("join_region"),
                        "hard_recall": rr.get("hard_region"),
                        "core_recall": rr.get("core_core"),
                    }) + "\n")

        summary = {"col_model": col_model, "row_model": row_model, "table_model": s1_model}
        for split in splits:
            sm = split_metrics_all.get(split, {})
            summary[f"cell_f1_{split}"] = sm.get("cell_f1")
            summary[f"uj_h_{split}"] = sm.get("uj_h")
            summary[f"parent_row_recall_{split}"] = sm.get("parent_row_recall")
            summary[f"parent_col_recall_{split}"] = sm.get("parent_col_recall")
            rr = sm.get("region_recall", {})
            summary[f"union_recall_{split}"] = rr.get("union_region")
            summary[f"join_recall_{split}"] = rr.get("join_region")
            summary[f"hard_recall_{split}"] = rr.get("hard_region")
        s1_dev = stage1_data.get("dev_topk_10", {})
        summary["recall_any_10_dev"] = s1_dev.get("recall_any")
        summary["recall_any_100_dev"] = stage1_data.get("dev_topk_100", {}).get("recall_any")
        s2_dev = stage2_data.get("dev", {})
        summary["relation_acc_dev"] = s2_dev.get("relation_acc")
        pd.DataFrame([summary]).to_csv(metrics_dir / "summary.csv", index=False)

    return total_queries > 0


# ── Oracle-RA Mode ────────────────────────────────────────────────

def build_oracle_stage2_entry(qt):
    """Build a synthetic Stage 2 entry from ground-truth query task.

    Uses .npz provenance maps to construct perfect column alignment
    and relation labels, bypassing Stage 1 retrieval and Stage 2 classification.
    """
    qid = qt["query_table_id"]
    seed_npz_path = TABLE_MAPS_DIR / f"{qid}.npz"
    if not seed_npz_path.exists():
        return None

    seed_npz = np.load(seed_npz_path)
    seed_col_pid = seed_npz["col_parent_idx"]  # query col -> parent col

    candidates = []
    for rel in qt.get("relevant", []):
        tid = rel["table_id"]
        relation = rel["relation"]  # "union" or "join"

        target_npz_path = TABLE_MAPS_DIR / f"{tid}.npz"
        if not target_npz_path.exists():
            continue
        target_npz = np.load(target_npz_path)
        target_col_pid = target_npz["col_parent_idx"]  # target col -> parent col

        # Build column alignment from shared parent column IDs
        # Map parent_col_id -> target_col_idx
        pid_to_target = {}
        for t_idx, pid in enumerate(target_col_pid):
            pid = int(pid)
            if pid >= 0:
                pid_to_target[pid] = t_idx

        pairs = []
        key_pair = None
        for q_idx, pid in enumerate(seed_col_pid):
            pid = int(pid)
            if pid >= 0 and pid in pid_to_target:
                t_idx = pid_to_target[pid]
                pairs.append([int(q_idx), int(t_idx), 1.0])

        # For join: key pair is the shared column(s); pick the first match
        if relation == "join" and pairs:
            key_pair = [pairs[0][0], pairs[0][1]]

        candidates.append({
            "table_id": tid,
            "stage1_score": 1.0,
            "alignment": {
                "pairs": pairs,
                "n_query_cols": len(seed_col_pid),
                "n_cand_cols": len(target_col_pid),
            },
            "relation_pred": relation,
            "relation_conf": 1.0,
            "key_pair": key_pair,
        })

    if not candidates:
        return None
    return {"query_table_id": qid, "candidates": candidates}


def process_combination_oracle_ra(row_model, query_tasks, manifest, splits,
                                   parents=None, save_enriched=False,
                                   skip_evaluation=False):
    """Process all queries using oracle Stage 1+2, varying only row model."""
    combo_name = f"oracle_ra__{row_model}"
    print(f"\n  Oracle-RA Combination: {combo_name}")
    t0 = time.time()

    # Load row embeddings
    print(f"    Loading row embeddings ({row_model})...")
    t_load = time.time()
    q_row_lookup, t_row_lookup = load_row_embeddings(row_model)
    print(f"    Loaded {len(q_row_lookup)} query + {len(t_row_lookup)} target "
          f"row embeddings in {time.time() - t_load:.1f}s")

    task_by_qid = {qt["query_table_id"]: qt for qt in query_tasks}
    total_queries = 0
    split_metrics_all = {}
    per_query_by_split = {}

    for split in splits:
        split_tasks = [qt for qt in query_tasks if qt["split"] == split]
        if not split_tasks:
            continue

        print(f"    Split: {split} ({len(split_tasks)} queries)")

        stage3_dir = STAGE3_ROOT / combo_name
        stage3_dir.mkdir(parents=True, exist_ok=True)
        if save_enriched:
            enriched_dir = ENRICHED_ROOT / combo_name / split
            enriched_dir.mkdir(parents=True, exist_ok=True)

        merge_log = []
        n_processed = 0
        split_eval_results = []
        tier_eval_results = defaultdict(list)

        for qt in split_tasks:
            qid = qt["query_table_id"]

            # Build oracle stage2 entry from ground truth
            oracle_entry = build_oracle_stage2_entry(qt)
            if oracle_entry is None:
                continue

            enriched, provenance, log_entry = process_query(
                qid, oracle_entry, manifest, q_row_lookup, t_row_lookup, split)

            if enriched is None:
                continue

            if save_enriched:
                csv_path = enriched_dir / f"{qid}.enriched.csv"
                enriched.to_csv(csv_path, index=False)

            if not skip_evaluation and parents is not None:
                qt_entry = task_by_qid.get(qid)
                if qt_entry:
                    parent_id = qt_entry["parent_id"]
                    parent_entry = parents.get(parent_id)
                    if parent_entry and Path(parent_entry["csv_path"]).exists():
                        parent_df = load_parent_csv(parent_entry)
                        eval_result = evaluate_query_inline(
                            qid, qt_entry, enriched, parent_df, parent_entry)
                        if eval_result is not None:
                            eval_result["tier"] = qt_entry["noise_tier"]
                            eval_result["qid"] = qid
                            eval_result["parent_id"] = parent_id
                            eval_result["source"] = parent_id.split("__", 1)[0]
                            split_eval_results.append(eval_result)
                            tier_eval_results[qt_entry["noise_tier"]].append(eval_result)

            merge_log.append(log_entry)
            n_processed += 1
            if n_processed % 500 == 0:
                print(f"      {n_processed}/{len(split_tasks)} queries processed...")

        log_path = stage3_dir / f"merge_log_{split}.jsonl"
        with open(log_path, "w") as f:
            for entry in merge_log:
                f.write(json.dumps(entry) + "\n")

        total_queries += n_processed

        if not skip_evaluation and parents is not None and split_eval_results:
            split_metrics_all[split] = _aggregate_split_metrics(
                split_eval_results, tier_eval_results,
                len(split_eval_results), len(split_tasks))
            per_query_by_split[split] = split_eval_results
            sm = split_metrics_all[split]
            _f = lambda v: f"{v:.4f}" if v is not None else "N/A"
            rr = sm.get("region_recall", {})
            print(f"      CellF1={_f(sm['cell_f1'])}  UJ-H={_f(sm.get('uj_h'))}  "
                  f"union={_f(rr.get('union_region'))}  join={_f(rr.get('join_region'))}  "
                  f"hard={_f(rr.get('hard_region'))}")

    elapsed = time.time() - t0
    print(f"    Done: {total_queries} queries in {elapsed:.1f}s")

    if not skip_evaluation and parents is not None and split_metrics_all:
        metrics_dir = METRICS_ROOT / combo_name
        metrics_dir.mkdir(parents=True, exist_ok=True)
        end_to_end = {"col_model": "oracle", "row_model": row_model,
                      "table_model": "oracle", "mode": "oracle_ra",
                      "splits": split_metrics_all}
        with open(metrics_dir / "end_to_end.json", "w") as f:
            json.dump(end_to_end, f, indent=2)

        # Per-query dump (for source splits and other post-hoc subgroupings)
        for split, results in per_query_by_split.items():
            per_query_path = metrics_dir / f"per_query_{split}.jsonl"
            with per_query_path.open("w") as f:
                for r in results:
                    rr = r.get("region_recall") or {}
                    f.write(json.dumps({
                        "qid": r.get("qid"),
                        "parent_id": r.get("parent_id"),
                        "source": r.get("source"),
                        "tier": r.get("tier"),
                        "uj_h": r.get("uj_h"),
                        "cell_f1": r.get("cell_f1"),
                        "union_recall": rr.get("union_region"),
                        "join_recall": rr.get("join_region"),
                        "hard_recall": rr.get("hard_region"),
                        "core_recall": rr.get("core_core"),
                    }) + "\n")

        summary = {"col_model": "oracle", "row_model": row_model,
                   "table_model": "oracle", "mode": "oracle_ra"}
        for split in splits:
            sm = split_metrics_all.get(split, {})
            summary[f"cell_f1_{split}"] = sm.get("cell_f1")
            summary[f"uj_h_{split}"] = sm.get("uj_h")
            rr = sm.get("region_recall", {})
            summary[f"union_recall_{split}"] = rr.get("union_region")
            summary[f"join_recall_{split}"] = rr.get("join_region")
            summary[f"hard_recall_{split}"] = rr.get("hard_region")
        pd.DataFrame([summary]).to_csv(metrics_dir / "summary.csv", index=False)

    return total_queries > 0


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Row Matching + Merge + Evaluation")
    parser.add_argument("--col_models", nargs="+", default=COLUMN_MODELS)
    parser.add_argument("--row_models", nargs="+", default=ROW_MODELS)
    parser.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root for DLTE outputs (default: {project_root}/results/evaluation/dlte)")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root (default: derived from script location)")
    parser.add_argument("--table_model", type=str, default=None,
                        help="Table model for Stage 1 retrieval (default: same as --col_models)")
    parser.add_argument("--embeddings_root", type=str, default=None,
                        help="Embeddings root (default: {project_root}/embeddings)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Data root containing 'datasets/dlte_v1/' (default: {project_root})")
    parser.add_argument("--save-enriched", action="store_true", default=False,
                        help="Write enriched CSVs to disk (for step15/debugging)")
    parser.add_argument("--skip-evaluation", action="store_true", default=False,
                        help="Skip evaluation, merge-only mode")
    parser.add_argument("--oracle-ra", action="store_true", default=False,
                        help="Oracle-RA mode: use ground-truth retrieval+alignment, test only row matching")
    args = parser.parse_args()

    resolve_paths(args)

    print("Step 10: Stage 3 — Row Matching + Merge + Evaluation")
    if args.oracle_ra:
        print("  *** ORACLE-RA MODE: ground-truth retrieval + alignment ***")
    print("=" * 60)
    print(f"Row models: {args.row_models}")
    if not args.oracle_ra:
        print(f"Column models: {args.col_models}")
    print(f"Splits: {args.splits}")
    print(f"Save enriched: {args.save_enriched}")
    print(f"Skip evaluation: {args.skip_evaluation}")

    # Load shared data
    print("\nLoading manifest...")
    manifest = load_manifest()
    print(f"  {len(manifest)} entries")

    print("Loading query tasks...")
    query_tasks = load_query_tasks()
    print(f"  {len(query_tasks)} tasks")

    parents = None
    if not args.skip_evaluation:
        print("Loading parents manifest...")
        parents = load_parents()
        print(f"  {len(parents)} parents")

    # ── Oracle-RA mode: sweep row models only ──
    if args.oracle_ra:
        total_combos = len(args.row_models)
        succeeded = 0
        for row_model in args.row_models:
            try:
                if process_combination_oracle_ra(
                        row_model, query_tasks, manifest, args.splits,
                        parents=parents,
                        save_enriched=args.save_enriched,
                        skip_evaluation=args.skip_evaluation):
                    succeeded += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"Oracle-RA: {succeeded}/{total_combos} row models processed")
        print(f"Metrics: {METRICS_ROOT}")
        print(f"{'='*60}")
        return 0 if succeeded == total_combos else 1

    # ── Normal mode: sweep col_models x row_models ──
    total_combos = len(args.col_models) * len(args.row_models)
    succeeded = 0

    for col_model in args.col_models:
        for row_model in args.row_models:
            try:
                if process_combination(col_model, row_model, query_tasks,
                                       manifest, args.splits,
                                       table_model=args.table_model,
                                       parents=parents,
                                       save_enriched=args.save_enriched,
                                       skip_evaluation=args.skip_evaluation):
                    succeeded += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Processed {succeeded}/{total_combos} combinations")
    if not args.skip_evaluation:
        print(f"Metrics: {METRICS_ROOT}")
    if args.save_enriched:
        print(f"Enriched CSVs: {ENRICHED_ROOT}")
    print(f"Merge logs: {STAGE3_ROOT}")
    print(f"{'='*60}")

    return 0 if succeeded == total_combos else 1


if __name__ == "__main__":
    sys.exit(main())
