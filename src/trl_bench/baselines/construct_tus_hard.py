#!/usr/bin/env python3
"""Construct TUS-Hard@0.15: a low-overlap subset of the TUS union search benchmark.

Filters TUS positive pairs by Hungarian-matched directional containment score,
retaining only "hard" pairs (score < threshold). Creates a filtered datalake
(removing easy-positive-only tables) so the existing run_search.py works unchanged.

Outputs:
    datasets/tus_hard/benchmark.pkl     -- hard GT (query -> hard positive list)
    datasets/tus_hard/overlap_scores.pkl -- all directed pairwise scores
    datasets/tus_hard/metadata.json     -- construction metadata
    datasets/tus_hard/datalake/         -- symlinks to kept tables
    datasets/tus_hard/query/          -- symlinks to hard-GT query tables only
    embeddings/column/{model}/tus_hard.pkl -- filtered embeddings

Usage:
    python utils/baselines/construct_tus_hard.py
    python utils/baselines/construct_tus_hard.py --threshold 0.15 --min_hard 5
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trl_bench.baselines.value_overlap.core import containment, extract_column_value_sets
from munkres import Munkres, make_cost_matrix, DISALLOWED


def compute_directed_overlap(query_cols: dict, cand_cols: dict) -> float:
    """Compute normalized directional containment via Hungarian matching.

    For each column pair, containment(query_col, cand_col) = |Q∩C|/|Q|.
    Hungarian matching finds optimal column alignment maximizing total containment.
    Score is normalized by min(num_query_cols, num_cand_cols).

    Returns a value in [0, 1].
    """
    q_names = list(query_cols.keys())
    c_names = list(cand_cols.keys())
    nrow, ncol = len(q_names), len(c_names)

    if nrow == 0 or ncol == 0:
        return 0.0

    graph = np.zeros((nrow, ncol), dtype=float)
    for i, qn in enumerate(q_names):
        for j, cn in enumerate(c_names):
            graph[i, j] = containment(query_cols[qn], cand_cols[cn])

    if graph.max() == 0:
        return 0.0

    max_graph = make_cost_matrix(
        graph, lambda cost: (graph.max() - cost) if cost != DISALLOWED else DISALLOWED
    )
    m = Munkres()
    indexes = m.compute(max_graph)

    matched_sum = sum(graph[row, col] for row, col in indexes)
    return matched_sum / min(nrow, ncol)


def construct_tus_hard(
    tus_dir: Path,
    output_dir: Path,
    threshold: float = 0.15,
    min_hard_positives: int = 5,
):
    """Build TUS-Hard benchmark artifacts."""
    print("=" * 60)
    print(f"Constructing TUS-Hard@{threshold}")
    print("=" * 60)

    # Load GT
    gt_path = tus_dir / "benchmark.pkl"
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    print(f"\nOriginal TUS: {len(gt)} queries, {sum(len(v) for v in gt.values())} positive pairs")

    # Build table-parse cache (one CSV read per unique table)
    print("\n[1/5] Building table-parse cache...")
    datalake_dir = tus_dir / "datalake"
    table_cache: dict[str, dict] = {}
    all_tables = set()
    for positives in gt.values():
        all_tables.update(positives)
    all_tables.update(gt.keys())

    for table_name in tqdm(sorted(all_tables), desc="Parsing tables", ncols=80):
        csv_path = datalake_dir / table_name
        if csv_path.exists():
            table_cache[table_name] = extract_column_value_sets(csv_path)
        else:
            table_cache[table_name] = {}

    print(f"  Cached {len(table_cache)} tables")

    # Compute directed overlap for all GT pairs
    print(f"\n[2/5] Computing directed overlap scores...")
    overlap_scores: dict[tuple[str, str], float] = {}
    n_pairs = sum(len(v) for v in gt.values())
    skipped = 0

    with tqdm(total=n_pairs, desc="Scoring pairs", ncols=80) as pbar:
        for query_name, positives in gt.items():
            q_cols = table_cache.get(query_name, {})
            if not q_cols:
                skipped += len(positives)
                pbar.update(len(positives))
                continue
            for pos_name in positives:
                c_cols = table_cache.get(pos_name, {})
                if not c_cols:
                    skipped += 1
                    pbar.update(1)
                    continue
                score = compute_directed_overlap(q_cols, c_cols)
                overlap_scores[(query_name, pos_name)] = score
                pbar.update(1)

    print(f"  Scored {len(overlap_scores)} directed pairs (skipped {skipped})")

    # Analyze distribution
    scores = np.array(list(overlap_scores.values()))
    print(f"\n  Overlap distribution:")
    print(f"    Mean:   {scores.mean():.4f}")
    print(f"    Median: {np.median(scores):.4f}")
    print(f"    Std:    {scores.std():.4f}")
    print(f"    < {threshold}: {(scores < threshold).sum()} pairs ({100*(scores < threshold).mean():.1f}%)")

    # Split into hard and ignored per query
    print(f"\n[3/5] Splitting GT at threshold {threshold}...")
    hard_gt: dict[str, list[str]] = {}
    ignored_positives: dict[str, list[str]] = {}
    all_easy_positives: set[str] = set()  # easy positives across ALL queries (incl. dropped)
    removed_queries = 0

    for query_name, positives in gt.items():
        hard = []
        ignored = []
        for pos_name in positives:
            score = overlap_scores.get((query_name, pos_name))
            if score is None:
                ignored.append(pos_name)  # couldn't score -> treat as ignored
                all_easy_positives.add(pos_name)
            elif score < threshold:
                hard.append(pos_name)
            else:
                ignored.append(pos_name)
                all_easy_positives.add(pos_name)

        if len(hard) >= min_hard_positives:
            hard_gt[query_name] = hard
            ignored_positives[query_name] = ignored
        else:
            removed_queries += 1

    print(f"  Retained: {len(hard_gt)} queries (removed {removed_queries} with < {min_hard_positives} hard positives)")
    hard_counts = [len(v) for v in hard_gt.values()]
    ignored_counts = [len(v) for v in ignored_positives.values()]
    print(f"  Hard positives per query: mean={np.mean(hard_counts):.1f}, median={np.median(hard_counts):.0f}, min={min(hard_counts)}, max={max(hard_counts)}")
    print(f"  Ignored positives per query: mean={np.mean(ignored_counts):.1f}")

    # Determine which tables to keep in the filtered datalake
    # Keep: retained query tables + hard positive tables + neutral tables (not in any GT).
    # Remove: tables that appear as easy positives (across ALL queries, incl. dropped)
    #         but are NOT retained queries or hard positives.
    print(f"\n[4/5] Building filtered datalake...")
    hard_tables = set()
    for tables in hard_gt.values():
        hard_tables.update(tables)
    query_tables = set(hard_gt.keys())
    keep_tables = hard_tables | query_tables

    # Remove any table that is an easy positive and not needed as a retained query/hard positive
    easy_only_tables = all_easy_positives - keep_tables

    all_datalake = set(os.listdir(datalake_dir))
    filtered_datalake = all_datalake - easy_only_tables

    print(f"  Original datalake: {len(all_datalake)} tables")
    print(f"  Removed (easy-only positives): {len(easy_only_tables)} tables")
    print(f"  Filtered datalake: {len(filtered_datalake)} tables")

    # Save artifacts
    print(f"\n[5/5] Saving artifacts...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "benchmark.pkl", "wb") as f:
        pickle.dump(hard_gt, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  benchmark.pkl: {len(hard_gt)} queries")

    with open(output_dir / "overlap_scores.pkl", "wb") as f:
        pickle.dump(overlap_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  overlap_scores.pkl: {len(overlap_scores)} pair scores")

    metadata = {
        "name": f"TUS-Hard@{threshold}",
        "source": "TUS (Table Union Search benchmark)",
        "threshold": threshold,
        "min_hard_positives": min_hard_positives,
        "normalization": "Hungarian-matched sum / min(num_cols_query, num_cols_positive)",
        "scoring": "directional containment: |Q∩C|/|Q| per column pair",
        "num_queries": len(hard_gt),
        "num_queries_removed": removed_queries,
        "total_hard_pairs": sum(len(v) for v in hard_gt.values()),
        "datalake_original": len(all_datalake),
        "datalake_removed": len(easy_only_tables),
        "datalake_filtered": len(filtered_datalake),
        "overlap_distribution": {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
            "below_threshold": int((scores < threshold).sum()),
            "total_scored": len(scores),
        },
        "constructed": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json")

    # Create filtered datalake directory with copies of kept tables
    dl_out = output_dir / "datalake"
    if dl_out.is_symlink():
        dl_out.unlink()
    elif dl_out.exists():
        shutil.rmtree(dl_out)
    dl_out.mkdir()
    for table_name in sorted(filtered_datalake):
        src = datalake_dir / table_name
        if src.exists():
            shutil.copy2(str(src), str(dl_out / table_name))
    print(f"  datalake/: {len(os.listdir(dl_out))} tables copied")

    # Query directory — only hard_gt query tables (copies from datalake)
    query_out = output_dir / "query"
    if query_out.is_symlink():
        query_out.unlink()
    elif query_out.exists():
        shutil.rmtree(query_out)
    query_out.mkdir()
    for query_name in sorted(hard_gt.keys()):
        src = datalake_dir / query_name
        if src.exists():
            shutil.copy2(str(src), str(query_out / query_name))
    print(f"  query/: {len(os.listdir(query_out))} query tables copied")

    # NOTE: Embeddings are NOT filtered here. tus_hard is registered in
    # datasets.yaml so the SLURM embedding generation pipeline treats it as
    # a standalone dataset. This ensures SSL models (e.g. Starmie) pretrain
    # on only the filtered datalake tables, not the full TUS datalake.

    print(f"\n{'=' * 60}")
    print(f"TUS-Hard@{threshold} construction complete")
    print(f"  Queries: {len(hard_gt)}")
    print(f"  Hard pairs: {sum(len(v) for v in hard_gt.values())}")
    print(f"  Datalake: {len(filtered_datalake)} tables (from {len(all_datalake)})")
    print(f"{'=' * 60}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Construct TUS-Hard benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Containment threshold for hard/ignored split")
    parser.add_argument("--min_hard", type=int, default=5,
                        help="Minimum hard positives per query to retain")
    args = parser.parse_args()

    tus_dir = PROJECT_ROOT / "datasets" / "tus"
    output_dir = PROJECT_ROOT / "datasets" / "tus_hard"

    construct_tus_hard(tus_dir, output_dir, args.threshold, args.min_hard)


if __name__ == "__main__":
    main()
