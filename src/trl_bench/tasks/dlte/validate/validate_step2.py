"""
Validate Step 2 outputs against PLAN.md test conditions.

Test conditions:
  1. Each query has exactly 1 union + 1 join with matching (parent_id, tier, rep)
  2. Every fragment has a .npz mapping file
  3. Tier 0 reconstruction: union(seed, union_target) + join → CellF1 >= 0.999
  4. Fragment bounds: 3 <= n_cols <= 22, 5 <= n_rows <= 220
     (upper col bound: 20 + max 2 spurious from Tier 3)
  5. row_parent_idx in [0, parent_n_rows-1] or -1
  6. col_parent_idx in [0, parent_n_cols-1] or -1
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dlte_v1"
MANIFEST_PATH = DATASET_ROOT / "manifests" / "fragments_manifest.jsonl"
PARENTS_PATH = DATASET_ROOT / "manifests" / "parents_filtered.jsonl"
GT_DIR = DATASET_ROOT / "ground_truth" / "table_maps"

ARTIFACT_COL_PATTERN = re.compile(r"^(Unnamed:\s*\d+)$")
SEED = 42

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


def main():
    global passed, failed

    # Load fragments manifest (resolve relative csv_path against project root)
    fragments = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            p = Path(entry["csv_path"])
            if not p.is_absolute():
                entry["csv_path"] = str(PROJECT_ROOT / p)
            fragments.append(entry)

    # Load parents (resolve relative csv_path against project root)
    parents = {}
    with open(PARENTS_PATH) as f:
        for line in f:
            p = json.loads(line.strip())
            csv_p = Path(p["csv_path"])
            if not csv_p.is_absolute():
                p["csv_path"] = str(PROJECT_ROOT / csv_p)
            parents[p["parent_id"]] = p

    print(f"Loaded {len(fragments)} fragments, {len(parents)} parents\n")

    # ── Test 1: Each query has exactly 1 union + 1 join ──────────
    print("Test 1: Each (parent_id, tier, rep) has exactly 1 seed, 1 union, 1 join")

    # Group by (parent_id, tier, rep)
    groups = defaultdict(lambda: defaultdict(int))
    for f in fragments:
        key = (f["parent_id"], f["noise_tier"], f["replicate_id"])
        groups[key][f["fragment_type"]] += 1

    bad_groups = []
    for key, counts in groups.items():
        if counts.get("seed", 0) != 1 or counts.get("union", 0) != 1 or counts.get("join", 0) != 1:
            bad_groups.append((key, dict(counts)))

    check("All groups have exactly 1 seed + 1 union + 1 join",
          len(bad_groups) == 0,
          f"{len(bad_groups)} bad groups: {bad_groups[:5]}")
    print()

    # ── Test 2: Every fragment has a .npz mapping ────────────────
    print("Test 2: Every fragment has a .npz ground truth file")
    missing_npz = []
    for f in fragments:
        npz_path = GT_DIR / f"{f['table_id']}.npz"
        if not npz_path.exists():
            missing_npz.append(f["table_id"])

    check("All .npz files exist",
          len(missing_npz) == 0,
          f"{len(missing_npz)} missing: {missing_npz[:5]}")
    print()

    # ── Test 3: Tier 0 reconstruction check ──────────────────────
    print("Test 3: Tier 0 reconstruction (CellF1 >= 0.999)")

    # Get all Tier 0 triplets
    tier0_seeds = {f["parent_id"]: f for f in fragments
                   if f["noise_tier"] == 0 and f["fragment_type"] == "seed"}
    tier0_unions = {f["parent_id"]: f for f in fragments
                    if f["noise_tier"] == 0 and f["fragment_type"] == "union"}
    tier0_joins = {f["parent_id"]: f for f in fragments
                   if f["noise_tier"] == 0 and f["fragment_type"] == "join"}

    # Sample 100 random parents for reconstruction check
    rng = np.random.RandomState(SEED)
    parent_ids = list(tier0_seeds.keys())
    n_check = min(100, len(parent_ids))
    sample_ids = rng.choice(parent_ids, n_check, replace=False)

    cell_f1_scores = []
    reconstruction_failures = []

    for pid in sample_ids:
        try:
            # Load parent
            p_info = parents[pid]
            parent_df = pd.read_csv(p_info["csv_path"], engine="python", on_bad_lines="skip")
            cols_to_drop = [c for c in parent_df.columns
                            if ARTIFACT_COL_PATTERN.match(str(c))]
            if cols_to_drop:
                parent_df = parent_df.drop(columns=cols_to_drop)

            # Load fragments and their mappings
            seed_info = tier0_seeds[pid]
            union_info = tier0_unions[pid]
            join_info = tier0_joins[pid]

            seed_df = pd.read_csv(seed_info["csv_path"])
            union_df = pd.read_csv(union_info["csv_path"])
            join_df = pd.read_csv(join_info["csv_path"])

            seed_gt = np.load(GT_DIR / f"{seed_info['table_id']}.npz")
            union_gt = np.load(GT_DIR / f"{union_info['table_id']}.npz")
            join_gt = np.load(GT_DIR / f"{join_info['table_id']}.npz")

            # Reconstruct parent using ground truth mappings
            n_parent_rows, n_parent_cols = parent_df.shape
            reconstructed = np.full((n_parent_rows, n_parent_cols), None, dtype=object)

            # Fill from seed
            for frag_row, parent_row in enumerate(seed_gt["row_parent_idx"]):
                for frag_col, parent_col in enumerate(seed_gt["col_parent_idx"]):
                    if parent_row >= 0 and parent_col >= 0:
                        reconstructed[parent_row, parent_col] = str(seed_df.iat[frag_row, frag_col])

            # Fill from union (same columns as seed, complementary rows)
            for frag_row, parent_row in enumerate(union_gt["row_parent_idx"]):
                for frag_col, parent_col in enumerate(union_gt["col_parent_idx"]):
                    if parent_row >= 0 and parent_col >= 0:
                        reconstructed[parent_row, parent_col] = str(union_df.iat[frag_row, frag_col])

            # Fill from join (key + non-seed columns, all rows)
            for frag_row, parent_row in enumerate(join_gt["row_parent_idx"]):
                for frag_col, parent_col in enumerate(join_gt["col_parent_idx"]):
                    if parent_row >= 0 and parent_col >= 0:
                        # Only fill if not already filled by seed/union
                        # (key column overlaps — should match)
                        if reconstructed[parent_row, parent_col] is None:
                            reconstructed[parent_row, parent_col] = str(join_df.iat[frag_row, frag_col])

            # Compare reconstructed vs parent (cell-level string match)
            total_cells = n_parent_rows * n_parent_cols
            matches = 0
            mismatches = []
            for r in range(n_parent_rows):
                for c in range(n_parent_cols):
                    parent_val = str(parent_df.iat[r, c])
                    recon_val = str(reconstructed[r, c])
                    if parent_val == recon_val:
                        matches += 1
                    else:
                        if len(mismatches) < 3:
                            mismatches.append(
                                f"({r},{c}): parent='{parent_val}' vs recon='{recon_val}'")

            cell_f1 = matches / total_cells if total_cells > 0 else 0
            cell_f1_scores.append(cell_f1)

            if cell_f1 < 0.999:
                reconstruction_failures.append(
                    f"{pid}: CellF1={cell_f1:.6f} ({matches}/{total_cells}) "
                    f"mismatches: {mismatches}")

        except Exception as e:
            reconstruction_failures.append(f"{pid}: exception — {e}")
            cell_f1_scores.append(0)

    mean_f1 = np.mean(cell_f1_scores)
    min_f1 = np.min(cell_f1_scores)
    check(f"Tier 0 reconstruction: mean CellF1={mean_f1:.6f}, min={min_f1:.6f}",
          min_f1 >= 0.999,
          f"{len(reconstruction_failures)} failures:\n    " +
          "\n    ".join(reconstruction_failures[:10]))
    print()

    # ── Test 4: Fragment bounds ──────────────────────────────────
    print("Test 4: Fragment bounds")
    bad_rows = [f for f in fragments if f["n_rows"] < 5 or f["n_rows"] > 220]
    bad_cols = [f for f in fragments if f["n_cols"] < 3 or f["n_cols"] > 22]

    check(f"All n_rows in [5, 220]",
          len(bad_rows) == 0,
          f"{len(bad_rows)} violations: {[(f['table_id'], f['n_rows']) for f in bad_rows[:5]]}")
    check(f"All n_cols in [3, 22]",
          len(bad_cols) == 0,
          f"{len(bad_cols)} violations: {[(f['table_id'], f['n_cols']) for f in bad_cols[:5]]}")
    print()

    # ── Test 5 & 6: Mapping index validity ───────────────────────
    print("Test 5 & 6: Ground truth index validity (sample 200)")
    rng2 = np.random.RandomState(SEED + 1)
    n_gt_check = min(200, len(fragments))
    gt_sample = rng2.choice(len(fragments), n_gt_check, replace=False)

    idx_violations = []
    for idx in gt_sample:
        f = fragments[idx]
        p_info = parents[f["parent_id"]]
        npz = np.load(GT_DIR / f"{f['table_id']}.npz")

        row_idx = npz["row_parent_idx"]
        col_idx = npz["col_parent_idx"]

        # Check row indices
        valid_row = np.all((row_idx >= -1) & (row_idx < p_info["n_rows"]))
        # Check col indices
        valid_col = np.all((col_idx >= -1) & (col_idx < p_info["n_cols"]))

        if not valid_row or not valid_col:
            idx_violations.append(
                f"{f['table_id']}: row_valid={valid_row}, col_valid={valid_col}, "
                f"row_range=[{row_idx.min()},{row_idx.max()}], "
                f"col_range=[{col_idx.min()},{col_idx.max()}], "
                f"parent=({p_info['n_rows']},{p_info['n_cols']})")

    check(f"All sampled GT indices valid",
          len(idx_violations) == 0,
          f"{len(idx_violations)} violations:\n    " +
          "\n    ".join(idx_violations[:10]))

    # Also check that mapping dimensions match fragment dimensions
    dim_mismatches = []
    for idx in gt_sample:
        f = fragments[idx]
        npz = np.load(GT_DIR / f"{f['table_id']}.npz")
        if len(npz["row_parent_idx"]) != f["n_rows"]:
            dim_mismatches.append(
                f"{f['table_id']}: row_map_len={len(npz['row_parent_idx'])} vs n_rows={f['n_rows']}")
        if len(npz["col_parent_idx"]) != f["n_cols"]:
            dim_mismatches.append(
                f"{f['table_id']}: col_map_len={len(npz['col_parent_idx'])} vs n_cols={f['n_cols']}")

    check(f"GT mapping dimensions match fragment dimensions",
          len(dim_mismatches) == 0,
          f"{len(dim_mismatches)} mismatches:\n    " +
          "\n    ".join(dim_mismatches[:10]))
    print()

    # ── Test 7: CSV files exist ──────────────────────────────────
    print("Test 7: All CSV files exist")
    missing_csv = [f["table_id"] for f in fragments
                   if not Path(f["csv_path"]).exists()]
    check("All fragment CSVs exist",
          len(missing_csv) == 0,
          f"{len(missing_csv)} missing: {missing_csv[:5]}")
    print()

    # ── Summary stats ────────────────────────────────────────────
    print("Additional stats:")
    for tier in range(4):
        tier_frags = [f for f in fragments if f["noise_tier"] == tier]
        rows = [f["n_rows"] for f in tier_frags]
        cols = [f["n_cols"] for f in tier_frags]
        print(f"  Tier {tier}: {len(tier_frags)} fragments, "
              f"rows=[{min(rows)},{np.median(rows):.0f},{max(rows)}], "
              f"cols=[{min(cols)},{np.median(cols):.0f},{max(cols)}]")

    # ── Final Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
