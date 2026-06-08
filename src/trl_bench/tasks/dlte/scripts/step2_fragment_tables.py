"""
Step 2: Table Fragmentation for DLTE benchmark.

For each parent table × replicate:
  1. Generate base split (which rows/cols go to seed) — shared across tiers
  2. For each noise tier (0-3):
     a. Seed (query):  key_col + subset of non-key cols, subset of rows — always clean
     b. Union (target): same cols as seed, complementary rows — noise applied
     c. Join (target):  key_col + remaining cols, ALL rows — noise applied
     d. Save CSVs + ground truth mappings (.npz)

Noise tiers:
  - Tier 0: clean (just slicing)
  - Tier 1: schema noise (col name perturbation + col order shuffle)
  - Tier 2: Tier 1 + 5% non-key cell corruption + row order shuffle
  - Tier 3: Tier 2 + 10% key-value perturbation + spurious rows/cols

Outputs:
  - datasets/dlte_v1/config/fragmentation.yaml
  - datasets/dlte_v1/queries/tables/{query_id}.csv
  - datasets/dlte_v1/lake/targets/tables/{target_id}.csv
  - datasets/dlte_v1/manifests/fragments_manifest.jsonl
  - datasets/dlte_v1/ground_truth/table_maps/{table_id}.npz
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
N_TIERS = 4       # 0, 1, 2, 3
N_REPLICATES = 1

# Fragmentation parameters
COL_FRAC_RANGE = (0.35, 0.50)   # fraction of non-key cols for seed
ROW_FRAC_RANGE = (0.60, 0.80)   # fraction of rows for seed
MIN_SEED_ROWS = 10
MIN_SEED_NON_KEY_COLS = 2       # ensures seed and join both have >= 3 cols

# Tier 1: schema noise
COL_NAME_PERTURB_PROB = 0.70
COL_ORDER_SHUFFLE = True

# Tier 2: cell noise
CELL_CORRUPTION_PROB = 0.05
ROW_ORDER_SHUFFLE = True

# Tier 3: hard noise
KEY_PERTURB_PROB = 0.10
SPURIOUS_ROWS_RANGE = (2, 5)
SPURIOUS_COLS_RANGE = (1, 2)

# Same artifact pattern as Step 1
ARTIFACT_COL_PATTERN = re.compile(r"^(Unnamed:\s*\d+)$")


# ═══════════════════════════════════════════════════════════════════
# RNG helpers
# ═══════════════════════════════════════════════════════════════════

def _make_rng(seed_str: str) -> np.random.RandomState:
    """Create a deterministic RNG from a seed string."""
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    return np.random.RandomState(seed_int)


def make_split_rng(parent_id: str, rep: int) -> np.random.RandomState:
    """RNG for base split (shared across tiers)."""
    return _make_rng(f"{GLOBAL_SEED}_{parent_id}_{rep}_split")


def make_noise_rng(parent_id: str, rep: int, tier: int, role: str) -> np.random.RandomState:
    """RNG for noise (per fragment, deterministic)."""
    return _make_rng(f"{GLOBAL_SEED}_{parent_id}_{rep}_noise_{tier}_{role}")


# ═══════════════════════════════════════════════════════════════════
# Noise functions
# ═══════════════════════════════════════════════════════════════════

def perturb_col_name(name: str, rng: np.random.RandomState) -> str:
    """Apply a random perturbation to a column name."""
    choice = rng.randint(0, 5)
    if choice == 0:
        return name.lower()
    elif choice == 1:
        return name.upper()
    elif choice == 2:
        return name.replace(" ", "_")
    elif choice == 3:
        return name.replace("_", " ")
    elif choice == 4:
        return name.rstrip() + " "
    return name


def corrupt_cell(value: str, rng: np.random.RandomState) -> str:
    """Corrupt a single cell value (Tier 2+)."""
    choice = rng.randint(0, 3)
    if choice == 0:
        return ""
    elif choice == 1:
        # Swap two adjacent characters
        if len(value) >= 2:
            pos = rng.randint(0, len(value) - 1)
            chars = list(value)
            swap_pos = min(pos + 1, len(chars) - 1)
            chars[pos], chars[swap_pos] = chars[swap_pos], chars[pos]
            return "".join(chars)
        return value
    elif choice == 2:
        # Truncate to first half
        if len(value) >= 3:
            n = max(1, len(value) // 2)
            return value[:n]
        return value
    return value


def perturb_key_value(value: str, rng: np.random.RandomState) -> str:
    """Subtle perturbation to a key value (Tier 3)."""
    choice = rng.randint(0, 3)
    if choice == 0:
        # Swap two adjacent characters
        if len(value) >= 2:
            pos = rng.randint(0, len(value) - 1)
            chars = list(value)
            swap_pos = min(pos + 1, len(chars) - 1)
            chars[pos], chars[swap_pos] = chars[swap_pos], chars[pos]
            return "".join(chars)
        return value
    elif choice == 1:
        return value + " "
    elif choice == 2:
        if len(value) >= 2:
            return value[:-1]
        return value
    return value


def generate_spurious_rows(df: pd.DataFrame, n: int,
                           rng: np.random.RandomState) -> pd.DataFrame:
    """Generate spurious rows by independently sampling values from each column."""
    data = {}
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            indices = rng.choice(len(non_null), size=n, replace=True)
            data[col] = non_null.iloc[indices].values
        else:
            data[col] = [np.nan] * n
    return pd.DataFrame(data)


def generate_spurious_col_values(df: pd.DataFrame,
                                 rng: np.random.RandomState) -> np.ndarray:
    """Generate a spurious column by shuffling an existing column's values."""
    source_idx = rng.randint(0, len(df.columns))
    values = df.iloc[:, source_idx].values.copy()
    rng.shuffle(values)
    return values


# ═══════════════════════════════════════════════════════════════════
# Base split computation
# ═══════════════════════════════════════════════════════════════════

def compute_base_split(n_rows: int, n_cols: int, key_col_idx: int,
                       rng: np.random.RandomState):
    """
    Determine which rows and columns go to the seed fragment.

    Returns:
        seed_row_idx:  sorted int32 array — parent row indices for seed
        seed_col_idx:  sorted int32 array — parent col indices for seed (includes key)
        union_row_idx: sorted int32 array — parent row indices for union
        join_col_idx:  sorted int32 array — parent col indices for join (key + non-seed)
    """
    # ── Column split ──
    non_key_indices = np.array([i for i in range(n_cols) if i != key_col_idx])
    n_non_key = len(non_key_indices)
    assert n_non_key >= 4, f"Expected >= 4 non-key cols, got {n_non_key}"

    col_frac = rng.uniform(COL_FRAC_RANGE[0], COL_FRAC_RANGE[1])
    n_seed_non_key = max(MIN_SEED_NON_KEY_COLS, round(col_frac * n_non_key))
    n_seed_non_key = min(n_seed_non_key, n_non_key - 2)  # join gets key + ≥2 = ≥3 cols

    chosen_cols = rng.choice(non_key_indices, n_seed_non_key, replace=False)
    seed_col_idx = np.sort(np.concatenate([[key_col_idx], chosen_cols])).astype(np.int32)

    seed_col_set = set(seed_col_idx)
    join_non_seed = np.array([i for i in non_key_indices if i not in seed_col_set])
    join_col_idx = np.sort(np.concatenate([[key_col_idx], join_non_seed])).astype(np.int32)

    # ── Row split ──
    row_frac = rng.uniform(ROW_FRAC_RANGE[0], ROW_FRAC_RANGE[1])
    n_seed_rows = max(MIN_SEED_ROWS, round(row_frac * n_rows))
    n_seed_rows = min(n_seed_rows, n_rows - 5)  # union gets ≥5 rows

    chosen_rows = rng.choice(n_rows, n_seed_rows, replace=False)
    seed_row_idx = np.sort(chosen_rows).astype(np.int32)
    union_row_idx = np.sort(np.array(
        [i for i in range(n_rows) if i not in set(seed_row_idx)]
    )).astype(np.int32)

    return seed_row_idx, seed_col_idx, union_row_idx, join_col_idx


# ═══════════════════════════════════════════════════════════════════
# Noise application
# ═══════════════════════════════════════════════════════════════════

def apply_noise(df: pd.DataFrame, tier: int, key_col_pos: int,
                row_parent_idx: np.ndarray, col_parent_idx: np.ndarray,
                rng: np.random.RandomState):
    """
    Apply noise to a target fragment (union or join).

    Args:
        df:             clean fragment DataFrame
        tier:           0-3
        key_col_pos:    position of key column in the fragment
        row_parent_idx: mapping from fragment rows → parent rows
        col_parent_idx: mapping from fragment cols → parent cols
        rng:            RandomState for this specific fragment

    Returns:
        (noisy_df, updated_row_parent_idx, updated_col_parent_idx)
    """
    df = df.copy()
    row_map = row_parent_idx.copy()
    col_map = col_parent_idx.copy()

    if tier == 0:
        return df, row_map, col_map

    # ── Tier 1: Schema noise ─────────────────────────────────────
    if tier >= 1:
        # Perturb column names
        new_cols = list(df.columns)
        for i in range(len(new_cols)):
            if rng.random() < COL_NAME_PERTURB_PROB:
                new_cols[i] = perturb_col_name(str(new_cols[i]), rng)
        df.columns = new_cols

        # Shuffle column order
        if COL_ORDER_SHUFFLE:
            perm = rng.permutation(len(df.columns))
            df = df.iloc[:, perm].copy()
            col_map = col_map[perm]
            # Track where key column ended up
            inv_perm = np.argsort(perm)
            key_col_pos = int(inv_perm[key_col_pos])

    # ── Tier 2: Cell noise ───────────────────────────────────────
    if tier >= 2:
        # Row order shuffle
        if ROW_ORDER_SHUFFLE:
            row_perm = rng.permutation(len(df))
            df = df.iloc[row_perm].reset_index(drop=True)
            row_map = row_map[row_perm]

        # Cell corruption (skip key column)
        n_rows, n_cols = df.shape
        mask = rng.random((n_rows, n_cols)) < CELL_CORRUPTION_PROB
        mask[:, key_col_pos] = False
        for i, j in zip(*np.where(mask)):
            df.iat[i, j] = corrupt_cell(str(df.iat[i, j]), rng)

    # ── Tier 3: Hard noise ───────────────────────────────────────
    if tier >= 3:
        # Key value perturbation
        n_rows = len(df)
        key_mask = rng.random(n_rows) < KEY_PERTURB_PROB
        for i in np.where(key_mask)[0]:
            df.iat[i, key_col_pos] = perturb_key_value(
                str(df.iat[i, key_col_pos]), rng)

        # Spurious rows
        n_spur_rows = rng.randint(SPURIOUS_ROWS_RANGE[0],
                                  SPURIOUS_ROWS_RANGE[1] + 1)
        spur_df = generate_spurious_rows(df, n_spur_rows, rng)
        df = pd.concat([df, spur_df], ignore_index=True)
        row_map = np.concatenate([
            row_map, np.full(n_spur_rows, -1, dtype=np.int32)
        ])

        # Spurious columns
        n_spur_cols = rng.randint(SPURIOUS_COLS_RANGE[0],
                                  SPURIOUS_COLS_RANGE[1] + 1)
        for _ in range(n_spur_cols):
            col_name = f"extra_{rng.randint(1000, 9999)}"
            df[col_name] = generate_spurious_col_values(df, rng)
        col_map = np.concatenate([
            col_map, np.full(n_spur_cols, -1, dtype=np.int32)
        ])

    return df, row_map, col_map


# ═══════════════════════════════════════════════════════════════════
# Main processing
# ═══════════════════════════════════════════════════════════════════

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV and drop artifact columns (same logic as Step 1)."""
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    cols_to_drop = [c for c in df.columns if ARTIFACT_COL_PATTERN.match(str(c))]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def process_parent(parent: dict, splits_lookup: dict,
                   query_dir: Path, target_dir: Path, gt_dir: Path,
                   project_root: Path):
    """
    Fragment one parent table across all tiers and replicates.

    Returns list of manifest records (dicts).
    """
    parent_id = parent["parent_id"]
    key_col_idx = parent["key_col_idx"]
    key_col_name = parent["key_col"]
    split = splits_lookup.get(parent_id, "unknown")

    df = load_and_clean(parent["csv_path"])
    n_rows, n_cols = df.shape

    records = []

    for rep in range(N_REPLICATES):
        # Base split — shared across all tiers
        s_rng = make_split_rng(parent_id, rep)
        seed_row_idx, seed_col_idx, union_row_idx, join_col_idx = \
            compute_base_split(n_rows, n_cols, key_col_idx, s_rng)

        # Key column positions in each fragment type
        key_pos_in_seed = int(np.where(seed_col_idx == key_col_idx)[0][0])
        key_pos_in_join = int(np.where(join_col_idx == key_col_idx)[0][0])

        for tier in range(N_TIERS):
            # ── Fragment IDs ──
            seed_id  = f"dlte_v1__{parent_id}__seed__t{tier}__r{rep}"
            union_id = f"dlte_v1__{parent_id}__union__t{tier}__r{rep}"
            join_id  = f"dlte_v1__{parent_id}__join__t{tier}__r{rep}"

            # ── Seed (always clean) ──
            seed_df = df.iloc[seed_row_idx, seed_col_idx].reset_index(drop=True).copy()
            seed_row_map = seed_row_idx.copy()
            seed_col_map = seed_col_idx.copy()

            # ── Union target ──
            union_df_clean = df.iloc[union_row_idx, seed_col_idx].reset_index(drop=True).copy()
            union_rng = make_noise_rng(parent_id, rep, tier, "union")
            union_df, union_row_map, union_col_map = apply_noise(
                union_df_clean, tier, key_pos_in_seed,
                union_row_idx.copy(), seed_col_idx.copy(), union_rng)

            # ── Join target ──
            all_row_idx = np.arange(n_rows, dtype=np.int32)
            join_df_clean = df.iloc[all_row_idx, join_col_idx].reset_index(drop=True).copy()
            join_rng = make_noise_rng(parent_id, rep, tier, "join")
            join_df, join_row_map, join_col_map = apply_noise(
                join_df_clean, tier, key_pos_in_join,
                all_row_idx.copy(), join_col_idx.copy(), join_rng)

            # ── Save CSVs ──
            seed_df.to_csv(query_dir / f"{seed_id}.csv", index=False)
            union_df.to_csv(target_dir / f"{union_id}.csv", index=False)
            join_df.to_csv(target_dir / f"{join_id}.csv", index=False)

            # ── Save ground truth mappings ──
            np.savez(gt_dir / f"{seed_id}.npz",
                     row_parent_idx=seed_row_map, col_parent_idx=seed_col_map)
            np.savez(gt_dir / f"{union_id}.npz",
                     row_parent_idx=union_row_map, col_parent_idx=union_col_map)
            np.savez(gt_dir / f"{join_id}.npz",
                     row_parent_idx=join_row_map, col_parent_idx=join_col_map)

            # ── Manifest records ──
            for role, frag_type, fid, fdf, rmap, cmap, fdir in [
                ("query", "seed", seed_id, seed_df, seed_row_map, seed_col_map, query_dir),
                ("lake",  "union", union_id, union_df, union_row_map, union_col_map, target_dir),
                ("lake",  "join", join_id, join_df, join_row_map, join_col_map, target_dir),
            ]:
                # Find key column name in the fragment (might be perturbed)
                key_col_frag = str(fdf.columns[
                    int(np.where(cmap == key_col_idx)[0][0])
                ]) if key_col_idx in cmap else key_col_name

                records.append({
                    "table_id": fid,
                    "parent_id": parent_id,
                    "split": split,
                    "role": role,
                    "fragment_type": frag_type,
                    "noise_tier": tier,
                    "replicate_id": rep,
                    "csv_path": str((fdir / f"{fid}.csv").relative_to(project_root)),
                    "n_rows": len(fdf),
                    "n_cols": len(fdf.columns),
                    "key_col_fragment": key_col_frag,
                    "key_col_parent": key_col_name,
                })

    return records


def write_config(output_path: Path):
    """Write the frozen fragmentation config as YAML."""
    config = {
        "global_seed": GLOBAL_SEED,
        "n_tiers": N_TIERS,
        "n_replicates": N_REPLICATES,
        "fragmentation": {
            "col_frac_range": list(COL_FRAC_RANGE),
            "row_frac_range": list(ROW_FRAC_RANGE),
            "min_seed_rows": MIN_SEED_ROWS,
            "min_seed_non_key_cols": MIN_SEED_NON_KEY_COLS,
        },
        "noise": {
            "tier_0": "clean — slicing only",
            "tier_1": {
                "col_name_perturb_prob": COL_NAME_PERTURB_PROB,
                "col_order_shuffle": COL_ORDER_SHUFFLE,
            },
            "tier_2": {
                "extends": "tier_1",
                "cell_corruption_prob": CELL_CORRUPTION_PROB,
                "row_order_shuffle": ROW_ORDER_SHUFFLE,
            },
            "tier_3": {
                "extends": "tier_2",
                "key_perturb_prob": KEY_PERTURB_PROB,
                "spurious_rows_range": list(SPURIOUS_ROWS_RANGE),
                "spurious_cols_range": list(SPURIOUS_COLS_RANGE),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Simple YAML-like output (avoid pyyaml dependency)
        json.dump(config, f, indent=2)
    print(f"Written config: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Fragment parent tables for DLTE")
    parser.add_argument("--project-root", type=str,
                        default=None,
                        help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[3]
    dataset_root = project_root / "datasets" / "dlte_v1"

    # ── Load Step 1 outputs ──
    manifest_path = dataset_root / "manifests" / "parents_filtered.jsonl"
    splits_path = dataset_root / "manifests" / "splits.json"

    parents = []
    with open(manifest_path) as f:
        for line in f:
            parents.append(json.loads(line.strip()))

    with open(splits_path) as f:
        splits = json.load(f)

    # Build parent_id → split lookup
    splits_lookup = {}
    for split_name, ids in splits.items():
        for pid in ids:
            splits_lookup[pid] = split_name

    print(f"Loaded {len(parents)} parents")
    print(f"  Splits: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}")

    # ── Create output directories ──
    query_dir  = dataset_root / "queries" / "tables"
    target_dir = dataset_root / "lake" / "targets" / "tables"
    gt_dir     = dataset_root / "ground_truth" / "table_maps"
    config_dir = dataset_root / "config"

    for d in [query_dir, target_dir, gt_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Write frozen config ──
    write_config(config_dir / "fragmentation.json")

    # ── Resolve relative csv_path entries from Step 1 manifest ──
    for parent in parents:
        csv_path = Path(parent["csv_path"])
        if not csv_path.is_absolute():
            parent["csv_path"] = str(project_root / csv_path)

    # ── Process all parents ──
    all_records = []
    t_start = time.time()

    for i, parent in enumerate(parents):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(parents) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(parents)}] {parent['parent_id']}"
                  f"  ({rate:.1f} parents/s, ETA {eta:.0f}s)")

        try:
            records = process_parent(parent, splits_lookup,
                                     query_dir, target_dir, gt_dir,
                                     project_root=project_root)
            all_records.extend(records)
        except Exception as e:
            print(f"  ERROR processing {parent['parent_id']}: {e}")
            import traceback
            traceback.print_exc()

    elapsed_total = time.time() - t_start

    # ── Write manifest ──
    manifest_out = dataset_root / "manifests" / "fragments_manifest.jsonl"
    with open(manifest_out, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"FRAGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total fragments: {len(all_records)}")
    print(f"  Queries (seed):   {sum(1 for r in all_records if r['fragment_type'] == 'seed')}")
    print(f"  Union targets:    {sum(1 for r in all_records if r['fragment_type'] == 'union')}")
    print(f"  Join targets:     {sum(1 for r in all_records if r['fragment_type'] == 'join')}")
    print(f"Time: {elapsed_total:.1f}s ({len(parents)/elapsed_total:.1f} parents/s)")

    # Per-tier counts
    for t in range(N_TIERS):
        count = sum(1 for r in all_records if r['noise_tier'] == t)
        print(f"  Tier {t}: {count} fragments")

    # Size stats
    rows = [r["n_rows"] for r in all_records]
    cols = [r["n_cols"] for r in all_records]
    print(f"\nFragment sizes:")
    print(f"  Rows: min={min(rows)}, median={np.median(rows):.0f}, max={max(rows)}")
    print(f"  Cols: min={min(cols)}, median={np.median(cols):.0f}, max={max(cols)}")

    print(f"\nWritten:")
    print(f"  Config:   {config_dir / 'fragmentation.json'}")
    print(f"  Manifest: {manifest_out}")
    print(f"  Queries:  {query_dir}/ ({sum(1 for r in all_records if r['role'] == 'query')} files)")
    print(f"  Targets:  {target_dir}/ ({sum(1 for r in all_records if r['role'] == 'lake')} files)")
    print(f"  GT maps:  {gt_dir}/ ({len(all_records)} files)")


if __name__ == "__main__":
    main()
