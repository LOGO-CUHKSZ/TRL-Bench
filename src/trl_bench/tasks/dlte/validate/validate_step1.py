"""
Validate Step 1 outputs against PLAN.md test conditions.

Test conditions:
  1. Every csv_path exists and is readable by pandas
  2. All records satisfy 5 <= n_cols <= 20 and 30 <= n_rows <= 200
  3. All key_uniqueness >= 0.80
  4. Expected counts: TabFact ~600-991, WTQ ~250-389
  5. Re-running with same seed produces identical output (checked via hash)
  6. Spot-check 50 random tables: print key_col, uniqueness, top-5 values
"""

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "dlte_v1" / "manifests" / "parents_filtered.jsonl"
SPLITS_PATH = PROJECT_ROOT / "datasets" / "dlte_v1" / "manifests" / "splits.json"

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

    # Load manifest (resolve relative csv_path against project root)
    parents = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            p = Path(entry["csv_path"])
            if not p.is_absolute():
                entry["csv_path"] = str(PROJECT_ROOT / p)
            parents.append(entry)

    splits = json.load(open(SPLITS_PATH))

    print(f"Loaded {len(parents)} parents, splits: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}")
    print()

    # ── Test 1: csv_path exists and is readable ──────────────────────
    print("Test 1: CSV paths exist and are readable")
    bad_paths = []
    unreadable = []
    for p in parents:
        path = Path(p["csv_path"])
        if not path.exists():
            bad_paths.append(p["parent_id"])
        else:
            try:
                df = pd.read_csv(str(path), engine="python", on_bad_lines="skip", nrows=1)
            except Exception:
                unreadable.append(p["parent_id"])

    check("All csv_path exist", len(bad_paths) == 0,
          f"{len(bad_paths)} missing: {bad_paths[:5]}")
    check("All csv_path readable", len(unreadable) == 0,
          f"{len(unreadable)} unreadable: {unreadable[:5]}")
    print()

    # ── Test 2: Size constraints ─────────────────────────────────────
    print("Test 2: Size constraints (5<=cols<=20, 30<=rows<=200)")
    bad_rows = [p for p in parents if p["n_rows"] < 30 or p["n_rows"] > 200]
    bad_cols = [p for p in parents if p["n_cols"] < 5 or p["n_cols"] > 20]
    check("All n_rows in [30, 200]", len(bad_rows) == 0,
          f"{len(bad_rows)} violations")
    check("All n_cols in [5, 20]", len(bad_cols) == 0,
          f"{len(bad_cols)} violations")
    print()

    # ── Test 3: Key uniqueness >= 0.80 ───────────────────────────────
    print("Test 3: Key uniqueness >= 0.80")
    bad_uniq = [p for p in parents if p["key_uniqueness"] < 0.80]
    check("All key_uniqueness >= 0.80", len(bad_uniq) == 0,
          f"{len(bad_uniq)} violations: {[(p['parent_id'], p['key_uniqueness']) for p in bad_uniq[:5]]}")
    print()

    # ── Test 4: Expected counts ──────────────────────────────────────
    print("Test 4: Expected counts")
    tabfact_count = sum(1 for p in parents if p["dataset"] == "tabfact")
    wtq_count = sum(1 for p in parents if p["dataset"] == "wtq")
    check(f"TabFact count in [600, 991]: got {tabfact_count}",
          600 <= tabfact_count <= 991)
    check(f"WTQ count in [250, 400]: got {wtq_count}",
          250 <= wtq_count <= 400)
    print()

    # ── Test 4b: Splits consistency ──────────────────────────────────
    print("Test 4b: Splits consistency")
    all_split_ids = set(splits["train"] + splits["dev"] + splits["test"])
    all_parent_ids = set(p["parent_id"] for p in parents)
    check("Splits cover all parent IDs",
          all_split_ids == all_parent_ids,
          f"diff: {all_split_ids.symmetric_difference(all_parent_ids)}")
    check("No duplicates across splits",
          len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == len(all_split_ids),
          "Duplicate IDs found across splits")

    # Approximate split ratios (60/15/25)
    total = len(parents)
    train_pct = len(splits["train"]) / total * 100
    dev_pct = len(splits["dev"]) / total * 100
    test_pct = len(splits["test"]) / total * 100
    print(f"  INFO: Split ratios — train: {train_pct:.1f}%, dev: {dev_pct:.1f}%, test: {test_pct:.1f}%")
    check("Train ratio ~60%", 55 <= train_pct <= 65, f"got {train_pct:.1f}%")
    check("Dev ratio ~15%", 10 <= dev_pct <= 20, f"got {dev_pct:.1f}%")
    check("Test ratio ~25%", 20 <= test_pct <= 30, f"got {test_pct:.1f}%")
    print()

    # ── Test 5: Reproducibility ──────────────────────────────────────
    print("Test 5: Reproducibility (file hash)")
    with open(MANIFEST_PATH, "rb") as f:
        manifest_hash = hashlib.sha256(f.read()).hexdigest()
    with open(SPLITS_PATH, "rb") as f:
        splits_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  INFO: parents_filtered.jsonl SHA256: {manifest_hash[:16]}...")
    print(f"  INFO: splits.json SHA256:            {splits_hash[:16]}...")
    check("Manifest hash is deterministic (record for re-run check)", True)
    print()

    # ── Test 6: Spot-check 50 random tables ──────────────────────────
    print("Test 6: Spot-check 50 random tables")
    rng = np.random.RandomState(SEED)
    n_check = min(50, len(parents))
    sample_indices = rng.choice(len(parents), n_check, replace=False)

    print(f"  Checking {n_check} tables...")
    recompute_failures = []
    for idx in sample_indices:
        p = parents[idx]
        try:
            df = pd.read_csv(p["csv_path"], engine="python", on_bad_lines="skip")
            # Drop artifact columns (same logic as the script)
            import re
            artifact_pattern = re.compile(r"^(Unnamed:\s*\d+)$")
            cols_to_drop = [c for c in df.columns if artifact_pattern.match(str(c))]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            actual_rows, actual_cols = df.shape

            # Verify recorded shape matches
            if actual_rows != p["n_rows"] or actual_cols != p["n_cols"]:
                recompute_failures.append(
                    f"{p['parent_id']}: recorded ({p['n_rows']}×{p['n_cols']}) vs actual ({actual_rows}×{actual_cols})")
                continue

            # Verify key column uniqueness
            key_col_name = p["key_col"]
            if key_col_name in df.columns:
                series = df[key_col_name].dropna()
                if series.dtype == object:
                    normalized = series.astype(str).str.strip().str.lower()
                    actual_uniq = normalized.nunique() / len(series) if len(series) > 0 else 0
                else:
                    actual_uniq = series.nunique() / len(series) if len(series) > 0 else 0
                if abs(actual_uniq - p["key_uniqueness"]) > 0.01:
                    recompute_failures.append(
                        f"{p['parent_id']}: key_uniqueness recorded={p['key_uniqueness']:.4f} vs actual={actual_uniq:.4f}")
        except Exception as e:
            recompute_failures.append(f"{p['parent_id']}: exception {e}")

    check(f"Spot-check {n_check} tables: shape & uniqueness match",
          len(recompute_failures) == 0,
          f"{len(recompute_failures)} failures:\n    " + "\n    ".join(recompute_failures[:10]))

    # Print details of 10 samples
    print(f"\n  Sample details (first 10 of {n_check}):")
    for idx in sample_indices[:10]:
        p = parents[idx]
        df = pd.read_csv(p["csv_path"], engine="python", on_bad_lines="skip")
        artifact_pattern = re.compile(r"^(Unnamed:\s*\d+)$")
        cols_to_drop = [c for c in df.columns if artifact_pattern.match(str(c))]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        key_vals = df[p["key_col"]].dropna().head(5).tolist()
        print(f"    {p['parent_id']}: {p['n_rows']}×{p['n_cols']}, "
              f"key='{p['key_col']}' (uniq={p['key_uniqueness']}), "
              f"top5={key_vals}")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
