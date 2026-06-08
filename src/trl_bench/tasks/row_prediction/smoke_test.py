#!/usr/bin/env python3
"""
Comprehensive smoke tests for row_prediction downstream task fixes.

Tests all 6 issues:
  1. Canonical val split usage (v2 data)
  2. Result JSON format (stripped keys, macro_f1, head_type)
  3. Z-score normalization in MLP path
  4. Dead code removal (downstream_utils.py)
  5. Heuristic auto-detection warning
  6. best_model.pt print guard

Usage:
    source load_env && python downstream_tasks/row_prediction/smoke_test.py
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_downstream.py"

# Test data: smallest v2 dataset with both classification + regression
V2_EMB_DIR = REPO_ROOT / "assets" / "embeddings" / "row_prediction" / "bert" / "openml_1063"

passed = 0
failed = 0
errors = []


def run_train(args, capture_output=True, timeout=120):
    """Run train_downstream.py with given args, return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(TRAIN_SCRIPT)] + args
    result = subprocess.run(
        cmd, capture_output=capture_output, text=True,
        timeout=timeout, cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(msg)


def test_issue1_canonical_val_mlp():
    """Issue 1: MLP path uses canonical val split, not ad-hoc."""
    print("\n=== Test: Issue 1 — Canonical val split (MLP) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "problems",
        ])
        check("MLP exits cleanly", rc == 0, f"rc={rc}, stderr={stderr[-300:]}")
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "problems", "results.json")
        check("results.json exists", os.path.exists(results_file))
        with open(results_file) as f:
            results = json.load(f)

        ds = results["data_stats"]
        # openml_1063: train=417, val=52, test=53
        check("train uses full canonical split (417)", ds["train"] == 417,
              f"got {ds['train']}")
        check("val uses canonical split (52)", ds.get("val") == 52,
              f"got {ds.get('val')}")
        check("test count correct (53)", ds["test"] == 53,
              f"got {ds['test']}")

        # Verify val shape is printed in output
        check("Val shape printed in output", "Val: (52," in stdout or "Val: (52," in stderr,
              "expected 'Val: (52,' in output")


def test_issue1_canonical_val_linear():
    """Issue 1: Linear probe path passes canonical val to LinearProbeRunner."""
    print("\n=== Test: Issue 1 — Canonical val split (Linear) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "linear",
            "--seed", "42",
            "--label_column", "problems",
        ])
        check("Linear exits cleanly", rc == 0, f"rc={rc}")
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "problems", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        ds = results["data_stats"]
        check("linear data_stats.train = 417", ds["train"] == 417, f"got {ds['train']}")
        check("linear data_stats.val = 52", ds.get("val") == 52, f"got {ds.get('val')}")

        # LinearProbeRunner with refit_trainval should train on train+val
        lpm = results["test_results"].get("linear_probe_meta", {})
        check("linear_probe_meta.refit_trainval is True",
              lpm.get("refit_trainval") is True, f"got {lpm.get('refit_trainval')}")
        # With refit, train_samples should be train+val = 417+52 = 469
        check("linear refit train_samples = 469 (train+val)",
              lpm.get("train_samples") == 469,
              f"got {lpm.get('train_samples')}")


def test_issue1_v1_fallback():
    """Issue 1: Fallback to ad-hoc val split when no val split exists (synthetic v1)."""
    print("\n=== Test: Issue 1 — V1 fallback (ad-hoc val) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic v1 embedding directory (no val, no splits map)
        v1_dir = os.path.join(tmpdir, "v1_emb")
        os.makedirs(v1_dir)
        np.random.seed(123)
        n_train, n_test, dim = 200, 50, 32
        np.save(os.path.join(v1_dir, "train_embeddings.npy"),
                np.random.randn(n_train, dim).astype(np.float32))
        np.save(os.path.join(v1_dir, "test_embeddings.npy"),
                np.random.randn(n_test, dim).astype(np.float32))
        np.save(os.path.join(v1_dir, "train_labels.npy"),
                np.random.randint(0, 3, size=n_train).astype(np.int64))
        np.save(os.path.join(v1_dir, "test_labels.npy"),
                np.random.randint(0, 3, size=n_test).astype(np.int64))
        # v1 metadata: no splits map, single label_column
        meta = {
            "version": "1.0",
            "format": "unified_row_embedding",
            "embedding_dim": dim,
            "label_column": "synth_class",
            "label_task_types": {},
            "task": "classification",
        }
        with open(os.path.join(v1_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        out_dir = os.path.join(tmpdir, "results")
        rc, stdout, stderr = run_train([
            "--embedding_dir", v1_dir,
            "--output_dir", out_dir,
            "--head_type", "mlp",
            "--seed", "42",
            "--task", "classification",
        ])
        check("V1 MLP exits cleanly", rc == 0, f"rc={rc}, stderr={stderr[-300:]}")
        if rc != 0:
            return

        results_file = os.path.join(out_dir, "synth_class", "results.json")
        if not os.path.exists(results_file):
            # single label, might be at out_dir directly
            results_file = os.path.join(out_dir, "results.json")

        check("V1 results.json exists", os.path.exists(results_file))
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
            ds = results["data_stats"]
            # 10% of 200 = 20 val, 180 train
            check("V1 fallback: train ~180", ds["train"] == 180,
                  f"got {ds['train']}")
            # val field omitted when no canonical val (by design)
            check("V1 fallback: val field absent (no canonical val)",
                  "val" not in ds,
                  f"got val={ds.get('val')}")


def test_issue1_regression_with_val():
    """Issue 1: Regression labels also use canonical val split."""
    print("\n=== Test: Issue 1 — Regression with canonical val ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "loc",  # regression label
        ])
        check("Regression MLP exits cleanly", rc == 0, f"rc={rc}")
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "loc", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        check("Regression task_type correct", results["task_type"] == "regression")
        ds = results["data_stats"]
        check("Regression uses full train (417)", ds["train"] == 417, f"got {ds['train']}")
        check("Regression uses canonical val (52)", ds.get("val") == 52, f"got {ds.get('val')}")


def test_issue2_result_format():
    """Issue 2: Result JSON keys are stripped, macro_f1 present, head_type present."""
    print("\n=== Test: Issue 2 — Result JSON format ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "problems",
            "--model", "bert",
            "--dataset", "openml_1063",
        ])
        check("Format test exits cleanly", rc == 0)
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "problems", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        tr = results["test_results"]
        # Keys should be stripped (no test_ prefix)
        check("Keys stripped: 'accuracy' not 'test_accuracy'",
              "accuracy" in tr and "test_accuracy" not in tr,
              f"keys: {list(tr.keys())}")
        check("macro_f1 present", "macro_f1" in tr, f"keys: {list(tr.keys())}")
        check("weighted_f1 present", "weighted_f1" in tr)
        check("loss present (stripped)", "loss" in tr and "test_loss" not in tr)
        check("head_type field present", "head_type" in results,
              f"keys: {list(results.keys())}")
        check("head_type value is 'mlp'", results.get("head_type") == "mlp")
        check("model field populated", results.get("model") == "bert")
        check("dataset field populated", results.get("dataset") == "openml_1063")

    # Also check linear probe format
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "linear",
            "--seed", "42",
            "--label_column", "problems",
        ])
        if rc == 0:
            results_file = os.path.join(tmpdir, "problems", "results.json")
            with open(results_file) as f:
                results = json.load(f)
            check("Linear head_type is 'linear'", results.get("head_type") == "linear")

            tr = results["test_results"]
            # Regression metrics for linear
            check("Linear: accuracy stripped", "accuracy" in tr and "test_accuracy" not in tr)
            check("Linear: macro_f1 present", "macro_f1" in tr)


def test_issue2_regression_metrics():
    """Issue 2: Regression results have mse, rmse, r2, mae for MLP and linear."""
    print("\n=== Test: Issue 2 — Regression metrics ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "loc",
        ])
        check("Regression MLP exits cleanly", rc == 0)
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "loc", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        tr = results["test_results"]
        train_results = results.get("train_results", {})
        check("MLP regression has 'mse'", "mse" in tr)
        check("MLP regression has 'rmse'", "rmse" in tr)
        check("MLP regression has 'r2'", "r2" in tr)
        check("MLP regression has 'mae'", "mae" in tr)
        check("MLP regression has 'pearson_r'", "pearson_r" in tr)
        check("MLP regression has 'spearman_r'", "spearman_r" in tr)
        check("MLP regression has no 'accuracy'", "accuracy" not in tr)
        check("MLP regression train_results present", isinstance(train_results, dict))
        check("MLP regression train_results has 'mse'", "mse" in train_results)
        check("MLP regression train_results has 'rmse'", "rmse" in train_results)
        check("MLP regression keeps scaled_test_results", "scaled_test_results" in results)
        check("MLP regression marks target_zscore", results.get("target_zscore") is True)

    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "linear",
            "--seed", "42",
            "--label_column", "loc",
        ])
        check("Regression linear exits cleanly", rc == 0)
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "loc", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        tr = results["test_results"]
        check("Linear regression has 'mse'", "mse" in tr)
        check("Linear regression has 'rmse'", "rmse" in tr)
        check("Linear regression has 'r2'", "r2" in tr)
        check("Linear regression has 'mae'", "mae" in tr)
        check("Linear regression has 'pearson_r'", "pearson_r" in tr)
        check("Linear regression has 'spearman_r'", "spearman_r" in tr)
        check("Linear regression has 'train_mse'", "train_mse" in tr)
        check("Linear regression has 'train_rmse'", "train_rmse" in tr)
        check("Linear regression has 'train_r2'", "train_r2" in tr)
        check("Linear regression has 'train_mae'", "train_mae" in tr)
        check("Linear regression has no 'accuracy'", "accuracy" not in tr)


def test_issue3_mlp_normalization():
    """Issue 3: MLP path applies z-score normalization (results differ from unnormalized)."""
    print("\n=== Test: Issue 3 — MLP z-score normalization ===")
    # We verify indirectly: the MLP path should produce reasonable results
    # even on embeddings with large scale differences. We test that results
    # contain finite, reasonable metric values.
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "problems",
        ])
        check("MLP with normalization exits cleanly", rc == 0)
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "problems", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        acc = results["test_results"].get("accuracy")
        check("MLP accuracy is a finite float", isinstance(acc, float) and 0 <= acc <= 1,
              f"got {acc}")


def test_issue4_dead_code():
    """Issue 4: downstream_utils.py is deleted, no references remain."""
    print("\n=== Test: Issue 4 — Dead code removal ===")
    dead_file = SCRIPT_DIR / "utils" / "downstream_utils.py"
    check("downstream_utils.py does not exist", not dead_file.exists(),
          f"file still exists at {dead_file}")

    # Check no imports reference it (exclude this test file)
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "import.*downstream_utils",
         str(SCRIPT_DIR), "--exclude=smoke_test.py"],
        capture_output=True, text=True,
    )
    check("No imports of downstream_utils", result.stdout.strip() == "",
          f"found: {result.stdout.strip()}")


def test_issue5_heuristic_warning():
    """Issue 5: Heuristic auto-detection prints WARNING when metadata lacks task_type."""
    print("\n=== Test: Issue 5 — Heuristic warning ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data WITHOUT label_task_types
        v1_dir = os.path.join(tmpdir, "no_task_type")
        os.makedirs(v1_dir)
        np.random.seed(456)
        n_train, n_test, dim = 100, 30, 16
        np.save(os.path.join(v1_dir, "train_embeddings.npy"),
                np.random.randn(n_train, dim).astype(np.float32))
        np.save(os.path.join(v1_dir, "test_embeddings.npy"),
                np.random.randn(n_test, dim).astype(np.float32))
        np.save(os.path.join(v1_dir, "train_labels.npy"),
                np.random.randint(0, 5, size=n_train).astype(np.int64))
        np.save(os.path.join(v1_dir, "test_labels.npy"),
                np.random.randint(0, 5, size=n_test).astype(np.int64))
        meta = {
            "version": "1.0",
            "format": "unified_row_embedding",
            "embedding_dim": dim,
            "label_column": "test_label",
            # No label_task_types — forces heuristic
        }
        with open(os.path.join(v1_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        out_dir = os.path.join(tmpdir, "results")
        rc, stdout, stderr = run_train([
            "--embedding_dir", v1_dir,
            "--output_dir", out_dir,
            "--head_type", "mlp",
            "--seed", "42",
        ])
        combined = stdout + stderr
        check("Heuristic WARNING printed",
              "WARNING: No task_type in metadata" in combined,
              f"not found in output")
        check("Auto-detected as classification (5 classes < 20)",
              "Auto-detected task type: CLASSIFICATION" in combined,
              "expected classification detection")


def test_issue5_ambiguous_heuristic():
    """Issue 5: Ambiguous heuristic (integer labels, >20 unique) prints extra warning."""
    print("\n=== Test: Issue 5 — Ambiguous heuristic warning ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        v1_dir = os.path.join(tmpdir, "ambiguous")
        os.makedirs(v1_dir)
        np.random.seed(789)
        n_train, n_test, dim = 200, 50, 16
        # 30 unique integer labels — heuristic will pick regression
        np.save(os.path.join(v1_dir, "train_embeddings.npy"),
                np.random.randn(n_train, dim).astype(np.float32))
        np.save(os.path.join(v1_dir, "test_embeddings.npy"),
                np.random.randn(n_test, dim).astype(np.float32))
        np.save(os.path.join(v1_dir, "train_labels.npy"),
                np.random.randint(0, 30, size=n_train).astype(np.int64))
        np.save(os.path.join(v1_dir, "test_labels.npy"),
                np.random.randint(0, 30, size=n_test).astype(np.int64))
        meta = {
            "version": "1.0",
            "format": "unified_row_embedding",
            "embedding_dim": dim,
            "label_column": "many_class",
        }
        with open(os.path.join(v1_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        out_dir = os.path.join(tmpdir, "results")
        rc, stdout, stderr = run_train([
            "--embedding_dir", v1_dir,
            "--output_dir", out_dir,
            "--head_type", "linear",
            "--seed", "42",
        ])
        combined = stdout + stderr
        check("Ambiguous WARNING printed",
              "Heuristic chose regression but labels are integers" in combined,
              "expected ambiguity warning")


def test_issue6_best_model_print():
    """Issue 6: best_model.pt NOT printed for linear probe, YES for MLP."""
    print("\n=== Test: Issue 6 — best_model.pt print guard ===")
    # Linear probe: should NOT print best_model.pt
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "linear",
            "--seed", "42",
            "--label_column", "problems",
        ])
        combined = stdout + stderr
        check("Linear: no best_model.pt in output",
              "best_model.pt" not in combined,
              "best_model.pt should not appear for linear probe")

    # MLP: SHOULD print best_model.pt
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "problems",
        ])
        combined = stdout + stderr
        check("MLP: best_model.pt in output",
              "best_model.pt" in combined,
              "best_model.pt should appear for MLP")


def test_multi_label_iteration():
    """All labels: script iterates over all labels when no --label_column given."""
    print("\n=== Test: Multi-label iteration ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "linear",
            "--seed", "42",
        ])
        check("Multi-label exits cleanly", rc == 0, f"rc={rc}")
        if rc != 0:
            return

        # Should have results for both labels: problems (classification) and loc (regression)
        problems_file = os.path.join(tmpdir, "problems", "results.json")
        loc_file = os.path.join(tmpdir, "loc", "results.json")
        check("problems/results.json exists", os.path.exists(problems_file))
        check("loc/results.json exists", os.path.exists(loc_file))

        if os.path.exists(problems_file):
            with open(problems_file) as f:
                r = json.load(f)
            check("problems is classification", r["task_type"] == "classification")

        if os.path.exists(loc_file):
            with open(loc_file) as f:
                r = json.load(f)
            check("loc is regression", r["task_type"] == "regression")


def test_partial_val_edge_case():
    """Edge case: val embeddings exist but val labels don't — should degrade gracefully."""
    print("\n=== Test: Edge case — partial val (embeddings, no labels) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy v2 dir but remove val label files
        partial_dir = os.path.join(tmpdir, "partial_val")
        shutil.copytree(str(V2_EMB_DIR), partial_dir)
        # Remove val label files
        for f in Path(partial_dir).glob("val_labels_*.npy"):
            f.unlink()

        out_dir = os.path.join(tmpdir, "results")
        rc, stdout, stderr = run_train([
            "--embedding_dir", partial_dir,
            "--output_dir", out_dir,
            "--head_type", "mlp",
            "--seed", "42",
            "--label_column", "problems",
        ])
        check("Partial val: MLP exits cleanly", rc == 0, f"rc={rc}")
        if rc != 0:
            return

        results_file = os.path.join(out_dir, "problems", "results.json")
        with open(results_file) as f:
            results = json.load(f)
        ds = results["data_stats"]
        # Should fallback to ad-hoc split (no canonical val)
        # Train should be < 417 (some carved off as val)
        check("Partial val: falls back to ad-hoc split",
              ds["train"] < 417,
              f"train={ds['train']} (expected < 417 from ad-hoc split)")


def test_dummy_baseline():
    """Dummy baseline: correct format, head_type, no features used."""
    print("\n=== Test: Dummy baseline (classification + regression) ===")

    # Classification dummy
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "dummy",
            "--seed", "42",
            "--label_column", "problems",
            "--model", "bert",
            "--dataset", "openml_1063",
        ])
        check("Dummy classification exits cleanly", rc == 0,
              f"rc={rc}\nstderr={stderr[:500] if stderr else ''}")
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "problems", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        check("Dummy head_type is 'dummy'", results.get("head_type") == "dummy")
        check("Dummy task_type is 'classification'",
              results.get("task_type") == "classification")

        tr = results["test_results"]
        check("Dummy: accuracy stripped", "accuracy" in tr and "test_accuracy" not in tr,
              f"keys: {list(tr.keys())}")
        check("Dummy: macro_f1 present", "macro_f1" in tr)
        acc = tr.get("accuracy", -1)
        check("Dummy: accuracy is a valid float", isinstance(acc, float) and 0 <= acc <= 1,
              f"got {acc}")

    # Regression dummy
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, stderr = run_train([
            "--embedding_dir", str(V2_EMB_DIR),
            "--output_dir", tmpdir,
            "--head_type", "dummy",
            "--seed", "42",
            "--label_column", "loc",
        ])
        check("Dummy regression exits cleanly", rc == 0,
              f"rc={rc}\nstderr={stderr[:500] if stderr else ''}")
        if rc != 0:
            return

        results_file = os.path.join(tmpdir, "loc", "results.json")
        with open(results_file) as f:
            results = json.load(f)

        check("Dummy regression head_type is 'dummy'",
              results.get("head_type") == "dummy")
        check("Dummy regression task_type is 'regression'",
              results.get("task_type") == "regression")

        tr = results["test_results"]
        check("Dummy regression: mse present", "mse" in tr)
        check("Dummy regression: r2 present", "r2" in tr)
        r2 = tr.get("r2", 999)
        check("Dummy regression: r2 <= 0 (predicting mean)",
              isinstance(r2, float) and r2 <= 0.01,
              f"got r2={r2}")


def main():
    global passed, failed

    if not V2_EMB_DIR.exists():
        print(f"ERROR: Test data not found at {V2_EMB_DIR}")
        print("Cannot run smoke tests without real embedding data.")
        return 1

    print("=" * 70)
    print("Row Prediction Smoke Tests — Comprehensive")
    print("=" * 70)

    tests = [
        test_issue1_canonical_val_mlp,
        test_issue1_canonical_val_linear,
        test_issue1_v1_fallback,
        test_issue1_regression_with_val,
        test_issue2_result_format,
        test_issue2_regression_metrics,
        test_issue3_mlp_normalization,
        test_issue4_dead_code,
        test_issue5_heuristic_warning,
        test_issue5_ambiguous_heuristic,
        test_issue6_best_model_print,
        test_multi_label_iteration,
        test_partial_val_edge_case,
        test_dummy_baseline,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            failed += 1
            msg = f"  CRASH: {test_fn.__name__}: {e}"
            print(msg)
            errors.append(msg)
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  {e}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
