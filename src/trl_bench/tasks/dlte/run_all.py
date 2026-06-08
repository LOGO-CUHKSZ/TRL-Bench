#!/usr/bin/env python3
"""
Run all DLTE pipeline combinations in parallel (no SLURM required).

Replicates the full set of jobs that generate_downstream_scripts.py would
produce for dlte_retrieval / dlte_alignment / dlte_merge, and executes
them directly via subprocess with concurrent.futures parallelism.

Stages run sequentially (1 → 2 → 3); jobs within a stage run in parallel.
Already-completed jobs are skipped automatically.

Usage:
    python run_dlte_all.py                       # run everything
    python run_dlte_all.py --max-workers 80      # cap parallelism
    python run_dlte_all.py --stage 1             # only Stage 1
    python run_dlte_all.py --dry-run             # show job counts

    # Use custom assets location:
    python run_dlte_all.py --assets-dir /path/to/dlte_assets

    # Or individually:
    python run_dlte_all.py --embeddings-dir /path/to/embeddings --results-dir /path/to/results
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Repo root: this file is src/trl_bench/tasks/dlte/run_all.py -> parents[4].
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# The DLTE stage scripts live next
# to this module under scripts/ (the original used downstream_tasks/dlte/scripts).
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
PYTHON = sys.executable

DLTE_TABLE_PKLS = ["dlte_v1_queries.pkl", "dlte_v1_targets.pkl", "ckan_subset.pkl"]
DLTE_ROW_PKLS = ["dlte_v1_queries.pkl", "dlte_v1_targets.pkl"]

# These globals are set by main() from CLI args before any discovery/jobs run.
TABLE_EMB_ROOT = None
COL_EMB_ROOT = None
ROW_EMB_ROOT = None
RESULTS_BASE = None
DATA_ROOT = None


# ── Discovery (mirrors generate_downstream_scripts.py logic) ─────

def _probe_pkl_variants(pkl_path):
    """Return set of non-None variant keys in a table embedding pkl."""
    _LEGACY = {"column_sum"}
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return None
        table_emb = data[0].get("table_embedding", {})
        if not isinstance(table_emb, dict):
            return {"column_mean"}
        variants = {k for k, v in table_emb.items() if v is not None} - _LEGACY
        return variants or None
    except Exception:
        return None


def discover_table_variants():
    """Return {model: [variant, ...]} from table embedding pkls (intersection)."""
    result = {}
    for model_dir in sorted(TABLE_EMB_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        if name.endswith("_bert_hybrid") or "_backup_" in name:
            continue
        pkls = [model_dir / p for p in DLTE_TABLE_PKLS]
        if not all(p.exists() for p in pkls):
            continue
        sets = []
        for p in pkls:
            vs = _probe_pkl_variants(p)
            if vs is None:
                break
            sets.append(vs)
        else:
            common = sets[0]
            for s in sets[1:]:
                common &= s
            if common:
                result[name] = sorted(common)
    return result


def discover_column_models():
    return sorted(
        d.name for d in COL_EMB_ROOT.iterdir()
        if d.is_dir() and all((d / p).exists() for p in DLTE_TABLE_PKLS)
    )


def discover_row_models():
    return sorted(
        d.name for d in ROW_EMB_ROOT.iterdir()
        if d.is_dir() and all((d / p).exists() for p in DLTE_ROW_PKLS)
    )


# ── Helpers ──────────────────────────────────────────────────────

def output_root_for(variant):
    return RESULTS_BASE if variant == "column_mean" else RESULTS_BASE / variant


def stage2_key(table_model, col_model):
    if table_model and table_model != col_model:
        return f"{table_model}__{col_model}"
    return col_model


# ── Completion checks ────────────────────────────────────────────

def s1_done(oroot, model):
    return (oroot / "stage1" / model / "topk_100.jsonl").exists()


def s2_done(oroot, key):
    return (oroot / "stage2" / key / "aligned_classified_topk_100.jsonl").exists()


def s3_done(oroot, combo):
    return (oroot / "metrics" / combo / "end_to_end.json").exists()


# ── Job runners (called in worker processes) ─────────────────────

def _run(cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=str(PROJECT_ROOT),
    )


def _emb_root_args():
    """Return --embeddings_root args if using a custom embeddings location."""
    if TABLE_EMB_ROOT and TABLE_EMB_ROOT.parent != PROJECT_ROOT / "assets" / "embeddings":
        return ["--embeddings_root", str(TABLE_EMB_ROOT.parent)]
    return []


def _data_root_args():
    """Return --data_root so the step scripts find ``datasets/dlte_v1/``.

    The reference implementation defaulted DATASET_ROOT to ``<project_root>/datasets/
    dlte_v1``; in this repo the DLTE data is staged under ``data/dlte_v1/`` (so
    ``<DATA_ROOT>/datasets/dlte_v1/ground_truth/query_tasks.jsonl``). Without
    this, every stage fails at step8:61 (FileNotFound on query_tasks.jsonl)."""
    if DATA_ROOT is not None:
        return ["--data_root", str(DATA_ROOT)]
    return []


def run_s1(args):
    model, variant = args
    oroot = output_root_for(variant)
    tag = f"{model}/{variant}"
    if s1_done(oroot, model):
        return (tag, "skip")
    try:
        _run([
            PYTHON, str(SCRIPTS_DIR / "step8_faiss_retrieval.py"),
            "--models", model,
            "--topk", "10", "50", "100",
            "--table_variant", variant,
            "--output_root", str(oroot),
            "--project_root", str(PROJECT_ROOT),
        ] + _emb_root_args() + _data_root_args())
        return (tag, "ok")
    except subprocess.CalledProcessError as e:
        return (tag, f"FAIL: {(e.stderr or '')[-300:]}")


def run_s2(args):
    col_model, variant, table_model = args
    oroot = output_root_for(variant)
    key = stage2_key(table_model, col_model)
    tag = f"{key}/{variant}"
    if s2_done(oroot, key):
        return (tag, "skip")
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "step9_column_alignment.py"),
        "--models", col_model,
        "--topk", "100",
        "--output_root", str(oroot),
        "--project_root", str(PROJECT_ROOT),
    ] + _emb_root_args() + _data_root_args()
    if table_model and table_model != col_model:
        cmd.extend(["--table_model", table_model])
    try:
        _run(cmd)
        return (tag, "ok")
    except subprocess.CalledProcessError as e:
        return (tag, f"FAIL: {(e.stderr or '')[-300:]}")


def run_s3(args):
    """Run a grouped Stage 3 job: one row_model × multiple col_models."""
    row_model, variant, table_model, col_models_for_group = args
    oroot = output_root_for(variant)

    # Filter to col_models that aren't already done
    pending = []
    for col_model in col_models_for_group:
        key = stage2_key(table_model, col_model)
        combo = f"{key}__{row_model}"
        if not s3_done(oroot, combo):
            pending.append(col_model)

    tag = f"{row_model}/{table_model}/{variant} ({len(col_models_for_group)} cols)"
    if not pending:
        return (tag, "skip", len(col_models_for_group), 0)

    cmd = [
        PYTHON, str(SCRIPTS_DIR / "step10_row_matching.py"),
        "--col_models", *pending,
        "--row_models", row_model,
        "--splits", "dev", "test", "train",
        "--output_root", str(oroot),
        "--project_root", str(PROJECT_ROOT),
    ] + _emb_root_args() + _data_root_args()
    if table_model:
        cmd.extend(["--table_model", table_model])
    try:
        _run(cmd)
        return (tag, "ok", len(pending), len(col_models_for_group) - len(pending))
    except subprocess.CalledProcessError as e:
        return (tag, f"FAIL: {(e.stderr or '')[-300:]}", 0, 0)


# ── Orchestrator ─────────────────────────────────────────────────

def run_stage(name, jobs, runner, max_workers, dry_run=False):
    n = len(jobs)
    if n == 0:
        print(f"  {name}: nothing to do")
        return 0, 0, 0
    if dry_run:
        print(f"  {name}: {n} jobs (dry-run)")
        return 0, 0, 0

    w = min(n, max_workers)
    print(f"\n{'='*60}")
    print(f"  {name}: {n} jobs, {w} workers")
    print(f"{'='*60}")

    t0 = time.time()
    ok = skip = fail = 0
    failures = []

    with ProcessPoolExecutor(max_workers=w) as pool:
        futures = {pool.submit(runner, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            tag, status = result[0], result[1]
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failures.append((tag, status))
            if i % max(1, n // 20) == 0 or i == n:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n - i) / rate if rate > 0 else 0
                print(f"  [{i:>{len(str(n))}}/{n}] ok={ok} skip={skip} fail={fail}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  {name} complete: {ok} ok, {skip} skipped, {fail} failed  ({elapsed:.1f}s)")
    for tag, status in failures[:20]:
        print(f"    FAILED: {tag}: {status[:120]}")
    if len(failures) > 20:
        print(f"    ... and {len(failures) - 20} more failures")
    return ok, skip, fail


def main():
    parser = argparse.ArgumentParser(description="Run all DLTE pipeline combinations in parallel")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers for all stages (overridden by per-stage args)")
    parser.add_argument("--s1-workers", type=int, default=None,
                        help="Max workers for Stage 1 (retrieval)")
    parser.add_argument("--s2-workers", type=int, default=None,
                        help="Max workers for Stage 2 (alignment)")
    parser.add_argument("--s3-workers", type=int, default=None,
                        help="Max workers for Stage 3 (merge+eval)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3],
                        help="Only run this stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show job counts without running")
    parser.add_argument("--assets-dir", type=str, default=None,
                        help="Custom assets root (expects embeddings/ and evaluation_results/ subdirs)")
    parser.add_argument("--embeddings-dir", type=str, default=None,
                        help="Custom embeddings root (expects table/, column/, row/ subdirs)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Custom results root for DLTE outputs")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="DLTE data root containing datasets/dlte_v1/ "
                             "(default: <project_root>/data/dlte_v1)")
    args = parser.parse_args()

    # ── Resolve paths ──
    global TABLE_EMB_ROOT, COL_EMB_ROOT, ROW_EMB_ROOT, RESULTS_BASE, DATA_ROOT

    if args.assets_dir:
        assets = Path(args.assets_dir).resolve()
        emb_base = assets / "embeddings"
        results_base = assets / "evaluation_results" / "dlte"
    else:
        emb_base = PROJECT_ROOT / "assets" / "embeddings"
        results_base = PROJECT_ROOT / "assets" / "evaluation_results" / "dlte"

    if args.embeddings_dir:
        emb_base = Path(args.embeddings_dir).resolve()
    if args.results_dir:
        results_base = Path(args.results_dir).resolve()

    TABLE_EMB_ROOT = emb_base / "table"
    COL_EMB_ROOT = emb_base / "column"
    ROW_EMB_ROOT = emb_base / "row"
    RESULTS_BASE = results_base
    DATA_ROOT = (Path(args.data_dir).resolve() if args.data_dir
                 else PROJECT_ROOT / "data" / "dlte_v1")

    print(f"Embeddings: {emb_base}")
    print(f"Results:    {RESULTS_BASE}")

    # ── Discover ──
    print("Discovering embeddings...")
    table_variants = discover_table_variants()
    col_models = discover_column_models()
    row_models = discover_row_models()

    # Build (table_model, variant) list for Stage 1
    s1_jobs = []
    for model, variants in sorted(table_variants.items()):
        for v in variants:
            s1_jobs.append((model, v))

    # Stage 2: every (table_model, variant) × every col_model
    s2_jobs = []
    for table_model, variant in s1_jobs:
        for col_model in col_models:
            s2_jobs.append((col_model, variant, table_model))

    # Stage 3: group by (row_model, table_model, variant) → pass all col_models
    # This loads row embeddings once per group instead of once per combo.
    from collections import defaultdict
    s3_groups = defaultdict(list)  # (row_model, variant, table_model) -> [col_models]
    s3_combo_count = 0
    for table_model, variant in s1_jobs:
        for col_model in col_models:
            for row_model in row_models:
                s3_groups[(row_model, variant, table_model)].append(col_model)
                s3_combo_count += 1
    s3_jobs = [(rm, var, tm, cms) for (rm, var, tm), cms in sorted(s3_groups.items())]

    for model, variants in sorted(table_variants.items()):
        print(f"  table  {model}: {variants}")
    print(f"  column: {col_models}")
    print(f"  row:    {row_models}")
    print(f"\nStage 1 (retrieval):  {len(s1_jobs):>5} jobs")
    print(f"Stage 2 (alignment):  {len(s2_jobs):>5} jobs")
    print(f"Stage 3 (merge+eval): {len(s3_jobs):>5} jobs ({s3_combo_count} combos, "
          f"{len(col_models)} col_models/job)")
    print(f"Total:                {len(s1_jobs)+len(s2_jobs)+len(s3_jobs):>5} jobs")

    if args.dry_run:
        return

    # ── Parallelism settings ──
    n_cpus = os.cpu_count() or 1
    # Auto defaults — stages are sequential so full CPU is available
    w1_auto = min(len(s1_jobs), n_cpus)
    w2_auto = min(len(s2_jobs), n_cpus)
    w3_auto = min(len(s3_jobs), n_cpus)
    # Per-stage args > --max-workers > auto
    base = args.max_workers
    w1 = args.s1_workers or base or w1_auto
    w2 = args.s2_workers or base or w2_auto
    w3 = args.s3_workers or base or w3_auto

    print(f"\nWorkers: s1={w1}, s2={w2}, s3={w3}  (cpus={n_cpus})")

    t_total = time.time()
    totals = {"ok": 0, "skip": 0, "fail": 0}

    for stage_num, (name, jobs, runner, workers) in enumerate([
        ("Stage 1 (Retrieval)", s1_jobs, run_s1, w1),
        ("Stage 2 (Alignment)", s2_jobs, run_s2, w2),
        ("Stage 3 (Merge+Eval)", s3_jobs, run_s3, w3),
    ], 1):
        if args.stage and args.stage != stage_num:
            continue
        ok, skip, fail = run_stage(name, jobs, runner, workers)
        totals["ok"] += ok
        totals["skip"] += skip
        totals["fail"] += fail

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"ALL DONE: {totals['ok']} ok, {totals['skip']} skipped, {totals['fail']} failed")
    print(f"Total wall time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
