#!/usr/bin/env python3
"""
Submit DLTE pipeline jobs to SLURM with automatic prerequisite detection.

Scans generated sbatch scripts, checks which prerequisites are met, and
submits only the jobs that are ready to run. Idempotent — rerun after jobs
finish to submit the next stage.

Usage:
    # Show what's ready, blocked, and done (dry run)
    python slurm/submit_dlte.py --dry-run

    # Submit all ready jobs
    python slurm/submit_dlte.py

    # Submit only a specific stage
    python slurm/submit_dlte.py --stages retrieval
    python slurm/submit_dlte.py --stages alignment merge

    # Filter to specific models
    python slurm/submit_dlte.py --col-models bert tabert --row-models vime dae

    # Watch mode: rescan and submit every N minutes
    python slurm/submit_dlte.py --watch 5

    # Skip already-completed jobs (default: on)
    python slurm/submit_dlte.py --rerun   # resubmit even if output exists
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


STAGES = ["retrieval", "alignment", "merge"]

STAGE_DIRS = {
    "retrieval": "dlte_retrieval",
    "alignment": "dlte_alignment",
    "merge": "dlte_merge",
}


def get_project_root():
    return Path(__file__).resolve().parent.parent


def parse_script_vars(script_path):
    """Extract key variables from a generated sbatch script."""
    text = script_path.read_text()
    vals = {}
    for pattern, key in [
        (r'^MODEL="(.*)"', "model"),
        (r'^COL_MODEL="(.*)"', "col_model"),
        (r'^ROW_MODEL="(.*)"', "row_model"),
        (r'^TABLE_MODEL="(.*)"', "table_model"),
        (r'^OUTPUT_ROOT="(.*)"', "output_root"),
    ]:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            vals[key] = m.group(1)
    return vals


def check_retrieval_done(output_root, model):
    """Stage 1 is done if topk_100.jsonl exists."""
    return (Path(output_root) / "stage1" / model / "topk_100.jsonl").exists()


def check_alignment_done(output_root, stage2_key):
    """Stage 2 is done if aligned_classified_topk_100.jsonl exists."""
    return (Path(output_root) / "stage2" / stage2_key / "aligned_classified_topk_100.jsonl").exists()


def check_merge_done(output_root, combo_name):
    """Merge+eval is done if end_to_end.json exists in metrics."""
    return (Path(output_root) / "metrics" / combo_name / "end_to_end.json").exists()


def derive_stage2_key(table_model, col_model):
    if table_model and table_model != col_model:
        return f"{table_model}__{col_model}"
    return col_model


def is_running_or_pending(job_name, active_jobs):
    """Check if a job with this name is already in the SLURM queue."""
    return job_name in active_jobs


def get_active_jobs():
    """Get set of active (RUNNING/PENDING) job names from squeue."""
    try:
        result = subprocess.run(
            ["squeue", "-u", subprocess.check_output(["whoami"]).decode().strip(),
             "--format=%j", "--noheader"],
            capture_output=True, text=True, check=True)
        return set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def classify_scripts(scripts_dir, stage, active_jobs, rerun=False,
                     col_models=None, row_models=None, table_models=None):
    """Classify scripts into ready, blocked, done, and active."""
    ready, blocked, done, active = [], [], [], []

    if not scripts_dir.exists():
        return ready, blocked, done, active

    for script in sorted(scripts_dir.glob("*.sbatch")):
        v = parse_script_vars(script)
        output_root = v.get("output_root", "")

        # Extract job name from SBATCH header
        text = script.read_text()
        m = re.search(r'#SBATCH --job-name=(\S+)', text)
        job_name = m.group(1) if m else script.stem

        # Apply model filters
        # Note: retrieval scripts set MODEL (the table model for Stage 1),
        # not COL_MODEL. --col-models should NOT filter retrieval, because
        # cross-model alignments (e.g. tuta__bert) need retrieval for tuta
        # even when --col-models=bert. Only --table-models filters retrieval.
        # For alignment/merge, the effective table model is table_model if
        # set, otherwise col_model (coupled scripts leave TABLE_MODEL="").
        if stage == "retrieval":
            model = v.get("model", "")
            if table_models and model not in table_models:
                continue
        elif stage == "alignment":
            col_model = v.get("col_model", v.get("model", ""))
            table_model = v.get("table_model", "")
            effective_table = table_model if table_model else col_model
            if col_models and col_model not in col_models:
                continue
            if table_models and effective_table not in table_models:
                continue
        elif stage == "merge":
            col_model = v.get("col_model", "")
            table_model = v.get("table_model", "")
            effective_table = table_model if table_model else col_model
            if col_models and col_model not in col_models:
                continue
            if row_models and v.get("row_model", "") not in row_models:
                continue
            if table_models and effective_table not in table_models:
                continue

        # Check if already in queue
        if is_running_or_pending(job_name, active_jobs):
            active.append((script, v, "in queue"))
            continue

        # Check if already done
        if not rerun:
            is_done = False
            if stage == "retrieval":
                is_done = check_retrieval_done(output_root, v.get("model", ""))
            elif stage == "alignment":
                col_model = v.get("col_model", v.get("model", ""))
                table_model = v.get("table_model", "")
                s2key = derive_stage2_key(table_model, col_model)
                is_done = check_alignment_done(output_root, s2key)
            elif stage == "merge":
                table_model = v.get("table_model", "")
                col_model = v.get("col_model", "")
                row_model = v.get("row_model", "")
                s2key = derive_stage2_key(table_model, col_model)
                combo = f"{s2key}__{row_model}"
                is_done = check_merge_done(output_root, combo)
            if is_done:
                done.append((script, v, "output exists"))
                continue

        # Check prerequisites
        prereqs_met = True
        reason = ""
        if stage == "alignment":
            col_model = v.get("col_model", v.get("model", ""))
            table_model = v.get("table_model", "")
            s1_model = table_model if table_model else col_model
            if not check_retrieval_done(output_root, s1_model):
                prereqs_met = False
                reason = f"need stage1/{s1_model}"
        elif stage == "merge":
            table_model = v.get("table_model", "")
            col_model = v.get("col_model", "")
            s2key = derive_stage2_key(table_model, col_model)
            if not check_alignment_done(output_root, s2key):
                prereqs_met = False
                reason = f"need stage2/{s2key}"

        if prereqs_met:
            ready.append((script, v, "ready"))
        else:
            blocked.append((script, v, reason))

    return ready, blocked, done, active


def submit_job(script_path, dry_run=False):
    """Submit a single sbatch script. Returns (success, job_id_or_error)."""
    if dry_run:
        return True, "DRY-RUN"
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        job_id = output.split()[-1] if "Submitted" in output else output
        return True, job_id
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, "sbatch not found"


def print_stage_summary(stage, ready, blocked, done, active, verbose=False):
    """Print summary for one stage."""
    total = len(ready) + len(blocked) + len(done) + len(active)
    if total == 0:
        return

    print(f"\n  {stage.upper()} ({total} scripts)")
    print(f"    Done: {len(done)}  |  Ready: {len(ready)}  |  "
          f"Active: {len(active)}  |  Blocked: {len(blocked)}")

    if verbose and blocked:
        # Group blocked by reason
        reasons = {}
        for _, v, reason in blocked:
            reasons.setdefault(reason, 0)
            reasons[reason] += 1
        for reason, count in sorted(reasons.items()):
            print(f"    Blocked ({count}): {reason}")

    if verbose and active:
        for script, v, _ in active[:5]:
            print(f"    Active: {script.stem}")
        if len(active) > 5:
            print(f"    ... and {len(active) - 5} more")


def run_once(args, scripts_base, active_jobs):
    """Run one pass: classify all stages, submit ready jobs. Returns counts."""
    stages = args.stages or STAGES
    total_submitted = 0
    total_blocked = 0
    total_done = 0

    for stage in stages:
        scripts_dir = scripts_base / STAGE_DIRS[stage]

        ready, blocked, done, active = classify_scripts(
            scripts_dir, stage, active_jobs,
            rerun=args.rerun,
            col_models=args.col_models,
            row_models=args.row_models,
            table_models=args.table_models)

        print_stage_summary(stage, ready, blocked, done, active,
                            verbose=args.verbose or args.dry_run)

        if ready:
            submitted = 0
            failed = 0
            for script, v, _ in ready:
                success, result = submit_job(script, args.dry_run)
                if success:
                    submitted += 1
                    if args.verbose:
                        print(f"    Submitted: {script.stem} -> {result}")
                else:
                    failed += 1
                    print(f"    FAILED: {script.stem} -> {result}")
                if not args.dry_run:
                    time.sleep(args.delay)

            action = "Would submit" if args.dry_run else "Submitted"
            print(f"    {action}: {submitted}" +
                  (f", failed: {failed}" if failed else ""))
            total_submitted += submitted

        total_blocked += len(blocked)
        total_done += len(done)

    return total_submitted, total_blocked, total_done


def main():
    parser = argparse.ArgumentParser(
        description="Submit DLTE pipeline jobs with automatic prerequisite detection",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        help="Only process these stages (default: all)")
    parser.add_argument("--col-models", nargs="+", dest="col_models",
                        help="Filter to these column models")
    parser.add_argument("--row-models", nargs="+", dest="row_models",
                        help="Filter to these row models")
    parser.add_argument("--table-models", nargs="+", dest="table_models",
                        help="Filter to these table models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be submitted without submitting")
    parser.add_argument("--rerun", action="store_true",
                        help="Resubmit even if output already exists")
    parser.add_argument("--watch", type=int, metavar="MINUTES", default=None,
                        help="Rescan and submit every N minutes until all done")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between submissions in seconds (default: 0.3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    project_root = get_project_root()
    scripts_base = project_root / "slurm" / "scripts" / "generated" / "downstream"

    if not scripts_base.exists():
        print(f"Error: Scripts directory not found: {scripts_base}")
        print("Run generate_downstream_scripts.py first.")
        sys.exit(1)

    iteration = 0
    while True:
        iteration += 1
        if args.watch and iteration > 1:
            print(f"\n{'=' * 60}")
        print(f"DLTE Pipeline Status" +
              (f" (iteration {iteration})" if args.watch else ""))
        print("=" * 60)

        active_jobs = get_active_jobs()
        submitted, blocked, done = run_once(args, scripts_base, active_jobs)

        # Summary
        stages = args.stages or STAGES
        total_scripts = 0
        for stage in stages:
            d = scripts_base / STAGE_DIRS[stage]
            if d.exists():
                total_scripts += len(list(d.glob("*.sbatch")))

        print(f"\n{'─' * 60}")
        action = "Would submit" if args.dry_run else "Submitted"
        print(f"  {action}: {submitted}  |  Done: {done}  |  Blocked: {blocked}")

        if not args.watch:
            break

        if blocked == 0 and submitted == 0:
            print("\nAll jobs complete or in queue. Done.")
            break

        if args.dry_run:
            print("\nDry run — not entering watch loop.")
            break

        print(f"\nNext scan in {args.watch} minutes... (Ctrl+C to stop)")
        try:
            time.sleep(args.watch * 60)
        except KeyboardInterrupt:
            print("\nStopped.")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
