#!/usr/bin/env python3
"""
Submit downstream task evaluation jobs to SLURM.

Handles both model evaluation scripts and embedding-free baseline scripts.
DLTE tasks are excluded — use submit_dlte.py for those.

Usage:
    # Submit all downstream jobs
    python submit_downstream.py

    # Submit specific tasks
    python submit_downstream.py --tasks union_search join_search

    # Submit specific models (includes baselines like value_overlap)
    python submit_downstream.py --models bert value_overlap

    # Submit specific datasets
    python submit_downstream.py --datasets santos tus

    # Dry run
    python submit_downstream.py --dry-run

    # Skip script generation (use existing scripts)
    python submit_downstream.py --no-generate
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# DLTE tasks require multi-stage dependency tracking — use submit_dlte.py
DLTE_TASKS = {'dlte_retrieval', 'dlte_alignment', 'dlte_merge'}


def get_project_root() -> Path:
    """File at slurm/submit_downstream.py; one .parent reaches the repo root."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def submit_job(script_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Submit a single job to SLURM."""
    if dry_run:
        return True, "DRY-RUN"

    try:
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        job_id = output.split()[-1] if 'Submitted' in output else output
        return True, job_id
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, "sbatch command not found (not on a SLURM cluster?)"


def discover_from_manifest(manifest_path: Path) -> list[dict]:
    """Load script entries from the downstream manifest."""
    if not manifest_path.exists():
        return []
    with open(manifest_path, 'r') as f:
        return json.load(f)


def discover_from_scripts(scripts_dir: Path, subdir_to_task: dict) -> list[dict]:
    """Fallback discovery: parse script content for JOB_KEY, MODEL/BASELINE_NAME, DATASET, TASK."""
    entries = []
    if not scripts_dir.exists():
        return entries

    for task_subdir in sorted(scripts_dir.iterdir()):
        if not task_subdir.is_dir():
            continue

        task_name = subdir_to_task.get(task_subdir.name)
        if not task_name:
            continue

        for script_path in sorted(task_subdir.glob('*.sbatch')):
            entry = _parse_script(script_path, task_name)
            if entry:
                entries.append(entry)

    return entries


def _parse_script(script_path: Path, task_name: str) -> dict | None:
    """Parse a generated sbatch script to extract metadata."""
    try:
        content = script_path.read_text()
    except OSError:
        return None

    # Extract key fields from bash variable assignments
    job_key = _extract_var(content, 'JOB_KEY')
    model = _extract_var(content, 'MODEL') or _extract_var(content, 'BASELINE_NAME')
    dataset = _extract_var(content, 'DATASET')
    head_type = _extract_var(content, 'HEAD_TYPE')
    is_baseline = 'BASELINE_NAME=' in content

    if not model or not dataset:
        return None

    # Resolve unresolved bash variable references in JOB_KEY.
    # Model templates render JOB_KEY with ${TASK} still as a bash variable
    # (resolved at runtime), but we need the concrete value for status tracking.
    if job_key and '${' in job_key:
        job_key = job_key.replace('${TASK}', task_name)
        job_key = job_key.replace('${MODEL}', model)
        job_key = job_key.replace('${DATASET}', dataset)
    if head_type and head_type != 'mlp' and job_key and not job_key.endswith(f"_{head_type}"):
        job_key = f"{job_key}_{head_type}"

    return {
        'task': task_name,
        'model': model,
        'dataset': dataset,
        'variant': None,
        'seed': None,
        'script_path': str(script_path),
        'is_baseline': is_baseline,
        'job_key': job_key or f"{model}_{dataset}_{task_name}",
    }


def _extract_var(content: str, var_name: str) -> str | None:
    """Extract a bash variable value like MODEL="bert" from script content."""
    pattern = rf'^{var_name}="([^"]*)"'
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1) if match else None


def filter_entries(
    entries: list[dict],
    task_filter: list[str] | None,
    model_filter: list[str] | None,
    dataset_filter: list[str] | None,
) -> list[dict]:
    """Apply --tasks, --models, --datasets filters."""
    result = entries
    if task_filter:
        result = [e for e in result if e['task'] in task_filter]
    if model_filter:
        result = [e for e in result if e['model'] in model_filter]
    if dataset_filter:
        result = [e for e in result if e['dataset'] in dataset_filter]
    return result


def initialize_status(status_file: Path, entries: list[dict]):
    """Initialize status file with PENDING entries."""
    status_file.parent.mkdir(parents=True, exist_ok=True)

    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
    else:
        status = {}

    timestamp = datetime.now().isoformat()
    for entry in entries:
        key = entry['job_key']
        if key not in status or status[key].get('status') != 'RUNNING':
            status[key] = {
                'status': 'PENDING',
                'message': 'Submitted',
                'timestamp': timestamp,
                'model': entry['model'],
                'dataset': entry['dataset'],
                'task': entry['task'],
                'is_baseline': entry.get('is_baseline', False),
            }

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def update_job_id(status_file: Path, job_key: str, slurm_job_id: str):
    """Persist SLURM job ID to status file."""
    import fcntl

    lock_path = str(status_file) + '.lock'
    with open(lock_path, 'w') as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
            if job_key in status:
                status[job_key]['slurm_job_id'] = slurm_job_id
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _mark_submit_failed(status_file: Path, job_key: str, error_msg: str):
    """Mark a job as SUBMIT_FAILED after sbatch fails."""
    import fcntl

    lock_path = str(status_file) + '.lock'
    with open(lock_path, 'w') as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
            if job_key in status:
                status[job_key]['status'] = 'SUBMIT_FAILED'
                status[job_key]['message'] = error_msg
                status[job_key]['timestamp'] = datetime.now().isoformat()
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(
        description='Submit downstream task evaluation jobs to SLURM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--tasks', nargs='+',
                        help='Submit jobs only for these tasks')
    parser.add_argument('--models', nargs='+',
                        help='Submit jobs only for these models/baselines')
    parser.add_argument('--datasets', nargs='+',
                        help='Submit jobs only for these datasets')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be submitted without actually submitting')
    parser.add_argument('--no-generate', action='store_true',
                        help='Skip script generation (use existing scripts)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--head-type', type=str, default='mlp',
                        choices=['mlp', 'linear', 'dummy', 'cosine_threshold', 'interaction'],
                        help='Probe head type (default: mlp). Non-MLP heads run on CPU.')
    parser.add_argument('--row-embedding-root', type=str, default=None,
                        help='Overlay row embedding root for row-level tasks '
                             '(e.g., embeddings/row_dim768)')
    parser.add_argument('--result-tag', type=str, default=None,
                        help='Row-level result tag (default: derived from --row-embedding-root)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between submissions in seconds (default: 0.5)')

    args = parser.parse_args()

    project_root = get_project_root()
    config_dir = project_root / 'slurm' / 'config'
    resources_config = load_yaml(config_dir / 'resources.yaml')
    output_dir = project_root / resources_config['paths']['downstream_scripts_dir']
    status_file = project_root / 'slurm' / 'logs' / 'downstream' / 'status' / 'job_status.json'

    # Check for DLTE tasks and warn
    if args.tasks:
        dlte_requested = [t for t in args.tasks if t in DLTE_TASKS]
        if dlte_requested:
            print(f"Note: DLTE tasks have multi-stage dependencies — use submit_dlte.py instead.")
            print(f"  Skipping: {dlte_requested}")
            args.tasks = [t for t in args.tasks if t not in DLTE_TASKS]
            if not args.tasks:
                print("No non-DLTE tasks remaining.")
                sys.exit(0)

    # Build OUTPUT_SUBDIRS reverse mapping for script discovery
    # Import from the generator to stay in sync
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_downstream_scripts import OUTPUT_SUBDIRS
    subdir_to_task = {v: k for k, v in OUTPUT_SUBDIRS.items()}

    # Generation phase
    if not args.no_generate:
        print("Generating downstream scripts...")
        gen_cmd = [sys.executable, str(project_root / 'slurm' / 'generate_downstream_scripts.py')]
        if args.tasks:
            gen_cmd += ['--tasks'] + args.tasks
        if args.datasets:
            gen_cmd += ['--datasets'] + args.datasets
        # Note: --models is NOT forwarded to the generator for model scripts
        # (the generator discovers all available embeddings). We filter at submission.
        # But for baselines, we do forward --models so only matching baselines are generated.
        if args.head_type != 'mlp':
            gen_cmd += ['--head-type', args.head_type]
        if args.row_embedding_root:
            gen_cmd += ['--row-embedding-root', args.row_embedding_root]
        if args.result_tag:
            gen_cmd += ['--result-tag', args.result_tag]
        if args.dry_run:
            gen_cmd += ['--dry-run']

        result = subprocess.run(gen_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error generating scripts:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            sys.exit(1)
        if args.verbose:
            print(result.stdout)

    # Discovery phase — manifest-first, with filesystem fallback.
    # When the manifest exists, use it as the authoritative list of scripts
    # from the current generation. Fall back to filesystem scanning only
    # when manifest is missing (e.g., --no-generate on old output).
    manifest_path = output_dir / 'downstream_manifest.json'
    manifest_entries = discover_from_manifest(manifest_path)

    if manifest_entries:
        entries = manifest_entries
        if args.verbose:
            print(f"  Using manifest: {len(entries)} entries from {manifest_path}")
    else:
        # Fallback: parse script content when no manifest is available
        entries = discover_from_scripts(output_dir, subdir_to_task)
        if args.verbose:
            print(f"  Manifest not found, discovered {len(entries)} scripts from filesystem")

    if not entries:
        print("No scripts found to submit.")
        sys.exit(0)

    # Filter
    # Exclude DLTE tasks even if they appear in discovered scripts
    entries = [e for e in entries if e['task'] not in DLTE_TASKS]

    entries = filter_entries(entries, args.tasks, args.models, args.datasets)

    if not entries:
        print("No scripts match the specified filters.")
        sys.exit(0)

    # Validate script files exist
    valid_entries = []
    for entry in entries:
        if Path(entry['script_path']).exists():
            valid_entries.append(entry)
        elif args.verbose:
            print(f"  SKIP: {entry['script_path']} (file not found)")
    entries = valid_entries

    if not entries:
        print("No valid scripts to submit (all filtered or missing).")
        sys.exit(0)

    # Initialize status (skip during dry-run to avoid filesystem side effects)
    if not args.dry_run:
        initialize_status(status_file, entries)

    # Submit
    print(f"\nSubmitting {len(entries)} jobs...")
    print("=" * 60)

    submitted = 0
    failed = 0

    for entry in entries:
        script_path = Path(entry['script_path'])
        label = f"{entry['model']}_{entry['dataset']} ({entry['task']})"

        success, result = submit_job(script_path, args.dry_run)

        if success:
            submitted += 1
            status_str = "[DRY-RUN]" if args.dry_run else f"Job {result}"
            print(f"  {label}: {status_str}")
            if not args.dry_run and result != "DRY-RUN":
                update_job_id(status_file, entry['job_key'], result)
        else:
            failed += 1
            print(f"  {label}: FAILED - {result}")
            if not args.dry_run:
                # Mark the entry as SUBMIT_FAILED so stale PENDING records
                # don't linger in the status file.
                _mark_submit_failed(status_file, entry['job_key'], result)

        if not args.dry_run and args.delay > 0:
            time.sleep(args.delay)

    print("=" * 60)
    print(f"Submitted: {submitted}")
    if failed:
        print(f"Failed:    {failed}")

    if not args.dry_run and submitted > 0:
        print(f"\nStatus file: {status_file}")


if __name__ == '__main__':
    main()
