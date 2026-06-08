#!/usr/bin/env python3
"""
Build a human-readable job tracking file from the downstream status JSON.

Reads the status file written by submit_downstream.py, enriches each entry
with GPU/CPU info parsed from the sbatch script, and queries squeue/sacct
for live status.  Writes a TSV summary to slurm/logs/downstream/job_tracker.tsv.

Usage:
    python slurm/build_job_tracker.py              # build once
    python slurm/build_job_tracker.py --refresh     # re-query slurm for latest state
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = PROJECT_ROOT / 'slurm' / 'logs' / 'downstream' / 'status' / 'job_status.json'
TRACKER_FILE = PROJECT_ROOT / 'slurm' / 'logs' / 'downstream' / 'job_tracker.tsv'


def load_status() -> dict:
    with open(STATUS_FILE) as f:
        return json.load(f)


def parse_gpu_from_script(script_path: str) -> str:
    """Return 'gpu:N' or 'cpu' based on #SBATCH --gres in the script."""
    try:
        content = Path(script_path).read_text()
    except OSError:
        return 'unknown'
    match = re.search(r'#SBATCH\s+--gres=gpu:(\S+)', content)
    if match:
        return f'gpu:{match.group(1)}'
    return 'cpu'


def parse_time_from_script(script_path: str) -> str:
    """Return the --time value from the script."""
    try:
        content = Path(script_path).read_text()
    except OSError:
        return 'unknown'
    match = re.search(r'#SBATCH\s+--time=(\S+)', content)
    return match.group(1) if match else 'unknown'


def parse_mem_from_script(script_path: str) -> str:
    """Return the --mem value from the script."""
    try:
        content = Path(script_path).read_text()
    except OSError:
        return 'unknown'
    match = re.search(r'#SBATCH\s+--mem=(\S+)', content)
    return match.group(1) if match else 'unknown'


def query_squeue() -> dict:
    """Query squeue for running/pending jobs, return {job_id: state}."""
    try:
        result = subprocess.run(
            ['squeue', '-u', subprocess.check_output(['whoami']).decode().strip(),
             '-o', '%i %T', '--noheader'],
            capture_output=True, text=True, timeout=10,
        )
        states = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        return states
    except Exception:
        return {}


def query_sacct(job_ids: list[str]) -> dict:
    """Query sacct for completed/failed jobs."""
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            ['sacct', '-j', ','.join(job_ids),
             '--format=JobID,State,ExitCode,Elapsed', '--noheader', '--parsable2'],
            capture_output=True, text=True, timeout=30,
        )
        states = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split('|')
            if len(parts) >= 4:
                jid = parts[0].split('.')[0]  # strip .batch suffix
                if jid in job_ids:
                    states[jid] = {
                        'state': parts[1],
                        'exit_code': parts[2],
                        'elapsed': parts[3],
                    }
        return states
    except Exception:
        return {}


def build_tracker(refresh: bool = False):
    status = load_status()

    # Collect job IDs for sacct query
    job_ids = [v.get('slurm_job_id', '') for v in status.values() if v.get('slurm_job_id')]

    # Query live state
    squeue_states = query_squeue() if refresh else {}
    sacct_states = query_sacct(job_ids) if refresh else {}

    # Load manifest for script paths
    manifest_path = PROJECT_ROOT / 'slurm' / 'scripts' / 'generated' / 'downstream' / 'downstream_manifest.json'
    script_lookup = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for entry in json.load(f):
                script_lookup[entry.get('job_key', '')] = entry.get('script_path', '')

    rows = []
    for job_key, info in sorted(status.items()):
        slurm_id = info.get('slurm_job_id', '')
        script_path = script_lookup.get(job_key, '')

        # Determine live status
        if refresh and slurm_id:
            if slurm_id in squeue_states:
                live_status = squeue_states[slurm_id]
            elif slurm_id in sacct_states:
                live_status = sacct_states[slurm_id]['state']
            else:
                live_status = info.get('status', 'UNKNOWN')
        else:
            live_status = info.get('status', 'UNKNOWN')

        gpu = parse_gpu_from_script(script_path) if script_path else 'unknown'
        time_limit = parse_time_from_script(script_path) if script_path else 'unknown'
        mem = parse_mem_from_script(script_path) if script_path else 'unknown'

        rows.append({
            'job_key': job_key,
            'slurm_job_id': slurm_id,
            'task': info.get('task', ''),
            'model': info.get('model', ''),
            'dataset': info.get('dataset', ''),
            'status': live_status,
            'gpu': gpu,
            'time_limit': time_limit,
            'memory': mem,
            'is_baseline': info.get('is_baseline', False),
            'submitted_at': info.get('timestamp', ''),
        })

    # Write TSV
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = ['job_key', 'slurm_job_id', 'task', 'model', 'dataset',
              'status', 'gpu', 'time_limit', 'memory', 'is_baseline', 'submitted_at']
    with open(TRACKER_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    status_counts = Counter(r['status'] for r in rows)
    gpu_counts = Counter(r['gpu'] for r in rows)

    print(f"Wrote {len(rows)} entries to {TRACKER_FILE}")
    print(f"\nStatus breakdown:")
    for s, c in status_counts.most_common():
        print(f"  {s}: {c}")
    print(f"\nResource breakdown:")
    for g, c in gpu_counts.most_common():
        print(f"  {g}: {c}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build job tracking TSV')
    parser.add_argument('--refresh', action='store_true',
                        help='Query squeue/sacct for live job status')
    args = parser.parse_args()
    build_tracker(refresh=args.refresh)
