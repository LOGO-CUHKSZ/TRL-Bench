#!/usr/bin/env bash
# Submit every .sbatch in slurm/jobs/ to slurm.
set -euo pipefail

JOBS_DIR="${1:-slurm/jobs}"

if [ ! -d "$JOBS_DIR" ]; then
    echo "no jobs dir at $JOBS_DIR — run 'python slurm/generate_jobs.py' first" >&2
    exit 1
fi

for f in "$JOBS_DIR"/*.sbatch; do
    sbatch "$f"
done
