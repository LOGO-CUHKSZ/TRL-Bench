#!/bin/bash
# DEPRECATED: Use the SLURM pipeline instead:
#   python slurm/tools/generate_downstream_scripts.py --tasks union_search join_search
#   python slurm/tools/submit_downstream.py --models value_overlap
#
# Run all value-overlap baselines for join_search and union_search.
#
# Usage (local):
#   bash utils/baselines/value_overlap/run_all.sh
#
# Usage (SLURM):
#   sbatch --job-name=value_overlap --time=06:00:00 --mem=32G --cpus-per-task=4 \
#          --partition=cpubase --output=slurm/logs/value_overlap_%j.out \
#          utils/baselines/value_overlap/run_all.sh

set -euo pipefail

# Resolve project root — handle both local and SLURM execution
# SLURM copies scripts to a spool dir, so dirname($0) doesn't work.
# Use SLURM_SUBMIT_DIR if available, otherwise resolve from script path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Value-Overlap Baselines"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"

# Setup environment
source load_env 2>/dev/null || true
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo ""
echo "[1/2] Union Search (Santos + TUS + UGEN-V1 + UGEN-V2)"
echo "------------------------------------------"
python -m utils.baselines.value_overlap.union_search \
    --datasets santos tus ugen_v1 ugen_v2 \
    --K 10 --threshold 0.0

echo ""
echo "[2/2] Join Search (all OpenData variants)"
echo "------------------------------------------"
python -m utils.baselines.value_overlap.join_search \
    --datasets opendata_main opendata_can opendata_usa opendata_uk_sg \
    --k 50 --k_values 10 20 50

echo ""
echo "=========================================="
echo "All value-overlap baselines complete"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  results/evaluation/union_search/value_overlap/"
echo "  results/evaluation/join_search/value_overlap/"
