#!/bin/bash
# Step 5: Submit column embedding SLURM jobs for DLTE fragments.
#
# Submits 14 jobs: 7 column models × 2 splits (queries + targets)
# Uses the existing SLURM infrastructure.
#
# Usage:
#   bash downstream_tasks/dlte/scripts/step5_submit_column_embeddings.sh          # submit all
#   bash downstream_tasks/dlte/scripts/step5_submit_column_embeddings.sh --dry-run # preview only
#
# Monitor:
#   python slurm/tools/monitor_jobs.py --datasets dlte_v1_queries dlte_v1_targets
#   squeue -u $USER -n emb_bert_dlte_v1_queries,emb_tapas_dlte_v1_targets,...
#
# Validate outputs:
#   python downstream_tasks/dlte/validate/validate_step5.py

set -euo pipefail
cd "$(dirname "$0")/../../.."  # project root

echo "Step 5: Column Embeddings for DLTE"
echo "==================================="
echo ""

# Generate scripts (idempotent)
echo "Generating SLURM scripts..."
python slurm/tools/generate_scripts.py --datasets dlte_v1_queries dlte_v1_targets

echo ""
echo "Submitting jobs..."

# Submit via the existing submit_all.py
python slurm/tools/submit_all.py \
    --datasets dlte_v1_queries dlte_v1_targets \
    "$@"

echo ""
echo "Expected outputs (per model):"
echo "  embeddings/column/{model}/dlte_v1_queries.pkl  (5,516 tables)"
echo "  embeddings/column/{model}/dlte_v1_targets.pkl  (11,032 tables)"
echo ""
echo "Models: bert, gte, starmie, tabbie, tabert, tabsketchfm, tapas, turl"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER | grep dlte"
echo "  python slurm/tools/monitor_jobs.py --datasets dlte_v1_queries dlte_v1_targets"
