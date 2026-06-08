#!/bin/bash
# Step 6: Submit row embedding SLURM jobs for DLTE fragments.
#
# Submits 30 jobs: 15 row models × 2 splits (queries + targets)
#   Models: BERT, DAE, GTE, SAINT, SCARF, SubTab, TABBIE,
#           TabICL, TabPFN, TabTransformer, TabularBinning, TransTab, TuTa, VIME
#
# Usage:
#   bash downstream_tasks/dlte/scripts/step6_submit_row_embeddings.sh          # submit all
#   bash downstream_tasks/dlte/scripts/step6_submit_row_embeddings.sh --dry-run # preview only
#
# Monitor:
#   squeue -u $USER | grep row_
#
# Validate outputs:
#   python downstream_tasks/dlte/validate/validate_step6.py

set -euo pipefail
cd "$(dirname "$0")/../../.."  # project root

echo "Step 6: Row Embeddings for DLTE"
echo "==============================="
echo ""

# Generate scripts (idempotent)
echo "Generating SLURM scripts..."
python slurm/tools/generate_row_scripts.py \
    --models bert dae gte saint scarf subtab tabbie tabicl tabpfn tabtransformer tabular_binning transtab tuta vime \
    --datasets dlte_v1_queries dlte_v1_targets

echo ""
echo "Submitting jobs..."

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
fi

SCRIPTS_DIR="slurm/scripts/generated/row_embeddings"
MODELS=(bert dae gte saint scarf subtab tabbie tabicl tabpfn tabtransformer tabular_binning transtab tuta vime)
DATASETS=(dlte_v1_queries dlte_v1_targets)

submitted=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        script="${SCRIPTS_DIR}/${model}_${dataset}.sbatch"
        if [[ ! -f "$script" ]]; then
            echo "  SKIP: $script not found"
            continue
        fi
        if [[ -n "$DRY_RUN" ]]; then
            echo "  [DRY-RUN] Would submit: $script"
        else
            job_id=$(sbatch "$script" | grep -oP '\d+')
            echo "  Submitted ${model}/${dataset}: Job $job_id"
            submitted=$((submitted + 1))
        fi
    done
done

echo ""
echo "Submitted $submitted jobs"
echo ""
echo "Expected outputs (per model):"
echo "  embeddings/row/{model}/dlte_v1_queries.pkl  (5,516 tables)"
echo "  embeddings/row/{model}/dlte_v1_targets.pkl  (11,032 tables)"
echo ""
echo "Models: bert, dae, gte, saint, scarf, subtab, tabbie, tabicl, tabpfn, tabtransformer, tabular_binning, transtab, tuta, vime"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER | grep row_"
