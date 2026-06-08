#!/usr/bin/env bash
# Layer-2 release verification: fresh-extraction smoke matrix.
#
# Submits one Slurm job per cell across (model family x task family). Each
# cell runs the full user-derived pipeline (Stage-0 -> Stage-1 -> Stage-2 ->
# Stage-3 -> envelope) from a clean state. Pass criterion is "envelope JSON
# produced + metrics in a sane range" (not a byte-match).
#
# Usage:
#   scripts/smoke_matrix.sh                       # submit all cells
#   scripts/smoke_matrix.sh anchor tabicl tabbie  # submit named cells only
#   scripts/smoke_matrix.sh --summary <log_dir>   # summarize a prior run
#
# Configuration (env vars, no site-specific defaults in release tree):
#   TRLB_SMOKE_ACCOUNT     required  Slurm account (e.g. def-pi).
#   TRLB_SMOKE_PARTITION   required  Slurm partition for GPU jobs.
#   TRLB_SMOKE_CKPT_ROOT   optional  Path to licensed checkpoints
#                                    (default: ./checkpoints).
#   TRLB_SMOKE_ENV_SCRIPT  optional  Site-local env loader to source before
#                                    `python -m trl_bench.run` (default:
#                                    ./load_env if present; else no-op).
#   TRLB_SMOKE_SECRETS_ENV optional  Site-local file exporting auth env vars
#                                    (OPENAI_API_KEY, TABPFN_TOKEN). Sourced
#                                    inside each cell's sbatch script so the
#                                    secrets land on the compute node. Slurm
#                                    does not always propagate the submitter
#                                    shell's env (Killarney does not), so
#                                    relying on `--export=ALL` is brittle.
#
# Outputs:
#   results_smoke/<cell>/<...>.json          per-cell envelope JSON
#   results_smoke/_matrix_<ts>/<cell>.log    per-cell stdout/stderr
#   results_smoke/_matrix_<ts>/SUMMARY.txt   final pass/fail summary

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Cell definitions ------------------------------------------------------
# Each row: NAME|MODEL|TASK|DATASET|SETTING|PROBE|TIME|NOTE
# (For row_prediction, SETTING is the LABEL COLUMN name, not a granularity.)
CELLS=(
    "anchor|bert|join_classification|spider_join|cls_embedding|linear|01:00:00|canonical col/table anchor"
    "tabicl|tabicl|row_prediction|openml_3|class|mlp|00:30:00|pure-row (reference uses mlp)"
    "tabbie|tabbie|join_classification|spider_join|cls_embedding|linear|01:30:00|licensed-checkpoint col/table"
    "tapas|tapas|join_classification|spider_join|cls_embedding|linear|06:00:00|HF auto-fetch col/table"
    "mpnet|mpnet|table_retrieval|nq_tables|cls_embedding|linear|06:00:00|table-direct + query encoder"
    "clust|bert|column_clustering|sato|cls_embedding|linear|06:00:00|deterministic (no probe head)"
    "gte|gte|join_classification|spider_join|cls_embedding|linear|01:30:00|GTE col/table (text encoder)"
    "turl|turl|join_classification|spider_join|column_mean|linear|01:30:00|TURL licensed-checkpoint (column_mean only, no CLS)"
    "tabert|tabert|join_classification|spider_join|column_mean|linear|01:30:00|TaBERT licensed-checkpoint (fairseq)"
    "tabpfn|tabpfn|row_prediction|openml_3|class|mlp|00:30:00|TabPFN pretrained-row (needs TABPFN_TOKEN)"
    "record|bert|record_linkage|wdc_products_small|row_embedding|linear|00:30:00|record_linkage e2e"
    "openai|openai|join_classification|spider_join|cls_embedding|linear|02:00:00|OpenAI API ablation (needs OPENAI_API_KEY)"
    "starmie|starmie|join_classification|spider_join|column_mean|linear|01:30:00|Starmie contrastive col (column_mean only)"
    "tabsketchfm|tabsketchfm|join_classification|spider_join|cls_embedding|linear|01:30:00|TabSketchFM col/table"
    "tuta|tuta|join_classification|spider_join|cls_embedding|linear|01:30:00|TUTA tree-transformer col"
    "sentence_t5|sentence_t5|table_retrieval|nq_tables|cls_embedding|linear|06:00:00|Sentence-T5 query encoder (mirror mpnet)"
    "tapex|tapex|join_classification|spider_join|cls_embedding|linear|02:00:00|TAPEX table-direct (BART encoder; not a query encoder, so join not retrieval)"
)

step() { printf '\n==[ %s ]==\n' "$*"; }
ok()   { printf '  OK: %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*"; exit 1; }

# ---- Summary mode ----------------------------------------------------------
if [ "${1:-}" = "--summary" ]; then
    LOG_DIR="${2:-}"
    [ -d "$LOG_DIR" ] || fail "--summary expects a log dir (results_smoke/_matrix_<ts>)"
    SUMMARY="$LOG_DIR/SUMMARY.txt"
    : > "$SUMMARY"
    step "smoke-matrix summary: $LOG_DIR"
    for log in "$LOG_DIR"/*.log; do
        [ -e "$log" ] || continue
        NAME=$(basename "$log" .log)
        ENV=$(grep -oE 'results_smoke/[^ ]+\.json' "$log" 2>/dev/null | tail -1 || true)
        if [ -n "$ENV" ] && [ -f "$ENV" ]; then
            printf '  PASS  %-10s  envelope: %s\n' "$NAME" "$ENV" | tee -a "$SUMMARY"
        elif grep -qE "CANCELLED|TIME LIMIT" "$log"; then
            printf '  KILL  %-10s  (slurm killed -- time limit?)\n' "$NAME" | tee -a "$SUMMARY"
        else
            TAIL=$(tail -3 "$log" | tr '\n' ' ' | head -c 120)
            printf '  FAIL  %-10s  %s\n' "$NAME" "$TAIL" | tee -a "$SUMMARY"
        fi
    done
    printf '\nfull summary written to %s\n' "$SUMMARY"
    exit 0
fi

# ---- Submission mode -------------------------------------------------------
[ -n "${TRLB_SMOKE_ACCOUNT:-}" ]  || fail "TRLB_SMOKE_ACCOUNT not set (e.g. export TRLB_SMOKE_ACCOUNT=def-pi)"
[ -n "${TRLB_SMOKE_PARTITION:-}" ] || fail "TRLB_SMOKE_PARTITION not set (e.g. export TRLB_SMOKE_PARTITION=gpubase_l40s_b2)"
CKPT_ROOT="${TRLB_SMOKE_CKPT_ROOT:-$REPO_ROOT/checkpoints}"
ENV_SCRIPT="${TRLB_SMOKE_ENV_SCRIPT:-$REPO_ROOT/load_env}"
SECRETS_ENV="${TRLB_SMOKE_SECRETS_ENV:-}"

# Filter cells by name if args provided
SELECT_NAMES="$*"
SELECTED=()
for cell in "${CELLS[@]}"; do
    NAME="${cell%%|*}"
    if [ -z "$SELECT_NAMES" ] || echo " $SELECT_NAMES " | grep -q " $NAME "; then
        SELECTED+=("$cell")
    fi
done
[ ${#SELECTED[@]} -gt 0 ] || fail "no cells matched (available: $(printf '%s ' "${CELLS[@]%%|*}"))"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/results_smoke/_matrix_$TS"
mkdir -p "$LOG_DIR"

step "Layer-2 smoke matrix: submitting ${#SELECTED[@]} cell(s)"
printf '  account=%s  partition=%s  ckpt_root=%s\n' \
    "$TRLB_SMOKE_ACCOUNT" "$TRLB_SMOKE_PARTITION" "$CKPT_ROOT"
printf '  env-script: %s\n' "$ENV_SCRIPT"
printf '  logs:       %s\n' "$LOG_DIR"

for cell in "${SELECTED[@]}"; do
    IFS='|' read -r NAME MODEL TASK DATASET SETTING PROBE TIME NOTE <<< "$cell"
    JOB_LOG="$LOG_DIR/${NAME}.log"

    # Write a per-cell bash script (sbatch --wrap runs via /bin/sh by default,
    # which can't parse the CC lmod profile's bash-only [[ ]] / (( )) syntax.
    # Using a real script with #!/bin/bash shebang gives full bash semantics.
    SCRIPT="$LOG_DIR/${NAME}.sbatch.sh"
    cat > "$SCRIPT" <<SBATCH_EOF
#!/bin/bash
set -e
# CC lmod profile (bash-only). No-op on non-CC hosts.
[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ] && \\
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
cd '$REPO_ROOT'
[ -f '$ENV_SCRIPT' ] && source '$ENV_SCRIPT' >/dev/null 2>&1 || true
if [ -n '$SECRETS_ENV' ] && [ -f '$SECRETS_ENV' ]; then set -a; source '$SECRETS_ENV' >/dev/null 2>&1; set +a; fi  # set -a: export keys so child python inherits them even if the file omits 'export'
export TRL_BENCH_CKPT_ROOT='$CKPT_ROOT'
rm -rf 'results_smoke/$NAME' 'embeddings_smoke/$NAME'
echo "=== cell: $NAME ($MODEL x $TASK x $DATASET x $SETTING x $PROBE) ==="
python -m trl_bench.run \\
    --model '$MODEL' --task '$TASK' --dataset '$DATASET' \\
    --setting '$SETTING' --probe '$PROBE' --seed 42 \\
    --results-dir 'results_smoke/$NAME' \\
    --embeddings-dir 'embeddings_smoke/$NAME' \\
    --data-root data 2>&1
ENV_JSON=\$(find 'results_smoke/$NAME' -name '*.json' | head -1)
echo "=== envelope: \$ENV_JSON ==="
SBATCH_EOF
    chmod +x "$SCRIPT"

    SUBMIT_OUT=$(sbatch \
        --account="$TRLB_SMOKE_ACCOUNT" \
        --partition="$TRLB_SMOKE_PARTITION" \
        --gres=gpu:1 --cpus-per-task=8 --mem=48G --time="$TIME" \
        --job-name="smk_$NAME" --output="$JOB_LOG" \
        "$SCRIPT")
    JOB_ID=$(echo "$SUBMIT_OUT" | grep -oE '[0-9]+' | head -1)
    printf '  submitted  %-10s  job=%s  time=%s  -- %s\n' \
        "$NAME" "$JOB_ID" "$TIME" "$NOTE"
done

printf '\nAll jobs submitted. Check progress with:\n'
printf '  squeue -u \$USER -n smk_anchor,smk_tabicl,smk_tabbie,smk_tapas,smk_mpnet,smk_clust\n'
printf '\nOnce all jobs finish, summarize with:\n'
printf '  scripts/smoke_matrix.sh --summary %s\n' "$LOG_DIR"
