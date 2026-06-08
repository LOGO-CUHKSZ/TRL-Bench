#!/usr/bin/env bash
#
# Finalize the TRL-Bench release: commit the pending edit, run the full
# test suite, document author-placeholder status, then push to GitHub.
#
# This script is IDEMPOTENT — re-running skips work that's already done.
#
# Usage:
#   bash scripts/finalize_release.sh         # commit + test + check, then prompt
#   bash scripts/finalize_release.sh commit  # just commit pending edit
#   bash scripts/finalize_release.sh test    # just run tests
#   bash scripts/finalize_release.sh check   # just check placeholders
#   bash scripts/finalize_release.sh push    # configure remote + push

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="$HOME/anaconda3/etc/profile.d/conda.sh"
ENV="${TRL_BENCH_TEST_ENV:-venv}"
STEP="${1:-all}"

log() { echo "[finalize] $*"; }

do_commit() {
    log "=== STEP 1: commit pending registry cleanup ==="
    cd "$REPO"
    if git diff --quiet src/trl_bench/registry.py; then
        log "no pending edit in registry.py — already committed or never edited."
        return 0
    fi
    log "registry.py has uncommitted changes; staging + committing."
    git add src/trl_bench/registry.py
    git commit -m "fix(registry): remove duplicate record_linkage + stub ProbeConfig footguns

The previous _TASK_PROBE_CONFIG had:
  - A first 'record_linkage' entry with the default runner (run_task.py)
    on lines 220-228 that was silently overridden by a second 'record_linkage'
    entry on lines 247-262 with the correct task-specific runner. Removed
    the broken first entry.
  - Stub ProbeConfig entries for column_type_prediction, column_relation_prediction,
    and row_prediction that would silently dispatch with the default
    run_task.py runner — wrong for all three (each uses a distinct downstream
    script with a flat-key envelope schema). Removed; replaced with an
    explicit NOTE comment explaining they will be wired with their own
    ProbeConfig once their canonical .sbatch + train script are ported.

Now any attempt to dispatch these unwired tasks via 'trl-bench-run' raises a
clear SettingError pointing at docs/USAGE.md instead of silently
running the wrong subprocess.

Behaviour unchanged for the 6 verified probe-task families
(join_classification, join_containment, union_classification,
union_regression, table_subset, record_linkage)."
    log "committed."
}

do_test() {
    log "=== STEP 2: run test suite ==="
    cd "$REPO"
    # shellcheck disable=SC1090
    source "$CONDA"
    conda activate "$ENV"
    PYTHONPATH=src python -m pytest tests/ -q --ignore=tests/smoke_test.py
    log "tests passed."
}

do_check_placeholders() {
    log "=== STEP 3a: confirm author-placeholder status ==="
    cd "$REPO"
    local hits
    hits=$(grep -nE '^\s*#\s*TODO|^\s*<!-- TODO|TODO: Fill' pyproject.toml CITATION.cff README.md 2>/dev/null || true)
    if [[ -n "$hits" ]]; then
        log "WARNING: TODO placeholder(s) still present:"
        echo "$hits"
        log ""
        log "Fill these before public push:"
        log "  - pyproject.toml line 13 (authors array)"
        log "  - CITATION.cff (top of file: title, authors, DOI, etc.)"
        log "  - README.md (BibTeX block)"
        log ""
        log "If pushing to a private repo (fill later), use: bash \$0 push"
    else
        log "no TODO placeholders found — ready for public push."
    fi
}

do_push() {
    log "=== STEP 3b: push to GitHub ==="
    cd "$REPO"
    if ! git remote get-url origin >/dev/null 2>&1; then
        log "no 'origin' remote configured."
        read -r -p "[finalize] enter GitHub URL (e.g. git@github.com:logo-lab/trl-bench.git) or 'skip': " URL
        if [[ "$URL" == "skip" ]] || [[ -z "$URL" ]]; then
            log "skipping push."
            return 0
        fi
        git remote add origin "$URL"
    fi
    log "pushing master to origin..."
    git push -u origin master
    log "push complete: $(git remote get-url origin)"
}

case "$STEP" in
    commit) do_commit ;;
    test)   do_test ;;
    check)  do_check_placeholders ;;
    push)   do_push ;;
    all)
        do_commit
        do_test
        do_check_placeholders
        echo ""
        log "to push (after filling placeholders): bash $0 push"
        ;;
    *)
        echo "usage: $0 [commit|test|check|push|all]" >&2
        exit 2
        ;;
esac
