#!/usr/bin/env bash
# Release verification: simulate a fresh clone + the documented install +
# the test suite in a clean environment (no site-local setup). Catches the
# class of "works on my machine" bugs -- broken install command, missing
# extras, tests hard-failing without site-local scripts, collection errors,
# etc. -- in ~3-5 minutes.
#
# Usage:
#   scripts/verify_release.sh                    # verify current HEAD
#   scripts/verify_release.sh origin/master      # verify the pushed branch
#   ALL_EXTRAS=1 scripts/verify_release.sh       # also walk every declared extra
#   KEEP_SCRATCH=1 scripts/verify_release.sh     # keep the scratch dir on success
#
# Requires Python >=3.9 on $PATH + network for `pip install`.
# Exit 0 = ready. Non-zero = a gap to fix; logs are kept in the scratch
# dir for inspection.
#
# Compute Canada (Killarney / Beluga / Cedar / ...) note: their pyarrow
# wheel is a "noinstall" stub and their wheelhouse expects a specific
# install dance (`module load arrow` + a CC-configured venv). The script
# tries to accommodate by using `--system-site-packages` when it detects
# CC ($EBROOTARROW set), but the cleanest verification is on a stock
# Linux host with vanilla PyPI access (or in a Docker/Apptainer image).
# On CC, this script is best-effort; a stock Linux host gives the cleanest run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_REF="${1:-HEAD}"
SCRATCH="$(mktemp -d -t trlb_verify.XXXXXX)"
VERIFY_FAILED=0

cleanup() {
    if [ "${KEEP_SCRATCH:-0}" = "0" ] && [ "$VERIFY_FAILED" = "0" ]; then
        rm -rf "$SCRATCH"
    else
        printf '\n  scratch retained at %s\n' "$SCRATCH"
    fi
}
trap cleanup EXIT

step() { printf '\n==[ %s ]==\n' "$*"; }
ok()   { printf '  OK: %s\n' "$*"; }
fail() {
    printf '  FAIL: %s\n' "$*"
    VERIFY_FAILED=1
    # Dump any captured logs to stdout so the actual error survives when this
    # script runs under srun (compute-node /tmp is ephemeral after the job).
    for log in pytest.log pytest_slow.log collect.log pip.log; do
        [ -s "$SCRATCH/$log" ] && {
            printf '\n--- %s tail ---\n' "$log"
            tail -25 "$SCRATCH/$log"
        }
    done
    exit 1
}

step "1. clone $SOURCE_REF into scratch (simulates fresh, no-shared-state clone)"
git clone --quiet --no-local "$REPO_ROOT" "$SCRATCH/trl-bench"
( cd "$SCRATCH/trl-bench" && git checkout --quiet "$SOURCE_REF" )
cd "$SCRATCH/trl-bench"
ok "fresh clone at $SCRATCH/trl-bench (HEAD: $(git rev-parse --short HEAD))"

step "2. no maintainer-private artefacts in tracked tree"
for p in load_env venv data embeddings checkpoints; do
    [ ! -e "$p" ] || fail "$p present in fresh clone (should be gitignored)"
done
ok "load_env / venv / data / embeddings / checkpoints absent"

step "3a. fresh venv + the README install command (verbatim)"
# On Compute Canada hosts (/cvmfs/soft.computecanada.ca), pyarrow ships as a
# noinstall stub that refuses to build inside a fresh venv: their `arrow`
# module is a runtime prereq, not a pip-installable wheel. Use
# --system-site-packages so the venv inherits the module-provided pyarrow.
# (Slightly weakens fresh-venv isolation; this is CC's documented pattern.)
if [ -d /cvmfs/soft.computecanada.ca ] && [ -n "${EBROOTARROW:-}" ]; then
    python3 -m venv --system-site-packages "$SCRATCH/venv"
    printf '  note: Compute Canada host detected, venv created with --system-site-packages (pyarrow via `module load arrow`)\n'
else
    python3 -m venv "$SCRATCH/venv"
fi
# shellcheck source=/dev/null
. "$SCRATCH/venv/bin/activate"
pip install --quiet --upgrade pip setuptools wheel >/dev/null
INSTALL_LOG="$SCRATCH/pip.log"
if pip install -e ".[bert]" 2>&1 | tee "$INSTALL_LOG" >/dev/null; then
    if grep -qiE "no such extra|no matching distribution|^ERROR:" "$INSTALL_LOG"; then
        fail "pip install reported errors (log: $INSTALL_LOG)"
    fi
    ok 'pip install -e ".[bert]" succeeded (README install command)'
else
    fail "pip install -e \".[bert]\" exited non-zero (log: $INSTALL_LOG)"
fi

step "3b. add [dev] for the test runner (pytest is in the dev extra, not base)"
pip install --quiet -e ".[dev]" 2>>"$INSTALL_LOG" \
    || fail "pip install -e \".[dev]\" failed (log: $INSTALL_LOG)"
ok 'pip install -e ".[dev]" added (test runner installed)'

step "4. core + bert wrapper imports"
python -c "
import trl_bench.registry as r, trl_bench.run, trl_bench.data.stage
import trl_bench.models.bert
n = len(list(r.list_cells()))
assert n >= 50, f'list_cells() returned {n}, expected >=50'
print(f'imports OK; {n} valid (model, task) cells in the registry')
" || fail "core imports broken"
ok "core + bert imports succeed"

step "5. \`python -m trl_bench.run --help\` works (entry-point sanity)"
python -m trl_bench.run --help >/dev/null 2>&1 || fail "trl_bench.run --help failed"
ok "trl_bench.run entry point usable"

step "6. tests/ -m 'not slow' passes in a clean environment"
PYTEST_LOG="$SCRATCH/pytest.log"
python -m pytest tests/ -q -m "not slow" 2>&1 | tee "$PYTEST_LOG" >/dev/null || true
if tail -5 "$PYTEST_LOG" | grep -qE "[0-9]+ passed" \
   && ! tail -5 "$PYTEST_LOG" | grep -qE "failed|error"; then
    PASSED=$(tail -5 "$PYTEST_LOG" | grep -oE "[0-9]+ passed" | head -1)
    ok "tests/ non-slow: $PASSED"
else
    fail "tests/ non-slow had failures (log: $PYTEST_LOG; tail: $(tail -3 "$PYTEST_LOG" | tr '\n' ' '))"
fi

step "7. tests/ -m 'slow' skips cleanly (no hard fail)"
SLOW_LOG="$SCRATCH/pytest_slow.log"
python -m pytest tests/ -q -m "slow" 2>&1 | tee "$SLOW_LOG" >/dev/null || true
if grep -qE "skipped" "$SLOW_LOG" && ! grep -qE "failed|error" "$SLOW_LOG"; then
    SKIPPED=$(grep -oE "[0-9]+ skipped" "$SLOW_LOG" | head -1)
    ok "@slow: $SKIPPED (graceful skip)"
else
    fail "@slow tests didn't skip cleanly (log: $SLOW_LOG)"
fi

step "8. bare \`pytest\` collects cleanly (testpaths gate on vendored upstream tests)"
COLL_LOG="$SCRATCH/collect.log"
python -m pytest --collect-only -q 2>&1 | tee "$COLL_LOG" >/dev/null || true
if grep -qE "tests collected" "$COLL_LOG" && ! grep -qE "errors during collection" "$COLL_LOG"; then
    COLLECTED=$(grep -oE "[0-9]+ tests collected" "$COLL_LOG" | head -1)
    ok "bare pytest: $COLLECTED (testpaths=[tests] effective)"
else
    fail "bare pytest collection failed (log: $COLL_LOG)"
fi

if [ "${ALL_EXTRAS:-0}" = "1" ]; then
    step "9. (ALL_EXTRAS=1) every declared model extra installs cleanly"
    EXTRAS=$(sed -n '/\[project.optional-dependencies\]/,/^\[/p' pyproject.toml \
             | sed -nE 's/^([a-z_][a-z_0-9]*)[[:space:]]*=.*/\1/p' \
             | grep -vE '^(all|dev)$')
    for E in $EXTRAS; do
        if pip install --quiet -e ".[$E]" 2>&1 | grep -iqE "error"; then
            fail "extra [$E] failed to install (some extras need CUDA-capable host)"
        fi
        ok "extra [$E] installs cleanly"
    done
fi

printf '\n==[ release-verify PASSED for %s ]==\n' "$SOURCE_REF"
printf '  Next layer: scripts/smoke_matrix.sh (fresh-extraction smoke matrix; ~30 min on GPU)\n'
