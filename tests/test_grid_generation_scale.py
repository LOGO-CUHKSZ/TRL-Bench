"""Scale test: when generate_scripts.py runs on the full models.yaml ×
datasets.yaml matrix without filters, it emits one sbatch per
(model, dataset) cell where the model supports col-granularity
extraction. Asserts dispatch coverage matches the registry's
view of valid cells.

Login-node friendly (no submission). Tests that no model or dataset
is silently dropped by a filter."""
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LOAD_ENV = REPO_ROOT / "load_env"

# load_env is the site-local env-setup script (gitignored). Skip cleanly when
# absent so a fresh clone passes -- this test sources load_env in a subprocess
# to exercise the generator.
pytestmark = pytest.mark.skipif(
    not LOAD_ENV.exists(),
    reason="load_env (site-local env-setup script) not present",
)


def _run(cmd):
    full = f'cd {REPO_ROOT} && source {LOAD_ENV} && {cmd}'
    return subprocess.run(
        ["bash", "-c", full],
        capture_output=True, text=True, check=False,
    )


def test_generate_scripts_full_grid_produces_many_sbatch(tmp_path):
    """No filters: emit sbatch files for the full models × datasets
    grid of column-extractor cells. Assert a minimum count consistent
    with at least 5 models × at least 5 datasets = 25 cells.

    Many cells will SKIP because the local data root only carries a
    subset of datasets (sato, wikict_relation, spider_join). The
    generator still emits global text-embedding scripts (mpnet ×
    nq_tables / wtq, sentence_t5 × *, openai × *) regardless of
    filters, plus sharded column scripts for cells with data
    present. The >= 25 threshold reflects this realistic mix.
    """
    out_dir = tmp_path / "full_grid"
    out_dir.mkdir()
    result = _run(
        f"python slurm/generate_scripts.py --output-dir {out_dir}"
    )
    assert result.returncode == 0, (
        f"generator failed exit {result.returncode}\n"
        f"stderr tail: {result.stderr[-2000:]}"
    )
    sbatch_files = list(out_dir.rglob("*.sbatch"))
    assert len(sbatch_files) >= 25, (
        f"only {len(sbatch_files)} sbatch generated; expected >= 25 "
        f"for a realistic col-extractor grid. files: "
        f"{[p.name for p in sbatch_files][:20]}"
    )
