"""Smoke tests for slurm/generate_scripts.py and
slurm/generate_downstream_scripts.py — assert each generator emits a
syntactically reasonable sbatch file for a known cell. Login-node
friendly (no slurm submission, no GPU)."""
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LOAD_ENV = REPO_ROOT / "load_env"

# load_env is the site-local env-setup script (gitignored). Skip cleanly when
# absent so a fresh clone passes -- these tests source load_env in a subprocess
# to exercise the generators.
pytestmark = pytest.mark.skipif(
    not LOAD_ENV.exists(),
    reason="load_env (site-local env-setup script) not present",
)


def _run(cmd):
    """Run a command from REPO_ROOT under the load_env environment."""
    full = f'cd {REPO_ROOT} && source {LOAD_ENV} && {cmd}'
    return subprocess.run(
        ["bash", "-c", full],
        capture_output=True, text=True, check=False,
    )


def test_generate_scripts_emits_bert_spider_join_sbatch(tmp_path):
    """generate_scripts.py for (bert, spider_join) writes one sbatch.

    Asserts the emitted sbatch contains:
      - account directive (#SBATCH --account=...)
      - the load_env source line
      - the bert column-extractor python invocation
        (file path models/bert/generate_column_embeddings.py, NOT the
        unsupported module path trl_bench.models.bert.*)
    """
    out_dir = tmp_path / "scripts"
    out_dir.mkdir()
    cmd = (
        f"python slurm/generate_scripts.py "
        f"--models bert --datasets spider_join "
        f"--output-dir {out_dir}"
    )
    result = _run(cmd)
    assert result.returncode == 0, (
        f"generate_scripts.py exited {result.returncode}\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}"
    )
    # Filter to bert_spider_join-specific sbatch (the run also emits
    # global text-embedding sbatch files into the same output dir).
    bert_sbatch_files = list(out_dir.rglob("bert_spider_join*.sbatch"))
    assert len(bert_sbatch_files) >= 1, (
        f"no bert_spider_join sbatch emitted in {out_dir}. "
        f"listed: {[p.name for p in out_dir.rglob('*.sbatch')]}"
    )
    # spider_join is sharded: the generator emits several matching sbatch
    # files (per-shard extractors + a merge script that has no extractor
    # invocation), and rglob order is not stable. Assert against the union
    # of all matched files rather than a non-deterministic [0].
    content = "\n".join(p.read_text() for p in bert_sbatch_files)
    # account directive
    assert "--account=" in content, "missing --account directive"
    # Source env
    assert "source" in content and "load_env" in content
    # bert column extractor invocation (file-path form, not module path)
    assert "models/bert/generate_column_embeddings.py" in content, (
        "missing bert column-extractor script path"
    )


def test_generate_downstream_scripts_emits_bert_join_classification_cell(tmp_path):
    """generate_downstream_scripts.py for (bert, join_classification,
    spider_join, linear probe, seed=42) writes one sbatch.

    Uses --embeddings-dir embeddings (TRL-Bench layout). --head-type linear
    replaces the plan's mention of `--probes linear` (the actual CLI flag).
    """
    out_dir = tmp_path / "downstream"
    out_dir.mkdir()
    cmd = (
        f"python slurm/generate_downstream_scripts.py "
        f"--models bert --tasks join_classification "
        f"--datasets spider_join "
        f"--head-type linear --seeds 42 "
        f"--embeddings-dir embeddings "
        f"--output-dir {out_dir}"
    )
    result = _run(cmd)
    assert result.returncode == 0, (
        f"generate_downstream_scripts.py exited {result.returncode}\n"
        f"stdout: {result.stdout[-1500:]}\n"
        f"stderr: {result.stderr[-1500:]}"
    )
    sbatch_files = list(out_dir.rglob("*.sbatch"))
    assert len(sbatch_files) >= 1, (
        f"no sbatch emitted in {out_dir}. stdout tail: "
        f"{result.stdout[-1500:]}"
    )
    content = "\n".join(p.read_text() for p in sbatch_files)
    assert "--account=" in content
    assert "source" in content and "load_env" in content
    # downstream invokes run_task.py (canonical pair-task runner)
    assert "run_task" in content
    assert "spider_join" in content
    assert "seed" in content.lower() and "42" in content
