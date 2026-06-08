"""Regression guard: vendored slurm tools' project-root constants must
resolve to the actual repo root, not above it. Pre-fix the tools were
at slurm/tools/ in the reference repo and used `script_dir.parent.parent`;
after the move to slurm/ in TRL-Bench the right derivation is
`script_dir.parent`.
"""
from pathlib import Path
import importlib.util
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SLURM_DIR = REPO_ROOT / "slurm"


def _load_tool_module(filename):
    """Load a slurm/<filename>.py as a module via path import.

    Inserts SLURM_DIR onto sys.path so the tool's intra-package imports
    (e.g. `from generate_scripts import ...`) resolve.
    """
    path = SLURM_DIR / filename
    if str(SLURM_DIR) not in sys.path:
        sys.path.insert(0, str(SLURM_DIR))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _project_root_of(mod):
    """Return whatever the module uses as its project-root constant.

    Checks (in order): PROJECT_ROOT, _PROJECT_ROOT, _REPO_ROOT, REPO_ROOT,
    get_project_root().
    """
    for name in ("PROJECT_ROOT", "_PROJECT_ROOT", "_REPO_ROOT", "REPO_ROOT"):
        if hasattr(mod, name):
            return getattr(mod, name)
    if hasattr(mod, "get_project_root"):
        return mod.get_project_root()
    return None


@pytest.mark.parametrize("filename", [
    "submit_all.py",
    "submit_downstream.py",
    "submit_downstream_matrix.py",
    "submit_dlte.py",
    "build_job_tracker.py",
    "check_results.py",
    "generate_scripts.py",
    "generate_downstream_scripts.py",
    "generate_table_embedding_scripts.py",
    "generate_row_scripts.py",
    "generate_row_data_scripts.py",
    "generate_row_data_downstream_scripts.py",
    "validate_embeddings.py",
    "aggregate_results.py",
])
def test_project_root_resolves_to_repo_root(filename):
    mod = _load_tool_module(filename)
    pr = _project_root_of(mod)
    assert pr is not None, f"{filename}: no PROJECT_ROOT-style constant found"
    assert Path(pr) == REPO_ROOT, (
        f"{filename}: project root resolves to {pr}, expected {REPO_ROOT}"
    )
