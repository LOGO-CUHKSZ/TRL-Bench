"""Regression guard: vendored slurm tools should reference TRL-Bench's
documented public paths (results/evaluation/, embeddings/), not the
legacy ``assets/*`` data paths.

Allow-list: some substrings are intentional (e.g., a `--<flag>` whose
help text explains an overlay example pointing at a legacy ``assets/*``
path to preserve back-compat for users with legacy paths). When that's
the case, the test must explicitly recognize it. Default: zero hits.
"""
from pathlib import Path
import re

SLURM_DIR = Path(__file__).resolve().parent.parent / "slurm"


def _file_text(name):
    return (SLURM_DIR / name).read_text()


# Files that MUST have no assets/* substrings post-patch:
HARD_NO_ASSETS = [
    "aggregate_results.py",
    "check_results.py",
    "validate_embeddings.py",
    "generate_scripts.py",
    "generate_downstream_scripts.py",
    "generate_row_scripts.py",
    "generate_row_data_scripts.py",
    "generate_row_data_downstream_scripts.py",
    "generate_table_embedding_scripts.py",
    "submit_all.py",
    "submit_downstream.py",
    "submit_downstream_matrix.py",
]


import pytest

@pytest.mark.parametrize("filename", HARD_NO_ASSETS)
def test_no_legacy_assets_path(filename):
    text = _file_text(filename)
    legacy_hits = [
        m.group(0) for m in re.finditer(
            r"assets/(evaluation_results|embeddings)\b", text
        )
    ]
    assert not legacy_hits, (
        f"{filename}: still references legacy paths: "
        f"{sorted(set(legacy_hits))}"
    )


@pytest.mark.parametrize("filename,expected_substring", [
    ("aggregate_results.py", "results/evaluation"),
    ("check_results.py", "results/evaluation"),
    ("validate_embeddings.py", "embeddings/"),
    ("generate_scripts.py", "embeddings/"),
])
def test_uses_trl_bench_path(filename, expected_substring):
    text = _file_text(filename)
    assert expected_substring in text, (
        f"{filename}: expected to reference {expected_substring}"
    )


# == datasets/ regression guard ==============================================
# `datasets/` was the reference repo's data root; TRL-Bench uses `data/`.
# Sweeping `datasets/` substrings is independent of the `assets/` sweep and
# needs its own regression test — d7ba84d's sweep missed lines in
# slurm/scripts/templates/downstream/column_type_prediction.sbatch.template
# (preflight CSV-path checks at lines 207-208), found in the final code
# review. This test catches future regressions across both Python sources
# and sbatch templates.
_SLURM_FILES_NO_DATASETS_PREFIX = [
    "config/downstream/task_datasets.yaml",
    "config/downstream/tasks.yaml",
    "config/downstream/baselines.yaml",
    "config/datasets.yaml",
    "config/models.yaml",
    "scripts/templates/downstream/column_type_prediction.sbatch.template",
    "scripts/templates/downstream/column_relation_prediction.sbatch.template",
]


@pytest.mark.parametrize("relpath", _SLURM_FILES_NO_DATASETS_PREFIX)
def test_no_legacy_datasets_path(relpath):
    """Slurm config + template files should not reference the legacy
    `datasets/<dataset>/...` data root. TRL-Bench's data root is `data/`."""
    path = SLURM_DIR / relpath
    if not path.exists():
        pytest.skip(f"{relpath} not present (defer to follow-up if relevant)")
    text = path.read_text()
    legacy_hits = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # comments are documentation, not load-bearing paths
        for m in re.findall(r"datasets/[a-zA-Z_][\w-]*", line):
            # Allow DLTE's intentional nested layout: the stager produces
            # data/dlte_v1/datasets/dlte_v1/, so tables_source values like
            # "dlte_v1/datasets/dlte_v1" have `datasets/` as a SUBDIR, not the
            # root. Only the root-level `datasets/` is the legacy-layout bug.
            if "dlte_v1/datasets/" in line:
                continue
            legacy_hits.append(m)
    assert not legacy_hits, (
        f"{relpath}: still references legacy `datasets/...` prefix: "
        f"{sorted(set(legacy_hits))}"
    )


# Python generators build dataset dirs via `project_root / 'datasets' / name`
# — the literal-name regex above can't catch that variable-interpolated form.
# generate_downstream_scripts.py:818 (CRA branch) had exactly this bug, which
# slipped past the template-only test because the path is composed in Python,
# not written as a literal in the sbatch template. Guard the Path-segment form.
_PY_GENERATORS_NO_DATASETS_SEGMENT = [
    "generate_downstream_scripts.py",
    "generate_scripts.py",
    "generate_table_embedding_scripts.py",
    "generate_row_scripts.py",
    "generate_row_data_scripts.py",
    "generate_row_data_downstream_scripts.py",
    "submit_all.py",
    "submit_downstream.py",
]


@pytest.mark.parametrize("filename", _PY_GENERATORS_NO_DATASETS_SEGMENT)
def test_no_legacy_datasets_path_segment(filename):
    """Python generators must not construct `<root> / 'datasets' / <name>`
    Path segments — TRL-Bench's data root is `data/`. Catches the
    variable-interpolated form the literal-name test misses."""
    path = SLURM_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    text = path.read_text()
    # Match `'datasets'` or `"datasets"` used as a Path segment (preceded or
    # followed by ` / `), not the dict-key access `task_config.get('datasets'`.
    seg_hits = re.findall(r"/\s*['\"]datasets['\"]|['\"]datasets['\"]\s*/", text)
    assert not seg_hits, (
        f"{filename}: constructs a legacy `datasets/` Path segment "
        f"({len(seg_hits)} occurrence(s)); use `data` instead"
    )
