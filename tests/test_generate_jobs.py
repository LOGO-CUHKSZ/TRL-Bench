"""Tests for slurm/generate_jobs.py."""
import os
import sys
import textwrap
import yaml
from pathlib import Path

# Allow importing slurm/ scripts directly
SLURM_DIR = Path(__file__).resolve().parent.parent / "slurm"
sys.path.insert(0, str(SLURM_DIR))

import generate_jobs as gj


def test_generate_emits_one_file_per_cell(tmp_path):
    grid = {
        "seeds": [42, 52],
        "ctbench": {
            "tasks": ["column_clustering"],
            "settings": ["cls_embedding"],
            "probes": [None],
        },
    }
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid))

    tmpl = tmp_path / "base.tmpl"
    tmpl.write_text("# {MODEL} {TASK} {DATASET} {SETTING} {PROBE} {SEED}\n")

    out = tmp_path / "jobs"
    n = gj.generate(
        grid_path=grid_path,
        out_dir=out,
        template=tmpl,
        models=["bert"],
        datasets={"column_clustering": ["sotab"]},
    )
    files = list(out.glob("*.sbatch"))
    assert len(files) == n
    # 1 model x 1 task x 1 dataset x 1 setting x 1 probe x 2 seeds = 2 jobs
    assert n == 2


def test_main_with_real_config_shapes(tmp_path, monkeypatch):
    """Smoke-test the main() YAML parsing against real-shape models.yaml / task_datasets.yaml."""
    # Real-shape models.yaml
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(textwrap.dedent("""
        models:
          bert: {script: bert.py}
          gte:  {script: gte.py}
        text_embedding_jobs: {}
        repair_defaults: {}
    """))

    # Real-shape task_datasets.yaml
    td_yaml = tmp_path / "task_datasets.yaml"
    td_yaml.write_text(textwrap.dedent("""
        task_datasets:
          column_clustering:
            datasets:
              sotab: {files: ['a.csv']}
              wiki_ct: {files: ['b.csv']}
    """))

    # Real-shape grid.yaml (minimal)
    grid_yaml = tmp_path / "grid.yaml"
    grid_yaml.write_text(textwrap.dedent("""
        seeds: [42]
        ctbench:
          tasks: [column_clustering]
          settings: [cls_embedding]
          probes: [null]
    """))

    tmpl = tmp_path / "base.tmpl"
    tmpl.write_text("# {MODEL} {TASK} {DATASET} {SETTING} {PROBE} {SEED} {OUT_DIR}\n")

    out = tmp_path / "jobs"

    rc = gj.main([
        "--config", str(grid_yaml),
        "--models-config", str(models_yaml),
        "--datasets-config", str(td_yaml),
        "--template", str(tmpl),
        "--out-dir", str(out),
    ])
    assert rc == 0
    # bert + gte (2 models) x 2 datasets (sotab + wiki_ct) x 1 setting x 1 probe x 1 seed = 4 sbatch
    files = sorted(out.glob("*.sbatch"))
    assert len(files) == 4
    # Spot-check that {OUT_DIR} substitution worked
    body = files[0].read_text()
    assert str(out) in body
    assert "{OUT_DIR}" not in body
