#!/usr/bin/env python3
"""Generate sbatch files from configs/grid.yaml (single-cell tool).

For each valid (model, task, dataset, setting, probe, seed) cell in the grid,
write one .sbatch file that runs `python -m trl_bench.run ...` (the
auto-orchestrating single-cell driver) for that cell.

This is the README's reviewer-facing reproduction tool for individual verified
cells. For the FULL paper grid (Phase-1 embedding extraction shared across
(task, setting, probe, seed); Phase-2 downstream cells), use
`slurm/submit_all.py` + `slurm/submit_downstream.py` instead — the two-phase
pipeline factors out embedding extraction so it runs once per (model, dataset)
rather than once per cell.

Usage:
    python slurm/generate_jobs.py --config configs/grid.yaml [--out-dir slurm/jobs/]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from trl_bench.registry import is_valid_cell  # noqa: E402


def _render(template: str, **subs: str) -> str:
    out = template
    for k, v in subs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def generate(
    *, grid_path: Path, out_dir: Path, template: Path,
    models: list[str], datasets: dict[str, list[str]],
) -> int:
    """Walk the grid and write one sbatch per supported cell. Return count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = template.read_text()
    grid = yaml.safe_load(grid_path.read_text())
    seeds = grid["seeds"]

    count = 0
    for suite, cfg in grid.items():
        if suite == "seeds":
            continue
        for task in cfg.get("tasks", []):
            for ds in datasets.get(task, []):
                for setting in cfg.get("settings", [None]):
                    for probe in cfg.get("probes", [None]):
                        for model in models:
                            if not is_valid_cell(model, task):
                                continue
                            for seed in seeds:
                                name = f"{model}_{task}_{ds}_{setting}_{probe or 'none'}_seed{seed}"
                                path = out_dir / f"{name}.sbatch"
                                path.write_text(_render(
                                    tmpl,
                                    MODEL=model, TASK=task, DATASET=ds,
                                    SETTING=setting, PROBE=(probe or "none"),
                                    SEED=str(seed), JOB_NAME=name,
                                    OUT_DIR=str(out_dir),
                                ))
                                count += 1
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/grid.yaml")
    p.add_argument("--out-dir", default="slurm/jobs")
    p.add_argument("--template", default="slurm/templates/base.sbatch.tmpl")
    p.add_argument("--models-config", default="slurm/config/models.yaml")
    p.add_argument("--datasets-config", default="slurm/config/downstream/task_datasets.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # models.yaml has shape {"models": {model_name: {...}}, ...}
    models_cfg = yaml.safe_load(Path(args.models_config).read_text())
    if isinstance(models_cfg, dict) and "models" in models_cfg:
        models = list(models_cfg["models"].keys())
    else:
        models = list(models_cfg.keys()) if isinstance(models_cfg, dict) else list(models_cfg)

    # task_datasets.yaml has shape {"task_datasets": {task: {"datasets": {ds_name: {...}}}}}
    raw = yaml.safe_load(Path(args.datasets_config).read_text())
    if isinstance(raw, dict) and "task_datasets" in raw:
        datasets = {
            task: list(spec.get("datasets", {}).keys()) if isinstance(spec, dict) else list(spec)
            for task, spec in raw["task_datasets"].items()
        }
    else:
        datasets = raw if isinstance(raw, dict) else {}

    n = generate(
        grid_path=Path(args.config),
        out_dir=Path(args.out_dir),
        template=Path(args.template),
        models=models, datasets=datasets,
    )
    print(f"Generated {n} sbatch files in {args.out_dir}")
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
