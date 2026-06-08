#!/usr/bin/env python3
"""
Generate and optionally submit the standard downstream matrix across heads.

This helper runs the standard downstream pipeline in multiple passes so we can
cover:
- all model-based tasks
- embedding-free baselines
- all supported probe heads

It avoids two common pitfalls:
- non-MLP heads should only run on probe-capable tasks
- baselines should only be submitted once (on the MLP pass)

DLTE remains a separate pipeline handled by submit_dlte.py.

Examples:
    # Show the full plan without writing or submitting
    python slurm/submit_downstream_matrix.py --dry-run

    # Prepare and submit the full standard matrix for seed 72
    python slurm/submit_downstream_matrix.py --seeds 72

    # Limit to a subset of models or datasets
    python slurm/submit_downstream_matrix.py --seeds 72 --models bert gte random
    python slurm/submit_downstream_matrix.py --datasets santos tus
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _embedding_model_names(project_root: Path) -> list[str]:
    """Use the same discovery rules as the downstream generator."""
    from generate_downstream_scripts import (
        discover_embeddings,
        discover_row_embeddings,
        discover_table_embeddings,
    )

    models: set[str] = set()
    for discovered in (
        discover_embeddings(project_root),
        discover_table_embeddings(project_root),
        discover_row_embeddings(project_root),
    ):
        models.update(discovered.keys())
    return sorted(models)


def _requested_embedding_models(
    requested_models: list[str] | None,
    embedding_models: list[str],
) -> list[str] | None:
    """Strip baseline names from a user-provided model filter for non-MLP passes."""
    if not requested_models:
        return embedding_models
    filtered = [m for m in requested_models if m in set(embedding_models)]
    return filtered


def _run(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    if dry_run:
        print(f"[DRY-RUN] {printable}")
        return

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and submit the standard downstream matrix across heads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Seeds to use for seeded downstream tasks (default: [42])")
    parser.add_argument("--models", nargs="+",
                        help="Optional model filter. May include baseline names on the MLP pass.")
    parser.add_argument("--datasets", nargs="+",
                        help="Optional dataset filter applied to every pass.")
    parser.add_argument("--row-embedding-root", type=str, default=None,
                        help="Overlay row embedding root for row-level tasks "
                             "(e.g., embeddings/row_dim768)")
    parser.add_argument("--result-tag", type=str, default=None,
                        help="Row-level result tag (default: derived from --row-embedding-root)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between sbatch submissions within each pass (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the per-pass commands without executing them")
    args = parser.parse_args()

    project_root = get_project_root()
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    from generate_downstream_scripts import PROBE_TASKS, COSINE_THRESHOLD_TASKS, DLTE_TASKS

    tasks_config = load_yaml(project_root / "slurm" / "config" / "downstream" / "tasks.yaml")
    available_tasks = sorted(t for t in tasks_config["tasks"].keys() if t not in DLTE_TASKS)
    probe_tasks = sorted(t for t in available_tasks if t in PROBE_TASKS)
    cosine_tasks = sorted(t for t in available_tasks if t in COSINE_THRESHOLD_TASKS)

    embedding_models = _embedding_model_names(project_root)
    nonbaseline_models = _requested_embedding_models(args.models, embedding_models)

    passes: list[tuple[str, list[str], list[str] | None]] = [
        ("mlp", available_tasks, args.models),
        ("linear", probe_tasks, nonbaseline_models),
        ("dummy", probe_tasks, nonbaseline_models),
    ]
    if cosine_tasks:
        passes.append(("cosine_threshold", cosine_tasks, nonbaseline_models))

    print("Standard downstream matrix passes:")
    for head, tasks, models in passes:
        model_count = "all" if models is None else len(models)
        print(f"  {head}: {len(tasks)} tasks, models={model_count}")
    print()

    gen_script = project_root / "slurm" / "generate_downstream_scripts.py"
    submit_script = project_root / "slurm" / "submit_downstream.py"

    for head, tasks, model_filter in passes:
        print(f"=== {head} pass ===")

        gen_cmd = [
            sys.executable, str(gen_script),
            "--head-type", head,
            "--seeds", *[str(s) for s in args.seeds],
            "--tasks", *tasks,
        ]
        if model_filter:
            gen_cmd.extend(["--models", *model_filter])
        if args.datasets:
            gen_cmd.extend(["--datasets", *args.datasets])
        if args.row_embedding_root:
            gen_cmd.extend(["--row-embedding-root", args.row_embedding_root])
        if args.result_tag:
            gen_cmd.extend(["--result-tag", args.result_tag])
        if args.dry_run:
            gen_cmd.append("--dry-run")

        _run(gen_cmd, dry_run=args.dry_run)

        submit_cmd = [
            sys.executable, str(submit_script),
            "--head-type", head,
            "--no-generate",
            "--delay", str(args.delay),
            "--tasks", *tasks,
        ]
        if model_filter:
            submit_cmd.extend(["--models", *model_filter])
        if args.datasets:
            submit_cmd.extend(["--datasets", *args.datasets])
        if args.row_embedding_root:
            submit_cmd.extend(["--row-embedding-root", args.row_embedding_root])
        if args.result_tag:
            submit_cmd.extend(["--result-tag", args.result_tag])
        if args.dry_run:
            submit_cmd.append("--dry-run")

        _run(submit_cmd, dry_run=args.dry_run)
        print()

    print("DLTE is not included in this helper. Use submit_dlte.py separately when needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
