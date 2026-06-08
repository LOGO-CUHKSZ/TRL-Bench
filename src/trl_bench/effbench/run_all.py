"""Generate timed commands for all (model, workload, dataset) combinations.

Uses the existing embedding generation scripts wrapped with effbench.timer.
For each job, generates a command like:

    python -m effbench.timer --model bert --workload column ... \\
        -- python models/bert/generate_column_embeddings.py --input /path --output /tmp/out.pkl

Usage::

    python -m effbench.run_all --print          # List all commands
    python -m effbench.run_all --slurm          # Generate SLURM job array
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from effbench.spec import MODEL_REGISTRY, Workload


def collect_datasets(effbench_dir: Path) -> List[Dict]:
    """Collect all efficiency test datasets with metadata.

    Returns list of dicts with: path, dataset_id, source, workload_type ("row" or "column"),
    n_rows, n_columns.
    """
    datasets = []

    # --- Eff-Real row anchors ---
    anchors_file = effbench_dir / "anchors_row.json"
    if anchors_file.exists():
        with open(anchors_file) as f:
            data = json.load(f)
        for anchor in data.get("anchors", []):
            table_id = anchor["table_id"]
            datasets.append({
                "path": str(PROJECT_ROOT / "datasets" / "row_data" / table_id),
                "dataset_id": table_id,
                "source": "eff_real",
                "workload_type": "row",  # These are row-level datasets
                "n_rows": anchor.get("n_rows", 0),
                "n_columns": anchor.get("n_columns", 0),
            })

    # --- Eff-Scale row sweeps ---
    row_dir = effbench_dir / "scale_suite" / "row"
    if row_dir.exists():
        for d in sorted(row_dir.iterdir()):
            if not d.is_dir() or not (d / "data.csv").exists():
                continue
            meta = {}
            meta_file = d / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            datasets.append({
                "path": str(d),
                "dataset_id": d.name,
                "source": "eff_scale",
                "workload_type": "row",
                "n_rows": meta.get("n_rows", 0),
                "n_columns": meta.get("n_features", 0),
            })

    # --- Eff-Scale column sweeps ---
    col_dir = effbench_dir / "scale_suite" / "column"
    if col_dir.exists():
        for d in sorted(col_dir.iterdir()):
            if not d.is_dir() or not (d / "data.csv").exists():
                continue
            meta = {}
            meta_file = d / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            datasets.append({
                "path": str(d),
                "dataset_id": d.name,
                "source": "eff_scale",
                "workload_type": "column",
                "n_rows": meta.get("n_context_rows", 0),
                "n_columns": meta.get("n_columns", 0),
            })

    # --- Bridge tables (usable for both row and column) ---
    bridge_dir = effbench_dir / "scale_suite" / "bridge"
    if bridge_dir.exists():
        for d in sorted(bridge_dir.iterdir()):
            if not d.is_dir() or not (d / "data.csv").exists():
                continue
            meta = {}
            meta_file = d / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            datasets.append({
                "path": str(d),
                "dataset_id": d.name,
                "source": "bridge",
                "workload_type": "both",
                "n_rows": meta.get("n_rows", 0),
                "n_columns": meta.get("n_features", 0),
            })

    return datasets


def build_model_command(
    model_name: str,
    workload: Workload,
    dataset: Dict,
    output_base: Path,
    timeout: int = 3300,
) -> str | None:
    """Build the full timed command for one (model, workload, dataset) combination.

    Returns the command string, or None if the combination is invalid.
    """
    info = MODEL_REGISTRY[model_name]
    scripts = info.get("scripts", {})

    if workload not in scripts:
        return None

    # Gate: models requiring dataset.json can only run on eff_real anchors
    if info.get("requires_dataset_json") and dataset["source"] != "eff_real":
        return None

    # Gate: models requiring pre-existing checkpoints are skipped for now
    if info.get("requires_pretrained_checkpoint"):
        return None

    script_info = scripts[workload]
    script_path = script_info["script"]
    input_arg = script_info["input_arg"]
    output_arg = script_info["output_arg"]
    output_kind = script_info.get("output_kind", "file")
    extra_args = script_info.get("extra_args", [])

    ds_path = dataset["path"]
    ds_id = dataset["dataset_id"]
    ds_source = dataset["source"]

    # Build the output path for the embeddings
    emb_output = output_base / "embeddings" / model_name / workload.value / ds_id

    # Build the inner command (the actual model script)
    inner_cmd_parts = ["python", script_path]

    # Input argument
    inner_cmd_parts.extend([input_arg, ds_path])

    # Output argument — respect output_kind
    if output_kind == "dir":
        expected_output = str(emb_output)
        inner_cmd_parts.extend([output_arg, expected_output])
    else:
        expected_output = str(emb_output / "embeddings.pkl")
        inner_cmd_parts.extend([output_arg, expected_output])

    # SSL models need a checkpoint dir
    if script_info.get("needs_checkpoint_dir"):
        ckpt_dir = output_base / "checkpoints" / model_name / ds_id
        inner_cmd_parts.extend(["--checkpoint_base_dir", str(ckpt_dir)])

    inner_cmd_parts.extend(extra_args)
    inner_cmd = " ".join(inner_cmd_parts)

    # Build the outer timer command
    env_setup = info.get("env_setup", "")
    timer_parts = [
        "python -m effbench.timer",
        f"--model {model_name}",
        f"--workload {workload.value}",
        f"--dataset-id {ds_id}",
        f"--dataset-source {ds_source}",
        f"--n-rows {dataset.get('n_rows', 0)}",
        f"--n-columns {dataset.get('n_columns', 0)}",
        f"--timeout {timeout}",
        f"--expected-output {expected_output}",
        f"--output-dir {output_base / 'results'}",
    ]
    if env_setup:
        timer_parts.append(f"--env-setup '{env_setup}'")
    if info.get("needs_training"):
        timer_parts.append("--needs-training")

    return " ".join(timer_parts) + " -- " + inner_cmd


def is_compatible(workload: Workload, dataset: Dict) -> bool:
    """Check if a workload is compatible with a dataset type."""
    wl_type = dataset["workload_type"]
    if wl_type == "both":
        return True
    if workload == Workload.ROW and wl_type == "row":
        return True
    if workload in (Workload.COLUMN, Workload.TABLE) and wl_type == "column":
        return True
    # Row anchors can also be used for column/table workloads (they're just tables)
    if workload in (Workload.COLUMN, Workload.TABLE) and wl_type == "row":
        return True
    return False


def generate_jobs(effbench_dir: Path, output_base: Path) -> tuple[List[str], List[str]]:
    """Generate all valid (model, workload, dataset) commands.

    Returns two lists: (frozen_commands, training_commands) for split SLURM arrays.
    """
    datasets = collect_datasets(effbench_dir)
    frozen_cmds = []
    training_cmds = []

    # Track which (model, script, dataset) combos we've already generated
    # to deduplicate column/table when they use the same script
    seen_scripts = set()

    for model_name, info in MODEL_REGISTRY.items():
        if info["family"].value == "api":
            continue

        needs_training = info.get("needs_training", False)
        timeout = 13500 if needs_training else 3300  # 3h45m vs 55min

        for workload in info["workloads"]:
            scripts = info.get("scripts", {})
            if workload not in scripts:
                continue
            script_path = scripts[workload]["script"]

            for dataset in datasets:
                if not is_compatible(workload, dataset):
                    continue

                # Deduplicate: if column and table use the same script, only run once
                dedup_key = (model_name, script_path, dataset["dataset_id"])
                if dedup_key in seen_scripts:
                    continue
                seen_scripts.add(dedup_key)

                cmd = build_model_command(model_name, workload, dataset, output_base, timeout=timeout)
                if cmd:
                    if needs_training:
                        training_cmds.append(cmd)
                    else:
                        frozen_cmds.append(cmd)

    return frozen_cmds, training_cmds


def write_slurm_script(
    commands: List[str], output_path: Path,
    job_name: str = "effbench", time_limit: str = "01:00:00",
    mem: str = "32G", concurrency: int = 32,
) -> None:
    """Write a SLURM job array submission script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_file = output_path.with_suffix(".cmds")
    with open(cmd_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=effbench/logs/%A_%a.out
#SBATCH --error=effbench/logs/%A_%a.err
#SBATCH --array=1-{len(commands)}%{concurrency}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Load environment
source load_env

# Set up Python path
export PYTHONPATH={PROJECT_ROOT}:$PYTHONPATH

# Use existing HuggingFace cache
export HF_HOME=${{HF_HOME:-$HOME/.cache/huggingface}}

# Get the command for this array task
CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {cmd_file})

echo "=== Job $SLURM_JOB_ID Task $SLURM_ARRAY_TASK_ID ==="
echo "Host: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "$CMD"
eval $CMD
"""
    with open(output_path, "w") as f:
        f.write(script)

    print(f"  Script: {output_path}")
    print(f"  Commands: {cmd_file}")
    print(f"  Jobs: {len(commands)}, concurrency: %{concurrency}")
    print(f"  Time: {time_limit}, Mem: {mem}")


def main():
    parser = argparse.ArgumentParser(description="Generate EffBench timed jobs")
    parser.add_argument("--print", action="store_true", help="Print all commands")
    parser.add_argument("--slurm", action="store_true", help="Generate SLURM scripts")
    parser.add_argument("--output-base", type=str,
                        default=str(PROJECT_ROOT / "effbench"))
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max concurrent SLURM array jobs")
    args = parser.parse_args()

    effbench_dir = PROJECT_ROOT / "effbench"
    output_base = Path(args.output_base)
    frozen_cmds, training_cmds = generate_jobs(effbench_dir, output_base)

    total = len(frozen_cmds) + len(training_cmds)

    if args.print:
        print("=== Frozen (inference-only) ===")
        for cmd in frozen_cmds:
            print(cmd)
        print(f"\n=== Training (SSL/transfer) ===")
        for cmd in training_cmds:
            print(cmd)
        print(f"\nFrozen: {len(frozen_cmds)}, Training: {len(training_cmds)}, Total: {total}")

    if args.slurm:
        jobs_dir = effbench_dir / "jobs"

        print("=== Frozen models (1h walltime) ===")
        if frozen_cmds:
            write_slurm_script(
                frozen_cmds, jobs_dir / "run_frozen.sh",
                job_name="effbench_frozen", time_limit="01:00:00",
                concurrency=args.concurrency,
            )

        print("\n=== Training models (4h walltime) ===")
        if training_cmds:
            write_slurm_script(
                training_cmds, jobs_dir / "run_training.sh",
                job_name="effbench_train", time_limit="04:00:00",
                mem="48G", concurrency=args.concurrency,
            )

        print(f"\nTotal: {total} jobs ({len(frozen_cmds)} frozen + {len(training_cmds)} training)")
        print(f"\nSubmit with:")
        print(f"  sbatch {jobs_dir / 'run_frozen.sh'}")
        print(f"  sbatch {jobs_dir / 'run_training.sh'}")

    if not args.print and not args.slurm:
        print(f"Frozen: {len(frozen_cmds)}, Training: {len(training_cmds)}, Total: {total}")
        print("Use --print to list or --slurm to create SLURM scripts.")


if __name__ == "__main__":
    main()
