"""Thin wall-clock timer for existing embedding generation scripts.

Wraps any command with:
  - Wall-clock timing (time.perf_counter)
  - Peak GPU VRAM tracking (nvidia-smi polling, SLURM-aware GPU selection)
  - Timeout with headroom below SLURM walltime
  - OOM detection from stderr
  - Output existence verification
  - JSON result with SLURM metadata

Usage::

    python -m effbench.timer \\
        --model bert --workload column --dataset-id openml_46918 \\
        --dataset-source eff_real --expected-output /path/to/output.pkl \\
        --output-dir effbench/results \\
        -- python models/bert/generate_column_embeddings.py --input /path --output /path/to/output.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from effbench.spec import EffBenchResult


# ---------------------------------------------------------------------------
# SLURM-aware GPU resolution
# ---------------------------------------------------------------------------

def resolve_gpu_id() -> tuple[int, str]:
    """Resolve the physical GPU ID for nvidia-smi on SLURM.

    SLURM sets CUDA_VISIBLE_DEVICES to remap GPUs. The model sees cuda:0,
    but nvidia-smi needs the physical GPU ID.

    Returns:
        (physical_gpu_id, gpu_name)
    """
    # Try SLURM-specific env vars first
    gpu_str = (
        os.environ.get("SLURM_STEP_GPUS")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )

    if gpu_str:
        # Could be "0", "2", "0,1", or UUID-based
        first = gpu_str.split(",")[0].strip()
        try:
            physical_id = int(first)
        except ValueError:
            # UUID format — query by UUID
            physical_id = 0  # fallback
    else:
        physical_id = 0

    # Get GPU name
    gpu_name = ""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             f"--id={physical_id}"],
            text=True, timeout=5,
        )
        gpu_name = out.strip().split("\n")[0]
    except Exception:
        pass

    return physical_id, gpu_name


# ---------------------------------------------------------------------------
# GPU VRAM polling
# ---------------------------------------------------------------------------

class GpuVramMonitor:
    """Polls nvidia-smi in a background thread to track peak GPU VRAM."""

    def __init__(self, physical_gpu_id: int = 0, poll_interval: float = 0.5):
        self.physical_gpu_id = physical_gpu_id
        self.poll_interval = poll_interval
        self.peak_mb: float = 0.0
        self.poll_count: int = 0
        self.poll_ok: bool = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits",
                     f"--id={self.physical_gpu_id}"],
                    text=True, timeout=5,
                )
                mb = float(out.strip().split("\n")[0])
                self.poll_count += 1
                self.poll_ok = True
                if mb > self.peak_mb:
                    self.peak_mb = mb
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def start(self):
        self._stop.clear()
        self.peak_mb = 0.0
        self.poll_count = 0
        self.poll_ok = False
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak_mb


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_hardware_string(physical_gpu_id: int = 0) -> str:
    parts = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             f"--id={physical_gpu_id}"],
            text=True, timeout=5,
        )
        parts.append(out.strip())
    except Exception:
        parts.append(platform.machine())

    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / (1024**3)
        parts.append(f"{cores} cores, {ram_gb:.0f}GB RAM")
    except ImportError:
        pass

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------

def verify_output(expected_output: str | None) -> tuple[bool, str]:
    """Check that the expected output exists and is non-empty.

    Returns (verified, message).
    """
    if not expected_output:
        return True, "no output path specified"

    path = Path(expected_output)
    if path.is_file():
        size = path.stat().st_size
        if size > 0:
            return True, f"file exists ({size} bytes)"
        return False, "file exists but is empty"
    elif path.is_dir():
        contents = list(path.iterdir())
        if contents:
            return True, f"directory exists ({len(contents)} items)"
        return False, "directory exists but is empty"
    return False, "output not found"


# ---------------------------------------------------------------------------
# Core timing function
# ---------------------------------------------------------------------------

def time_command(
    cmd: List[str],
    model_name: str = "",
    workload: str = "",
    dataset_id: str = "",
    dataset_source: str = "",
    needs_training: bool = False,
    n_rows: int = 0,
    n_columns: int = 0,
    timeout: int = 3300,
    expected_output: str | None = None,
    env_setup: str = "",
) -> EffBenchResult:
    """Time a shell command and return an EffBenchResult."""
    # Resolve GPU
    physical_gpu_id, gpu_name = resolve_gpu_id()
    hardware = get_hardware_string(physical_gpu_id)

    # SLURM metadata
    hostname = socket.gethostname()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")

    # Pre-create output directory
    if expected_output:
        out_path = Path(expected_output)
        if out_path.suffix:  # looks like a file
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:  # looks like a directory
            out_path.mkdir(parents=True, exist_ok=True)

    # Start GPU monitoring
    gpu_monitor = GpuVramMonitor(physical_gpu_id=physical_gpu_id)
    gpu_monitor.start()

    # Run the command
    status = "success"
    error_message = ""
    return_code = -1
    start = time.perf_counter()

    try:
        # Join command into a string for shell execution.
        cmd_str = " ".join(cmd)
        # Prepend env_setup if specified (e.g., "source models/tabert/load_env")
        if env_setup:
            cmd_str = env_setup + " && " + cmd_str
        proc = subprocess.run(
            cmd_str,
            shell=True,
            executable="/bin/bash",
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT) + ":" + os.environ.get("PYTHONPATH", "")},
        )
        elapsed = time.perf_counter() - start
        return_code = proc.returncode

        if proc.returncode != 0:
            stderr = proc.stderr[-2000:] if proc.stderr else ""
            if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
                status = "oom"
                error_message = "CUDA out of memory"
            else:
                status = "error"
                error_message = stderr[-500:] if stderr else f"exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        status = "timeout"
        error_message = f"Timed out after {timeout}s"

    except Exception as e:
        elapsed = time.perf_counter() - start
        status = "error"
        error_message = str(e)

    # Stop GPU monitoring
    peak_vram = gpu_monitor.stop()

    # Verify output
    output_verified = False
    output_msg = ""
    if status == "success":
        output_verified, output_msg = verify_output(expected_output)
        if not output_verified and expected_output:
            status = "error"
            error_message = f"Output verification failed: {output_msg}"

    return EffBenchResult(
        model_name=model_name,
        workload=workload,
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        needs_training=needs_training,
        script=" ".join(cmd),
        wall_clock_seconds=elapsed,
        peak_gpu_vram_mb=peak_vram,
        n_rows=n_rows,
        n_columns=n_columns,
        hardware=hardware,
        gpu_name=gpu_name,
        device=f"cuda:{physical_gpu_id}",
        hostname=hostname,
        slurm_job_id=slurm_job_id,
        slurm_array_task_id=slurm_array_task_id,
        return_code=return_code,
        vram_monitor_ok=gpu_monitor.poll_ok,
        output_verified=output_verified,
        expected_output=expected_output or "",
        timeout_seconds=timeout,
        status=status,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Time an embedding generation script",
        usage="python -m effbench.timer [OPTIONS] -- COMMAND...",
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--workload", type=str, default="")
    parser.add_argument("--dataset-id", type=str, default="")
    parser.add_argument("--dataset-source", type=str, default="")
    parser.add_argument("--needs-training", action="store_true")
    parser.add_argument("--n-rows", type=int, default=0)
    parser.add_argument("--n-columns", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3300)
    parser.add_argument("--expected-output", type=str, default=None,
                        help="Path to expected output file/dir for verification")
    parser.add_argument("--env-setup", type=str, default="",
                        help="Shell command to run before the main command (e.g., 'source venv/bin/activate')")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "effbench" / "results"))

    args, cmd = parser.parse_known_args()

    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        parser.error("No command specified. Use: python -m effbench.timer [OPTIONS] -- COMMAND...")

    print(f"{'=' * 60}")
    print(f"TRL-EffBench Timer")
    print(f"  Model: {args.model} / {args.workload}")
    print(f"  Dataset: {args.dataset_id} ({args.dataset_source})")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = time_command(
        cmd=cmd,
        model_name=args.model,
        workload=args.workload,
        dataset_id=args.dataset_id,
        dataset_source=args.dataset_source,
        needs_training=args.needs_training,
        n_rows=args.n_rows,
        n_columns=args.n_columns,
        timeout=args.timeout,
        expected_output=args.expected_output,
        env_setup=args.env_setup,
    )

    # Print summary
    print(f"\n--- Result ---")
    print(f"  Status: {result.status}")
    print(f"  Wall clock: {result.wall_clock_seconds:.2f}s")
    print(f"  Peak GPU VRAM: {result.peak_gpu_vram_mb:.0f} MB"
          f" (monitor={'ok' if result.vram_monitor_ok else 'FAILED'})")
    print(f"  Output verified: {result.output_verified}")
    print(f"  GPU: {result.gpu_name} (physical {result.device})")
    print(f"  Host: {result.hostname}")
    if result.error_message:
        print(f"  Error: {result.error_message}")

    # Save result
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.model}_{args.workload}_{args.dataset_id}_{timestamp}.json"
    result_path = out_dir / filename
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  Saved: {result_path}")


if __name__ == "__main__":
    main()
