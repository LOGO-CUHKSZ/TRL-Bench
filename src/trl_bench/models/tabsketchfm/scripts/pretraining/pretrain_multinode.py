from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class LaunchConfig:
    nodes: List[str]
    nproc_per_node: int
    master_addr: str
    master_port: int
    workspace: Path
    pretrain_script: Path
    python_executable: str | None
    torchrun_cmd: str
    ssh_user: str | None
    ssh_extra_args: str | None
    env_exports: List[str]
    log_dir: Path
    dry_run: bool
    forward_args: List[str]
    pre_commands: List[str]
    shell_executable: str


def _normalize_hostname(host: str) -> str:
    if "@" in host:
        host = host.split("@", 1)[1]
    return host.split(".", 1)[0]


def _format_host(host: str, ssh_user: str | None) -> str:
    if "@" in host or ssh_user is None:
        return host
    return f"{ssh_user}@{host}"


def _quote_cmd(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _build_torchrun_command(cfg: LaunchConfig, node_rank: int) -> str:
    num_nodes = len(cfg.nodes)
    pretrain_parts: List[str] = []
    if cfg.python_executable:
        pretrain_parts.append(cfg.python_executable)
    pretrain_parts.append(str(cfg.pretrain_script))
    pretrain_parts += cfg.forward_args
    torchrun_parts = [
        cfg.torchrun_cmd,
        "--nnodes",
        str(num_nodes),
        "--nproc_per_node",
        str(cfg.nproc_per_node),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        f"{cfg.master_addr}:{cfg.master_port}",
        "--node_rank",
        str(node_rank),
        "--max_restarts",
        "0",
    ]
    torchrun_parts += pretrain_parts
    env_prefix = ""
    if cfg.env_exports:
        env_prefix = " ".join(cfg.env_exports) + " "
    return f"{env_prefix}{_quote_cmd(torchrun_parts)}"


def _build_shell_command(cfg: LaunchConfig, node_rank: int) -> str:
    torchrun_cmd = _build_torchrun_command(cfg, node_rank)
    segments = [f"cd {shlex.quote(str(cfg.workspace))}"]
    segments.extend(cfg.pre_commands)
    segments.append(torchrun_cmd)
    return " && ".join(segments)


def _launch_on_node(cfg: LaunchConfig, node_rank: int, raw_host: str) -> subprocess.Popen:
    host_for_display = _normalize_hostname(raw_host)
    log_path = cfg.log_dir / f"node_{node_rank}_{host_for_display}.log"
    cmd = _build_shell_command(cfg, node_rank)
    local_host = _normalize_hostname(socket.gethostname())
    run_locally = _normalize_hostname(raw_host) == local_host

    shell_invocation = [cfg.shell_executable, "-lc", cmd]
    shell_invocation_str = f"{cfg.shell_executable} -lc {shlex.quote(cmd)}"

    if cfg.dry_run:
        print(f"[dry-run] rank={node_rank} host={raw_host} -> {cmd}")
        return None

    log_file = log_path.open("w")

    if run_locally:
        process = subprocess.Popen(
            shell_invocation,
            stdout=log_file,
            stderr=log_file,
        )
    else:
        ssh_cmd: List[str] = ["ssh"]
        if cfg.ssh_extra_args:
            ssh_cmd.extend(shlex.split(cfg.ssh_extra_args))
        ssh_cmd.append(_format_host(raw_host, cfg.ssh_user))
        ssh_cmd.append(shell_invocation_str)
        process = subprocess.Popen(
            ssh_cmd,
            stdout=log_file,
            stderr=log_file,
        )
    process._log_file = log_file  # type: ignore[attr-defined]
    return process


def parse_args(argv: Sequence[str] | None = None) -> LaunchConfig:
    parser = argparse.ArgumentParser(
        description="Launch TabSketchFM pretraining across multiple nodes via torchrun."
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=["kn091", "kn092", "kn093", "kn094"],
        help="Ordered list of hostnames to participate in training (default: kn091 kn092 kn093 kn094).",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=4,
        help="Number of GPUs per node (torchrun --nproc_per_node argument).",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=None,
        help="Address of the rendezvous/master node (defaults to the first host).",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=12951,
        help="Rendezvous port shared by all nodes.",
    )
    # Default workspace is project root (two levels up from scripts/pretraining/)
    # Go 4 levels up: pretrain_multinode.py -> pretraining -> scripts -> tabsketchfm -> models -> project_root
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(project_root),
        help="Path to the shared project directory on every node.",
    )
    parser.add_argument(
        "--pretrain_script",
        type=str,
        default="models/tabsketchfm/pretrain.py",
        help="Relative or absolute path to the training entry script.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=str,
        default=None,
        help="Optional Python executable to insert before the training script (omit to rely on the environment).",
    )
    parser.add_argument(
        "--torchrun",
        dest="torchrun_cmd",
        type=str,
        default="torchrun",
        help="torchrun executable to invoke.",
    )
    parser.add_argument(
        "--ssh-user",
        dest="ssh_user",
        type=str,
        default=None,
        help="SSH username (defaults to current user).",
    )
    parser.add_argument(
        "--ssh-extra-args",
        dest="ssh_extra_args",
        type=str,
        default=None,
        help="Additional flags passed to every ssh invocation (e.g. '-o StrictHostKeyChecking=no').",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra KEY=VALUE pairs exported before torchrun (can be supplied multiple times).",
    )
    parser.add_argument(
        "--pre-cmd",
        action="append",
        default=[],
        help="Shell snippet executed before cd/torchrun on every node (repeatable, order preserved).",
    )
    parser.add_argument(
        "--shell",
        dest="shell_executable",
        type=str,
        default="bash",
        help="Shell used to run commands on each node (default: bash).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="multinode_logs",
        help="Directory (relative to workspace) for per-node stdout/stderr logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without launching anything.",
    )
    parser.add_argument(
        "pretrain_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to pretrain.py (prefix with --).",
    )

    raw_args = parser.parse_args(argv)

    forward_args = list(raw_args.pretrain_args)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    workspace = Path(raw_args.workspace).resolve()
    if not workspace.exists():
        parser.error(f"Workspace path does not exist: {workspace}")

    pretrain_script = Path(raw_args.pretrain_script)
    if not pretrain_script.is_absolute():
        pretrain_script = workspace / pretrain_script
    pretrain_script = pretrain_script.resolve()
    if not pretrain_script.exists():
        parser.error(f"Training script not found: {pretrain_script}")

    log_dir = Path(raw_args.log_dir)
    if not log_dir.is_absolute():
        log_dir = workspace / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    nodes = raw_args.nodes
    if not nodes:
        parser.error("At least one node must be specified")

    master_addr = raw_args.master_addr or _normalize_hostname(nodes[0])

    env_exports: List[str] = []
    for item in raw_args.env:
        if "=" not in item:
            parser.error(f"--env entries must be in KEY=VALUE format (got '{item}')")
        key, value = item.split("=", 1)
        if not key:
            parser.error(f"--env entry has empty key (got '{item}')")
        env_exports.append(f"{key}={shlex.quote(value)}")

    return LaunchConfig(
        nodes=nodes,
        nproc_per_node=raw_args.nproc_per_node,
        master_addr=master_addr,
        master_port=raw_args.master_port,
        workspace=workspace,
        pretrain_script=pretrain_script,
        python_executable=raw_args.python_executable,
        torchrun_cmd=raw_args.torchrun_cmd,
        ssh_user=raw_args.ssh_user,
        ssh_extra_args=raw_args.ssh_extra_args,
        env_exports=env_exports,
        log_dir=log_dir,
        dry_run=raw_args.dry_run,
        forward_args=forward_args,
        pre_commands=raw_args.pre_cmd,
        shell_executable=raw_args.shell_executable,
    )


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    processes = []

    try:
        for node_rank, host in enumerate(cfg.nodes):
            process = _launch_on_node(cfg, node_rank, host)
            if process is not None:
                processes.append((node_rank, host, process))

        if cfg.dry_run:
            return

        failed = False
        for node_rank, host, process in processes:
            return_code = process.wait()
            log_file = getattr(process, "_log_file", None)
            if log_file:
                log_file.close()
            if return_code != 0:
                print(
                    f"[error] rank {node_rank} on {host} exited with code {return_code}",
                    file=sys.stderr,
                )
                failed = True

        if failed:
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted, terminating remote processes...")
        for _, _, process in processes:
            process.terminate()
    finally:
        for _, _, process in processes:
            log_file = getattr(process, "_log_file", None)
            if log_file and not log_file.closed:
                log_file.close()


if __name__ == "__main__":
    main()
