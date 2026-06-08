#!/usr/bin/env python3
"""CLI for embedding repair (scan + repair)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# When invoked as `python -m trl_bench.utils.embedding_repair.cli`, the
# `trl_bench` package is already importable. When invoked as a bare script,
# ensure `src/` is on sys.path so `trl_bench.*` resolves.
SRC_ROOT = SCRIPT_DIR.parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trl_bench.utils.embedding_repair.config import (
    get_project_root,
    load_models_config,
    default_embeddings_path,
    default_checkpoint_path,
    default_model_args,
)
from trl_bench.utils.embedding_repair.registry import get_adapter
from trl_bench.utils.embedding_repair.core import scan_embeddings, repair_embeddings


def build_adapter(
    model: str,
    device: str | None,
    checkpoint_override: str | None,
    max_rows_override: int | None,
    models_cfg=None,
    root: Path | None = None,
):
    root = root or get_project_root()
    models_cfg = models_cfg or load_models_config(root)
    ckpt = checkpoint_override or default_checkpoint_path(model, models_cfg, root)
    defaults = default_model_args(model, models_cfg)

    if model == "bert":
        return get_adapter(model, model_name=ckpt, device=device)
    if model == "gte":
        return get_adapter(model, model_name=ckpt, device=device)
    if model == "doduo":
        return get_adapter(model, checkpoint_path=ckpt, model_variant="wikitable", device=device)
    if model == "starmie":
        return get_adapter(model, checkpoint_path=ckpt, device=device)
    if model == "tabert":
        max_rows = max_rows_override or defaults.get("max_rows", 100)
        return get_adapter(model, checkpoint_path=ckpt, max_rows=max_rows, device=device)
    if model == "tabsketchfm":
        return get_adapter(model, checkpoint_path=ckpt, device=device)
    if model == "tabbie":
        max_rows = max_rows_override or defaults.get("max_rows", 30)
        return get_adapter(model, checkpoint_path=ckpt, max_rows=max_rows, device=device)
    if model == "tapas":
        max_rows = max_rows_override or defaults.get("max_rows", 100)
        return get_adapter(model, model_name=ckpt, max_rows=max_rows, device=device)

    raise ValueError(f"Unsupported model: {model}")


def main():
    parser = argparse.ArgumentParser(description="Scan/repair missing column embeddings.")
    parser.add_argument("--model", required=True, choices=["bert", "doduo", "gte", "starmie", "tabert", "tabbie", "tabsketchfm", "tapas"])
    parser.add_argument("--dataset", required=True, help="Dataset name (used for default paths)")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.pkl (defaults to embeddings/column/{model}/{dataset}.pkl)")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path or model name")
    parser.add_argument("--action", choices=["scan", "repair"], default="scan")
    parser.add_argument("--report", default=None, help="Report path (jsonl for scan, json for repair)")
    parser.add_argument("--output", default=None, help="Output path for repaired embeddings (defaults to in-place)")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    parser.add_argument("--max_rows", type=int, default=None, help="Override max_rows for repair embedding")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for recursive repair (defaults to adapter recommendation or 64)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit tables processed (debug)")
    parser.add_argument("--dry_run", action="store_true", help="Do not write repaired embeddings")
    parser.add_argument(
        "--allow_failures",
        action="store_true",
        help="Do not raise on remaining missing columns after repair",
    )

    args = parser.parse_args()
    root = get_project_root()
    models_cfg = load_models_config(root)
    defaults = default_model_args(args.model, models_cfg)
    effective_max_rows = args.max_rows or defaults.get("max_rows")
    embeddings_path = Path(args.embeddings) if args.embeddings else default_embeddings_path(args.model, args.dataset, root)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    adapter = build_adapter(
        args.model,
        args.device,
        args.checkpoint,
        args.max_rows,
        models_cfg=models_cfg,
        root=root,
    )

    report_path = Path(args.report) if args.report else None
    if args.action == "scan":
        scan_embeddings(
            adapter,
            embeddings_path,
            report_path=report_path,
            max_tables=args.limit,
        )
        return

    repair_embeddings(
        adapter,
        embeddings_path,
        output_path=Path(args.output) if args.output else None,
        report_path=report_path,
        max_rows=effective_max_rows,
        chunk_size=args.chunk_size,
        max_tables=args.limit,
        dry_run=args.dry_run,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
