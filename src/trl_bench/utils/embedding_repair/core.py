"""Core logic for two-pass embedding repair."""

from __future__ import annotations

import json
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .adapters.base import BaseAdapter
from .io_utils import read_csv_header


def load_embeddings(path: Path) -> Tuple[Any, str]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data, "dict"
    return list(data), "list"


def save_embeddings(path: Path, data: Any, fmt: str) -> None:
    if fmt == "dict":
        payload = data
    else:
        payload = list(data)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)


def iter_records(data: Any, fmt: str) -> Iterable[Tuple[Any, Dict[str, Any]]]:
    if fmt == "dict":
        for key, record in data.items():
            yield key, record
    else:
        for idx, record in enumerate(data):
            yield idx, record


def update_record(data: Any, fmt: str, key: Any, record: Dict[str, Any]) -> None:
    if fmt == "dict":
        data[key] = record
    else:
        data[key] = record


def _missing_after_update(adapter: BaseAdapter, header: List[str], col_embeddings: Dict[int, Any]) -> List[int]:
    missing = []
    for i in range(len(header)):
        if i not in col_embeddings:
            missing.append(i)
            continue
        vec = col_embeddings[i]
        if adapter.is_missing_vector(vec):
            missing.append(i)
    return missing


def _embed_with_fallback(
    adapter: BaseAdapter,
    table_path: str,
    header: List[str],
    missing: List[int],
    col_embeddings: Dict[int, Any],
    max_rows: Optional[int],
    chunk_size: int,
    failures: Dict[str, List[int]],
) -> None:
    if not missing:
        return

    # Always chunk large requests up front to avoid oversized calls.
    try:
        chunk_size = max(1, int(chunk_size))
    except (TypeError, ValueError):
        chunk_size = 64
    if len(missing) > chunk_size:
        for i in range(0, len(missing), chunk_size):
            _embed_with_fallback(
                adapter,
                table_path,
                header,
                missing[i:i + chunk_size],
                col_embeddings,
                max_rows,
                chunk_size,
                failures,
            )
        return

    try:
        new_embs = adapter.embed_columns(table_path, missing, max_rows=max_rows)
    except Exception as exc:
        print(
            f"[repair] embed_columns failed for {table_path} cols={missing}: {exc}",
            file=sys.stderr,
        )
        new_embs = {}

    col_embeddings.update(new_embs)
    remaining = _missing_after_update(adapter, header, col_embeddings)
    remaining = [i for i in remaining if i in missing]

    if not remaining:
        return

    if len(remaining) == 1:
        # One-column retry: Some models (e.g., Doduo table-wise encoding) can
        # drop the *last* column when multiple columns are embedded together
        # due to the 512-token truncation. Retrying a single missing column
        # avoids a false failure and should be safe: this retry only happens
        # when a multi-column call leaves exactly one column missing.
        retry_idx = remaining[0]
        if len(missing) > 1:
            try:
                retry_embs = adapter.embed_columns(table_path, [retry_idx], max_rows=max_rows)
            except Exception as exc:
                print(
                    f"[repair] embed_columns failed for {table_path} cols={[retry_idx]}: {exc}",
                    file=sys.stderr,
                )
                retry_embs = {}
            col_embeddings.update(retry_embs)
            remaining_after_retry = _missing_after_update(adapter, header, col_embeddings)
            if retry_idx not in remaining_after_retry:
                return
        failures.setdefault(table_path, []).append(retry_idx)
        return

    # Split into smaller chunks
    if len(remaining) <= chunk_size:
        for idx in remaining:
            _embed_with_fallback(
                adapter,
                table_path,
                header,
                [idx],
                col_embeddings,
                max_rows,
                chunk_size,
                failures,
            )
        return

    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i:i + chunk_size]
        _embed_with_fallback(
            adapter,
            table_path,
            header,
            chunk,
            col_embeddings,
            max_rows,
            chunk_size,
            failures,
        )


def scan_embeddings(
    adapter: BaseAdapter,
    embeddings_path: Path,
    report_path: Optional[Path] = None,
    max_tables: Optional[int] = None,
) -> List[Dict[str, Any]]:
    data, fmt = load_embeddings(embeddings_path)
    report: List[Dict[str, Any]] = []
    count = 0

    for _, record in iter_records(data, fmt):
        table_path = adapter.get_table_path(record)
        if table_path is None:
            continue
        header = read_csv_header(table_path)
        missing = adapter.detect_missing(record, header)
        if missing:
            report.append({
                "table": table_path,
                "missing_indices": missing,
                "missing_names": [header[i] for i in missing],
            })
        count += 1
        if max_tables is not None and count >= max_tables:
            break

    if report_path is not None:
        report_path.write_text("\n".join(json.dumps(r) for r in report))

    return report


def repair_embeddings(
    adapter: BaseAdapter,
    embeddings_path: Path,
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    max_rows: Optional[int] = None,
    chunk_size: Optional[int] = None,
    max_tables: Optional[int] = None,
    dry_run: bool = False,
    allow_failures: bool = False,
) -> Dict[str, Any]:
    data, fmt = load_embeddings(embeddings_path)
    failures: Dict[str, List[int]] = {}
    repaired = 0
    scanned = 0
    effective_chunk_size = _resolve_chunk_size(adapter, chunk_size)

    for key, record in iter_records(data, fmt):
        table_path = adapter.get_table_path(record)
        if table_path is None:
            continue
        header = read_csv_header(table_path)
        missing = adapter.detect_missing(record, header)
        if missing:
            col_embeddings = adapter.get_column_embeddings(record)
            _embed_with_fallback(
                adapter,
                table_path,
                header,
                missing,
                col_embeddings,
                max_rows,
                effective_chunk_size,
                failures,
            )
            if not dry_run:
                record = adapter.update_record(record, header, col_embeddings)
                update_record(data, fmt, key, record)
            repaired += 1
        scanned += 1
        if max_tables is not None and scanned >= max_tables:
            break

    if report_path is not None:
        report_path.write_text(json.dumps(failures, indent=2))

    if not dry_run:
        save_path = output_path or embeddings_path
        save_embeddings(save_path, data, fmt)

    if failures and not allow_failures:
        report_hint = f" See report at {report_path}." if report_path else ""
        raise RuntimeError(
            f"Repair failed for {len(failures)} tables with missing columns.{report_hint}"
        )

    return {
        "scanned": scanned,
        "repaired_tables": repaired,
        "failures": failures,
    }


def _resolve_chunk_size(adapter: BaseAdapter, chunk_size: Optional[int]) -> int:
    if chunk_size is not None:
        try:
            return max(1, int(chunk_size))
        except (TypeError, ValueError):
            return 64
    recommended = adapter.recommended_chunk_size()
    if recommended is None:
        return 64
    try:
        return max(1, int(recommended))
    except (TypeError, ValueError):
        return 64
