"""CSV I/O helpers for embedding repair."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd

_CSV_FIELD_LIMIT_SET = False


def _ensure_csv_field_limit(limit: int = 100_000_000) -> None:
    """Raise the CSV field size limit to avoid truncation errors."""
    global _CSV_FIELD_LIMIT_SET
    if _CSV_FIELD_LIMIT_SET:
        return
    try:
        csv.field_size_limit(limit)
    except (OverflowError, ValueError):
        csv.field_size_limit(10_000_000)
    _CSV_FIELD_LIMIT_SET = True


def read_csv_header(csv_path: str) -> List[str]:
    """Read header only (no rows) to get column names."""
    _ensure_csv_field_limit()
    df = pd.read_csv(
        csv_path,
        nrows=0,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    return [str(c) for c in df.columns.tolist()]


def read_csv_subset(
    csv_path: str,
    col_indices: List[int],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Read a subset of columns by index with dtype=str for safety."""
    _ensure_csv_field_limit()
    if not col_indices:
        return pd.DataFrame()
    usecols = sorted(set(int(i) for i in col_indices))
    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        nrows=max_rows,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    # Preserve column order as in original header
    df = df[[df.columns[i] for i in range(len(df.columns))]]
    return df
