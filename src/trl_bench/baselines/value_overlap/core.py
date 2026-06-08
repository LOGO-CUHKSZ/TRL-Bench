"""Shared utilities for value-overlap baselines.

Provides cell-value normalization, column-name normalization, CSV column
extraction, and containment scoring.  Used by both join_search.py and
union_search.py.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

_NULL_VALUES = frozenset({'', 'nan', 'null', 'n/a', 'none', '-', 'na', '#n/a', '#ref!'})


def normalize_value(val: str) -> Optional[str]:
    """Normalize a cell value for set comparison.

    Returns None for empty / null-like values (excluded from sets).
    """
    v = val.strip().lower()
    if v in _NULL_VALUES:
        return None
    return v


def normalize_column_name(name: str) -> str:
    """Strip newlines from column names.

    Matches the convention in run_search_and_evaluate.py:70-74.
    """
    return name.replace('\n', '').replace('\r', '')


def extract_column_value_sets(csv_path: Path | str) -> dict[str, set[str]]:
    """Read a CSV and return {column_name: set_of_normalized_values}.

    - Strips ``\\n`` and ``\\r`` from column names (matching the embedding
      pipeline in ``convert_to_tuples``).
    - Disambiguates duplicate column names with ``_<index>`` suffix (matching
      ``run_search_and_evaluate.py:106-109``).
    - Uses ``csv.reader`` for robustness with messy OpenData CSVs.
    - Returns an empty dict if the file cannot be read.
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            raw_headers = next(reader, None)
            if raw_headers is None:
                return {}

            # Normalize and disambiguate headers
            headers: list[str] = []
            seen: set[str] = set()
            for idx, h in enumerate(raw_headers):
                name = normalize_column_name(h)
                if name in seen:
                    name = f"{name}_{idx}"
                seen.add(name)
                headers.append(name)

            # Collect values per column
            col_values: dict[str, set[str]] = {h: set() for h in headers}
            for row in reader:
                for i, cell in enumerate(row):
                    if i >= len(headers):
                        break
                    nv = normalize_value(cell)
                    if nv is not None:
                        col_values[headers[i]].add(nv)

            return col_values

    except Exception:
        return {}


def containment(query_set: set, candidate_set: set) -> float:
    """|Q ∩ C| / |Q|.  Returns 0.0 if Q is empty."""
    if not query_set:
        return 0.0
    return len(query_set & candidate_set) / len(query_set)
