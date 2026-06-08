"""Shared utility for table-list-based sharding.

Provides helpers to load a table list file (one CSV basename per line)
and filter a list of discovered CSV files/paths against it.  Reusable
by both column and (future) row embedding scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_table_list(path: str) -> set[str]:
    """Read a table list file and return a set of CSV basenames.

    Each line should be a bare filename, e.g. ``table_001.csv``.
    Blank lines and ``#``-prefixed comments are ignored.
    """
    names: set[str] = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                names.add(line)
    return names


def filter_csv_files(csv_files, table_list: set | None):
    """Filter *csv_files* to only those whose basename is in *table_list*.

    Works with strings (filenames or full paths) and :class:`~pathlib.Path`
    objects.  When *table_list* is ``None`` the input is returned unchanged.
    """
    if table_list is None:
        return csv_files

    filtered = []
    for f in csv_files:
        basename = os.path.basename(str(f)) if not isinstance(f, Path) else f.name
        if basename in table_list:
            filtered.append(f)
    return filtered
