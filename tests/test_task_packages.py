"""Packaging guard: every task subpackage that exposes a
``python -m trl_bench.tasks.<x>.<runner>`` entry point must be a real package
(contain ``__init__.py``) so ``setuptools.find_packages`` ships it in a
non-editable wheel.

Without ``__init__.py`` these dirs import fine under ``pip install -e .``
(resolved by on-disk path) but are silently dropped from a built wheel/sdist,
so the runtime dispatch in ``registry.py``
(``python -m trl_bench.tasks.union_search.run_search``) fails with
``ModuleNotFoundError`` after a normal ``pip install``.
"""
from __future__ import annotations

import pathlib

from setuptools import find_packages

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# Task subdirs that contain runner modules dispatched via ``python -m ...``
# (see the runner strings in registry.py). ``table_subset`` is intentionally
# excluded: it contains no python modules and is not dispatched as a runner.
_REQUIRED_TASK_PACKAGES = [
    "union_search",
    "join_search",
    "column_clustering",
    "column_type_prediction",
    "column_relation_prediction",
    "row_prediction",
    "schema_matching",
]


def test_task_runner_packages_discoverable_by_find_packages():
    discovered = set(find_packages(where=str(_SRC)))
    missing = [
        name
        for name in _REQUIRED_TASK_PACKAGES
        if f"trl_bench.tasks.{name}" not in discovered
    ]
    assert not missing, (
        "task packages missing __init__.py (dropped from non-editable wheel): "
        f"{missing}"
    )
