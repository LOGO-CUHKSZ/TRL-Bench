"""Compatibility helpers for loading pickle files across NumPy versions."""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from typing import BinaryIO


def install_numpy_pickle_compat() -> None:
    """Alias NumPy 2.x private pickle module paths to NumPy 1.x equivalents.

    Some embedding pickles were produced in an environment that serialized array
    objects from ``numpy._core.*``. Older NumPy environments expose those
    modules under ``numpy.core.*`` instead. Installing these aliases before
    ``pickle.load`` allows both formats to load cleanly.
    """

    try:
        np_core = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", np_core)

    for name in ("numeric", "multiarray", "umath", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pickle(path_or_file: str | Path | BinaryIO):
    """Load a pickle after installing NumPy compatibility aliases."""
    install_numpy_pickle_compat()

    if hasattr(path_or_file, "read"):
        return pickle.load(path_or_file)

    with open(path_or_file, "rb") as f:
        return pickle.load(f)
