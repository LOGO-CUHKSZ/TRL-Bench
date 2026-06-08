"""CTbench (column- and table-level) data loader.

Wraps `datasets.load_dataset("logo-lab/trl-ctbench", name=<task>, ...)`.
"""
from __future__ import annotations
from datasets import load_dataset

_REPO = "logo-lab/trl-ctbench"


def load(task: str, *, split: str = "train", revision: str | None = None):
    """Load a CTbench task split from the HF dataset.

    Args:
        task: One of the 13 CTbench task names (e.g., "column_clustering").
        split: "train" | "validation" | "test".
        revision: Optional HF dataset revision (commit SHA or tag) for
            reproducible pinning. None means latest.
    """
    ds = load_dataset(_REPO, name=task, split=split, revision=revision)
    return ds
