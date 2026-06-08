"""DLTE (Data-Lake Table Enrichment) data loader.

The HF dataset has two configs: "lake" (47,772 tables) and "manifests"
(1,379 parents + 16,548 fragments + lake_manifest).
"""
from __future__ import annotations
from datasets import load_dataset

_REPO = "logo-lab/trl-dlte"
_VALID_CONFIGS = {"lake", "manifests"}


def load(config: str, *, split: str | None = None, revision: str | None = None):
    """Load a DLTE config.

    Args:
        config: "lake" or "manifests".
        split: Optional split. Some configs are single-table.
        revision: Optional HF revision pin.
    """
    if config not in _VALID_CONFIGS:
        raise ValueError(f"unknown dlte config: {config!r}. Valid: {_VALID_CONFIGS}.")
    return load_dataset(_REPO, name=config, split=split, revision=revision)
