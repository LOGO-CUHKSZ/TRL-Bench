"""Guard tests for the SwitchTab restoration.

SwitchTab is a self-supervised *row* model that is part of the paper's
model set but was purged from the codebase (commit 8c7791a). These tests
pin down the contract the restoration must satisfy: the ts3l backend
imports, the registry granularity entry, and the row-config YAML blocks.

All assertions here are import-/config-level only (no model construction,
no training), so they are safe to run on the login node and in the fast
(offline) suite.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# == ts3l backend imports ====================================================

def test_switchtab_model_imports():
    from trl_bench.utils.ts3l.models import SwitchTab  # noqa: F401


def test_switchtab_lightning_imports():
    from trl_bench.utils.ts3l.pl_modules import SwitchTabLightning  # noqa: F401


def test_switchtab_functional_module_imports():
    # The other SSL models expose their functional helper as a *module*
    # re-exported from ``functional`` (e.g. ``from ...functional import scarf``).
    # Mirror that form for switchtab.
    from trl_bench.utils.ts3l.functional import switchtab  # noqa: F401


# == registry granularity ====================================================

def test_switchtab_granularity_is_row_only():
    from trl_bench.registry import _MODEL_GRANULARITIES

    assert _MODEL_GRANULARITIES["switchtab"] == frozenset({"row"})


# == row-config YAMLs ========================================================

def test_switchtab_in_row_data_models_yaml():
    cfg = yaml.safe_load(
        (REPO_ROOT / "slurm" / "config" / "row_data_models.yaml").read_text()
    )
    assert "switchtab" in cfg["models"]


def test_switchtab_in_row_models_yaml():
    cfg = yaml.safe_load(
        (REPO_ROOT / "slurm" / "config" / "row_models.yaml").read_text()
    )
    assert "switchtab" in cfg["models"]
