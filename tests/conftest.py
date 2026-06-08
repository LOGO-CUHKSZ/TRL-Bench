"""Pytest configuration for the TRL-Bench test suite.

Two release-hygiene behaviours so a fresh ``git clone`` + ``pytest`` is fast
and green:

1. **Offline by default.** Importing ``trl_bench.registry`` pulls in the model
   stack, and a stray HuggingFace hub round-trip on import was adding a
   ~10-minute connect-timeout to the first-collected test (observed as a 600s
   "setup" in ``--durations``). For the default (fast) suite we force
   HF/transformers into offline mode *before* any test module is imported, so
   such a call fails instantly instead of hanging. Cached assets still work;
   only network fetches are short-circuited. ``--runslow`` leaves the network
   on so the integration smoke tests can fetch/cache models. The vars use
   ``setdefault`` so an explicit override in the environment always wins.

2. **Slow tests opt-in.** ``@pytest.mark.slow`` tests (model loads, GPU,
   reference integration) are skipped unless ``--runslow`` is passed, so a
   bare ``pytest`` runs the fast unit suite rather than pulling in the
   30-minute integration smokes.
"""
from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (model loads, GPU, reference integration)",
    )


def pytest_configure(config):
    # Runs before collection imports any test module, so this is in time to
    # neutralise an import-time hub call in the fast suite.
    if not config.getoption("--runslow"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
