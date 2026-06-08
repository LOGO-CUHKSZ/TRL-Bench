"""Tests that the setting→embedding_type translation in the registry is
correct for every aggregation the paper uses on probe tasks.

The fast tests assert on the constructed command-line args.
"""
from __future__ import annotations

import pytest

from trl_bench.registry import build_command


_SETTINGS = ("cls_embedding", "column_mean", "token_mean")


def test_setting_translation_for_join_classification(tmp_path):
    """The registry maps setting -> --embedding_type as the paper uses."""
    expected = {"cls_embedding": "cls",
                "column_mean":   "column_mean",
                "token_mean":    "token_mean"}
    for setting, embed_type in expected.items():
        stages = build_command(
            model="bert", task="join_classification",
            dataset="spider_join", setting=setting,
            probe="linear", seed=42,
            results_dir=tmp_path,
            embeddings_path=tmp_path / "e.pkl",
            labels_path=tmp_path / "l.json",
            configs_root=tmp_path,
        )
        cmd = stages[0]
        args = dict(zip(cmd[3:][::2], cmd[3:][1::2]))
        assert args["--embedding_type"] == embed_type, \
            f"setting {setting!r} -> --embedding_type {args['--embedding_type']!r} (want {embed_type!r})"
