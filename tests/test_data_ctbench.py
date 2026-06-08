"""Tests for CTbench loader. Uses mock HF dataset to avoid network in CI."""
from unittest import mock
import pytest

from trl_bench.data import ctbench


def test_load_ctbench_calls_hf_with_correct_config():
    fake_ds = {"train": object()}
    with mock.patch("trl_bench.data.ctbench.load_dataset", return_value=fake_ds) as m:
        result = ctbench.load("column_clustering", split="train")

    m.assert_called_once_with(
        "logo-lab/trl-ctbench",
        name="column_clustering",
        split="train",
        revision=None,
    )
    assert result is fake_ds["train"] or result is fake_ds


def test_load_ctbench_with_revision_pin():
    with mock.patch("trl_bench.data.ctbench.load_dataset") as m:
        ctbench.load("join_search", split="test", revision="abc123")
    m.assert_called_once_with(
        "logo-lab/trl-ctbench",
        name="join_search",
        split="test",
        revision="abc123",
    )
