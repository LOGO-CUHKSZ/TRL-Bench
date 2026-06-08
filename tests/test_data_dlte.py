from unittest import mock
from trl_bench.data import dlte


def test_load_lake_config():
    with mock.patch("trl_bench.data.dlte.load_dataset") as m:
        dlte.load("lake")
    m.assert_called_once_with(
        "logo-lab/trl-dlte",
        name="lake",
        split=None,
        revision=None,
    )


def test_load_manifests_config():
    with mock.patch("trl_bench.data.dlte.load_dataset") as m:
        dlte.load("manifests", split="train")
    m.assert_called_once_with(
        "logo-lab/trl-dlte",
        name="manifests",
        split="train",
        revision=None,
    )
