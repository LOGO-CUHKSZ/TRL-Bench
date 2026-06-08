from unittest import mock
from trl_bench.data import rbench


def test_load_row_prediction():
    with mock.patch("trl_bench.data.rbench.load_dataset") as m:
        rbench.load("row_prediction", split="train", openml_id="40945")
    m.assert_called_once_with(
        "logo-lab/trl-rbench",
        name="row_prediction:40945",
        split="train",
        revision=None,
    )


def test_load_record_linkage_subconfig():
    with mock.patch("trl_bench.data.rbench.load_dataset") as m:
        rbench.load("record_linkage", split="test",
                    rl_dataset="deepmatcher_abt_buy")
    m.assert_called_once_with(
        "logo-lab/trl-rbench",
        name="record_linkage:deepmatcher_abt_buy",
        split="test",
        revision=None,
    )
