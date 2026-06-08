"""RBench (row-level) data loader.

RBench has two task families:
  - row_prediction: 50 OpenML-derived tables, sub-configured by OpenML id.
  - record_linkage: 16 entity-matching tasks, sub-configured by RL dataset name.
"""
from __future__ import annotations
from datasets import load_dataset

_REPO = "logo-lab/trl-rbench"


def load(task: str, *, split: str = "train", revision: str | None = None,
         openml_id: str | None = None, rl_dataset: str | None = None):
    """Load an RBench task split.

    Args:
        task: "row_prediction" or "record_linkage".
        split: "train" | "validation" | "test".
        revision: Optional HF revision pin.
        openml_id: Required when task == "row_prediction". OpenML table id (e.g., "40945").
        rl_dataset: Required when task == "record_linkage". Sub-config name
            (e.g., "deepmatcher_abt_buy", "wdc_products_small").
    """
    if task == "row_prediction":
        if openml_id is None:
            raise ValueError("row_prediction requires openml_id")
        name = f"row_prediction:{openml_id}"
    elif task == "record_linkage":
        if rl_dataset is None:
            raise ValueError("record_linkage requires rl_dataset")
        name = f"record_linkage:{rl_dataset}"
    else:
        raise ValueError(f"unknown rbench task: {task!r}")

    return load_dataset(_REPO, name=name, split=split, revision=revision)
