"""Data loader registry mapping task names to HF dataset specs."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    suite: str       # "ctbench" | "rbench" | "dlte"
    hf_repo: str     # e.g., "logo-lab/trl-ctbench"
    hf_config: str | None = None   # HF dataset config name (often = task name)


_TASKS: dict[str, DatasetSpec] = {
    # CTbench (column + table level)
    "column_type_prediction":    DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "column_clustering":         DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "column_relation_prediction": DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "join_search":               DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "schema_matching":           DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "union_search":              DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "table_subset":              DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "table_retrieval":           DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "semantic_parsing":          DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "join_classification":       DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "union_classification":      DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "union_regression":          DatasetSpec("ctbench", "logo-lab/trl-ctbench"),
    "join_containment":          DatasetSpec("ctbench", "logo-lab/trl-ctbench"),  # paper alias: "Column Overlap"

    # RBench
    "row_prediction":            DatasetSpec("rbench", "logo-lab/trl-rbench"),
    "record_linkage":            DatasetSpec("rbench", "logo-lab/trl-rbench"),

    # DLTE
    "dlte_retrieval":            DatasetSpec("dlte", "logo-lab/trl-dlte"),
    "dlte_alignment":            DatasetSpec("dlte", "logo-lab/trl-dlte"),
    "dlte_merge":                DatasetSpec("dlte", "logo-lab/trl-dlte"),
}


def list_tasks() -> list[str]:
    """Return the canonical task name list."""
    return sorted(_TASKS.keys())


def get_dataset_spec(task: str) -> DatasetSpec:
    """Return the dataset spec for a task; raise KeyError if unknown."""
    if task not in _TASKS:
        raise KeyError(f"unknown task: {task!r}. Use trl_bench.data.list_tasks() to see all.")
    return _TASKS[task]


__all__ = ["DatasetSpec", "list_tasks", "get_dataset_spec"]
