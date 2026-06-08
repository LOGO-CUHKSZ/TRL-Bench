"""Tests for data loader task-name registry."""
import pytest
from trl_bench.data import get_dataset_spec, list_tasks


def test_list_tasks_includes_main_paper_tasks():
    tasks = list_tasks()
    # CTbench tasks
    assert "column_clustering" in tasks
    assert "join_search" in tasks
    assert "schema_matching" in tasks
    # RBench tasks
    assert "row_prediction" in tasks
    assert "record_linkage" in tasks
    # DLTE tasks
    assert "dlte_retrieval" in tasks
    assert "dlte_alignment" in tasks
    assert "dlte_merge" in tasks


def test_get_dataset_spec_returns_suite_and_config():
    spec = get_dataset_spec("column_clustering")
    assert spec.suite == "ctbench"
    assert spec.hf_repo == "logo-lab/trl-ctbench"

    spec = get_dataset_spec("record_linkage")
    assert spec.suite == "rbench"
    assert spec.hf_repo == "logo-lab/trl-rbench"

    spec = get_dataset_spec("dlte_retrieval")
    assert spec.suite == "dlte"
    assert spec.hf_repo == "logo-lab/trl-dlte"


def test_get_dataset_spec_unknown_task_raises():
    with pytest.raises(KeyError, match="unknown task"):
        get_dataset_spec("not_a_real_task")
