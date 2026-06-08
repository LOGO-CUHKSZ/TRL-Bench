"""End-to-end smoke test: BERT on join_classification.

This test is marked `slow` because it actually invokes the BERT wrapper
(downloading the model from HF on first call) and a probe head. Run with:

    pytest tests/smoke_test.py -v -m slow

Preconditions:
    - `pip install -e .[bert]` has been run.
    - The HF datasets `logo-lab/trl-{ctbench,rbench,dlte}` are accessible
      (huggingface-cli login if private).
    - `python -m trl_bench.data.stage` has materialized data for the
      join_classification + spider_join cell, OR run.py wires in
      stage_for_run before dispatching (deferred to Task 25 integration).

This test is part of Task 20 (smoke harness) and is exercised end-to-end
by Task 25 (verify on clean install).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_smoke_bert_join_classification(tmp_path):
    """One-cell reproduction run that lands a results JSON on disk."""
    repo_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable, "-m", "trl_bench.run",
        "--model", "bert",
        "--task", "join_classification",
        "--dataset", "spider_join",
        "--setting", "cls_embedding",
        "--probe", "linear",
        "--seed", "42",
        "--embeddings-dir", str(tmp_path / "emb"),
        "--results-dir",    str(tmp_path / "res"),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"smoke test failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    expected = (tmp_path / "res" / "evaluation" / "join_classification" /
                "bert" / "cls_embedding" / "linear" /
                "bert_spider_join_seed42.json")
    assert expected.exists(), f"expected output JSON not found at {expected}"
    data = json.loads(expected.read_text())
    assert "test_results" in data or "test_results_accuracy" in data or \
           any(k.startswith("test_") for k in data), \
           f"results JSON missing expected metric keys: {list(data.keys())}"
