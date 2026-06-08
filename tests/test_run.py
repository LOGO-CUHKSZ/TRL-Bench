"""Tests for the run.py entry point.

The end-to-end probe execution is exercised in ``tests/integration/`` which
runs against a real embedding pickle. The tests in this module focus on the
CLI surface, the stage-loop short-circuit semantics, and the envelope wrap.
"""
from __future__ import annotations
import sys
import json
from unittest import mock
import pytest

from trl_bench import run as run_mod


_BASE_ARGS = [
    "--model", "bert",
    "--task",  "join_classification",
    "--dataset", "spider_join",
    "--setting", "cls_embedding",
    "--probe",   "linear",
    "--seed",    "42",
]


def test_main_invokes_each_stage_in_order(tmp_path):
    fake_stages = [["python", "a"], ["python", "b"]]
    args = _BASE_ARGS + [
        "--embeddings-path", str(tmp_path / "emb.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path / "results"),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=fake_stages), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0
    assert sp.call_count == 2
    assert sp.call_args_list[0].args[0] == fake_stages[0]
    assert sp.call_args_list[1].args[0] == fake_stages[1]


def test_main_returns_nonzero_and_short_circuits_on_stage_failure(tmp_path):
    fake_stages = [["python", "a"], ["python", "b"]]
    args = _BASE_ARGS + [
        "--embeddings-path", str(tmp_path / "emb.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path / "results"),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=fake_stages), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=7)
        rc = run_mod.main(args)
    assert rc == 7
    assert sp.call_count == 1   # second stage skipped


def test_main_rejects_invalid_cell(tmp_path):
    # saint exports only row embeddings; table_retrieval requires table.
    args = [
        "--model", "saint", "--task", "table_retrieval",
        "--dataset", "tus", "--setting", "cls_embedding",
        "--probe", "linear", "--seed", "42",
        "--embeddings-path", str(tmp_path / "x.pkl"),
        "--labels-path",     str(tmp_path / "l.json"),
        "--results-dir",     str(tmp_path),
    ]
    rc = run_mod.main(args)
    assert rc == 2


def test_envelope_wrap_flattens_test_results(tmp_path):
    # Simulate run_task.py's output: nested test_results dict.
    raw_dir = (tmp_path / "evaluation/join_classification/bert/cls_embedding"
               / "seed42/linear")
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("results.json").write_text(json.dumps({
        "task_name": "join_spider_join",
        "task_type": "classification",
        "head_type": "linear",
        "seed": 42,
        "embedding_type": "cls",
        "test_results": {
            "accuracy": 0.7727,
            "weighted_f1": 0.7756,
            "loss": None,
        },
        "data_stats": {"train": 5146, "valid": 742, "test": 1474},
    }))

    args = _BASE_ARGS + [
        "--embeddings-path", str(tmp_path / "emb.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0

    envelope_path = (tmp_path / "evaluation/join_classification/bert/cls_embedding"
                     / "linear/bert_spider_join_seed42.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    # Reference flat schema:
    assert env["model"]    == "bert"
    assert env["dataset"]  == "spider_join"
    assert env["task"]     == "join_classification"
    assert env["seed"]     == 42
    assert env["head_type"] == "linear"
    assert env["test_results_accuracy"]    == 0.7727
    assert env["test_results_weighted_f1"] == 0.7756
    assert env["test_results_loss"]        is None
    assert env["data_stats"]["train"] == 5146
    assert env["status"] == "completed"
    assert env["hyperparameters"]["task_type"]    == "classification"
    assert env["hyperparameters"]["learning_rate"] == 0.001


def test_row_prediction_envelope_preserves_train_results_block(tmp_path, monkeypatch):
    """row_prediction envelope is STRUCTURED (not flat). The reference
    envelope preserves test_results / training / data_stats / train_results /
    scaled_test_results / target_zscore / target_scaler blocks intact, plus
    top-level task / task_type / head_type / seed / model / dataset /
    variant / label_column from the runner. The wrapper adds slurm_job_id +
    status (and skips ``hyperparameters`` since YAML drives the runner).
    """
    # Tests run under slurm (per the login-node "slurm-first" policy) inherit
    # SLURM_JOB_ID; clear it so the envelope exercises the deterministic
    # "local" fallback regardless of where the suite runs.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    # Simulate train_downstream.py's output (regression with z-score scaling).
    raw_dir = (tmp_path / "evaluation/row_prediction/gte/openml_44992"
               / "seed42/FPS")
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("results.json").write_text(json.dumps({
        "task_name": "row_prediction_FPS",
        "task": "row_prediction",
        "task_type": "regression",
        "head_type": "mlp",
        "seed": 42,
        "model": "gte",
        "dataset": "openml_44992",
        "variant": None,
        "label_column": "FPS",
        "test_results": {"loss": 24.13, "mse": 24.13, "r2": 0.9917,
                         "mae": 3.51, "pearson_r": 0.996, "spearman_r": 0.996,
                         "rmse": 4.91},
        "training": {"best_epoch": 79, "best_value": 0.0076,
                     "total_epochs": 94},
        "data_stats": {"train": 19699, "test": 2463, "input_dim": 768,
                       "n_classes": None, "val": 2462},
        "train_results": {"loss": 13.99, "r2": 0.9953, "mae": 2.74,
                          "rmse": 3.74, "mse": 13.99,
                          "pearson_r": 0.998, "spearman_r": 0.998},
        "scaled_test_results": {"loss": 0.008, "r2": 0.9917, "mse": 0.008,
                                "mae": 0.064, "rmse": 0.089,
                                "pearson_r": 0.996, "spearman_r": 0.996},
        "target_zscore": True,
        "target_zscore_split_mode": "canonical_val",
        "target_scaler": {"mean": 100.0, "scale": 54.9},
    }))

    args = [
        "--model", "gte", "--task", "row_prediction",
        "--dataset", "openml_44992", "--setting", "FPS",
        "--probe", "mlp", "--seed", "42",
        "--embeddings-path", str(tmp_path / "row_embs"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0

    envelope_path = (tmp_path / "evaluation/row_prediction/gte/openml_44992"
                     / "FPS/gte_openml_44992_seed42.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())

    # Top-level pass-through from runner.
    assert env["task"]         == "row_prediction"
    assert env["task_type"]    == "regression"
    assert env["head_type"]    == "mlp"
    assert env["seed"]         == 42
    assert env["model"]        == "gte"
    assert env["dataset"]      == "openml_44992"
    assert env["label_column"] == "FPS"
    assert env["task_name"]    == "row_prediction_FPS"

    # Structured metric blocks preserved intact (no flattening).
    assert isinstance(env["test_results"], dict)
    assert env["test_results"]["r2"] == 0.9917
    assert env["training"]["best_epoch"] == 79
    assert env["data_stats"]["train"] == 19699
    assert env["train_results"]["r2"] == 0.9953
    assert env["scaled_test_results"]["r2"] == 0.9917
    assert env["target_zscore"] is True
    assert env["target_zscore_split_mode"] == "canonical_val"
    assert env["target_scaler"]["mean"] == 100.0

    # Release-only bookkeeping.
    assert env["slurm_job_id"] == "local"
    assert env["status"] == "completed"

    # No `hyperparameters` block — runner is YAML-driven.
    assert "hyperparameters" not in env


def test_semantic_parsing_envelope_parses_log_correctly(tmp_path):
    """semantic_parsing envelope reads from ``test.log`` (JSON) — not
    ``results.json`` — and writes flat keys ``accuracy`` / ``oracle_accuracy``
    plus a ``hyperparameters={seed, beam_size}`` block. The envelope sits
    one level above ``seed<S>/`` and has NO ``head_type`` axis (semantic
    parsing has no probe head).
    """
    # Simulate run_test.py's output (test.log is a JSON dump).
    raw_dir = (tmp_path / "evaluation/semantic_parsing/bert/mpnet/seed72")
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("test.log").write_text(json.dumps({
        "accuracy":        0.24286372007366483,
        "oracle_accuracy": 0.3812154696132597,
    }))

    args = [
        "--model", "bert", "--task", "semantic_parsing",
        "--dataset", "wiki_table_questions", "--setting", "mpnet",
        # semantic_parsing has no probe; pass a sentinel because argparse
        # may require it. The wrapper does not consume head_type for semparse.
        "--seed", "72",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0

    envelope_path = (tmp_path / "evaluation/semantic_parsing/bert/mpnet"
                     / "bert_wiki_table_questions_seed72.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())

    # Reference flat schema:
    assert env["model"]    == "bert"
    assert env["dataset"]  == "wiki_table_questions"
    assert env["task"]     == "semantic_parsing"
    assert env["seed"]     == 72
    assert env["accuracy"]        == 0.24286372007366483
    assert env["oracle_accuracy"] == 0.3812154696132597
    assert env["status"] == "completed"

    # hyperparameters block: {seed, beam_size}.
    assert env["hyperparameters"]["seed"]      == 72
    assert env["hyperparameters"]["beam_size"] == 5

    # No `head_type` axis — semantic_parsing has no probe head.
    assert "head_type" not in env


# == DLTE: envelope coverage =================================================


def test_dlte_retrieval_envelope_collates_topk_siblings(tmp_path):
    """dlte_retrieval envelope groups sibling ``metrics_test_topk_<K>.json``
    files under ``metrics.test_topk_<K>`` matching the reference shape.
    """
    raw_dir = tmp_path / "evaluation" / "dlte" / "stage1" / "bert"
    raw_dir.mkdir(parents=True)
    # Write three K-files; the envelope picks topk_100 as raw via the run.py
    # filename convention and then discovers the siblings to flesh out
    # ``metrics``.
    for k in (10, 50, 100):
        (raw_dir / f"metrics_test_topk_{k}.json").write_text(json.dumps({
            "k": k, "n_queries": 1380, "recall_any": 0.5 + k / 1000,
            "recall_union": 0.7, "recall_join": 0.3,
        }))
    args = [
        "--model", "bert", "--task", "dlte_retrieval",
        "--dataset", "dlte_v1", "--setting", "diagonal",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0

    envelope_path = (tmp_path / "evaluation" / "dlte_retrieval" / "bert"
                     / "diagonal" / "bert_dlte_v1_seed42.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["task"]    == "dlte_retrieval"
    assert env["model"]   == "bert"
    assert env["dataset"] == "dlte_v1"
    assert env["table_model"] == "bert"
    assert env["col_model"]   == "bert"
    assert env["status"] == "completed"
    # All three K-blocks present.
    assert set(env["metrics"]) == {"test_topk_10", "test_topk_50", "test_topk_100"}
    assert env["metrics"]["test_topk_100"]["k"] == 100
    assert env["metrics"]["test_topk_100"]["recall_any"] == 0.6


def test_dlte_alignment_envelope_includes_calibration_block(tmp_path):
    """dlte_alignment envelope pulls in the ``calibration_dev.json`` sibling
    (when present) under ``calibration`` and the raw metrics under
    ``metrics.test`` (mirrors the round5 stage2.json reference shape).
    """
    raw_dir = tmp_path / "evaluation" / "dlte" / "stage2" / "bert"
    raw_dir.mkdir(parents=True)
    (raw_dir / "metrics_test_topk_100.json").write_text(json.dumps({
        "split": "test", "n_queries": 1380, "relation_acc": 0.93,
        "key_col_acc": 1.0, "col_align_f1_union": 0.997,
    }))
    (raw_dir / "calibration_dev.json").write_text(json.dumps({
        "best_thresholds": {"tau_union": 0.9, "tau_join_max": 0.4},
        "best_macro_f1": 0.4014,
    }))
    args = [
        "--model", "bert", "--task", "dlte_alignment",
        "--dataset", "dlte_v1", "--setting", "diagonal",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0
    envelope_path = (tmp_path / "evaluation" / "dlte_alignment" / "bert"
                     / "diagonal" / "bert_dlte_v1_seed42.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["col_model"] == "bert"
    assert env["task"] == "dlte_alignment"
    assert env["calibration"]["best_macro_f1"] == 0.4014
    assert env["metrics"]["test"]["relation_acc"] == 0.93
    assert env["status"] == "completed"


# == Deterministic training-free tasks: envelope coverage ====================
# These tasks have no probe head, no seed dimension, and the reference
# layout is flat — ``<task>/<model>/<model>_<dataset>.json``. Three of the
# four runners only emit metrics to stdout; ``run.py`` tees stdout into
# ``stage_run.log`` before the envelope wrapper regex-extracts the metrics.


def _stdout_writing_subprocess_mock(payload: bytes):
    """Build a subprocess.run mock that writes ``payload`` to the ``stdout=``
    file descriptor before returning rc=0.

    This mirrors what the real runner does when ``run.py`` tees its stdout
    into ``stage_run.log``: the test's synthetic payload replaces the runner's
    real banner output.
    """
    def _side_effect(cmd, **kwargs):
        fd = kwargs.get("stdout")
        if fd is not None and hasattr(fd, "write"):
            fd.write(payload)
            fd.flush()
        return mock.Mock(returncode=0)
    return _side_effect


def test_column_clustering_envelope_parses_stdout_banner(tmp_path):
    """column_clustering envelope reads from ``stage_run.log`` (captured
    stdout) and regex-extracts purity / num_clusters / NMI / ARI / coverage
    plus the {k, target_avg_size} hyperparameter block.
    """
    synthetic_stdout = (
        b"  Number of clusters:    2,391\n"
        b"  Avg cluster size:      50.44\n"
        b"  Purity:                0.7917 (79.17%)\n"
        b"  NMI:                   0.5417\n"
        b"  ARI:                   0.0426\n"
        b"  Total columns:         120,605\n"
        b"  Coverage:              120605/120609 (0.999967)\n"
        b"  Missing tables:        4\n"
        b"  Column mismatches:     0\n"
    )
    args = [
        "--model", "bert", "--task", "column_clustering",
        "--dataset", "sato", "--setting", "column_mean",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command",
                    return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run",
                    side_effect=_stdout_writing_subprocess_mock(synthetic_stdout)):
        rc = run_mod.main(args)
    assert rc == 0
    envelope_path = (tmp_path / "evaluation" / "column_clustering" / "bert"
                     / "bert_sato.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    # Top-level metadata mirrors reference.
    assert env["model"]   == "bert"
    assert env["dataset"] == "sato"
    assert env["task"]    == "column_clustering"
    # hyperparameters block from registry.
    assert env["hyperparameters"]["k"]               == 20
    assert env["hyperparameters"]["target_avg_size"] == 50
    # Parsed metrics — verify against the synthetic stdout exactly.
    assert env["purity"]           == 0.7917
    assert env["num_clusters"]     == 2391
    assert env["avg_cluster_size"] == 50.44
    assert env["nmi"]              == 0.5417
    assert env["ari"]              == 0.0426
    assert env["total_columns"]    == 120605
    assert env["matched_columns"]  == 120605
    assert env["expected_columns"] == 120609
    assert env["missing_tables_count"] == 4
    assert env["col_mismatches"]       == 0
    assert env["status"] == "completed"


def test_schema_matching_envelope_reads_native_results_json(tmp_path):
    """schema_matching envelope reads the runner's native
    ``<output_dir>/results.json`` and projects only the flat reference keys
    (micro/macro precision/recall/f1, pairs counts, recall_at_gt, gt_coverage).
    """
    # The runner writes results.json into envelope_dir (same dir as the
    # envelope, since path_layout="deterministic"). Pre-write a synthetic one;
    # subprocess is mocked so the runner doesn't actually run.
    out_dir = tmp_path / "evaluation" / "schema_matching" / "turl"
    out_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(json.dumps({
        "model": "turl", "dataset": "valentine", "task": "schema_matching",
        "config": {"matching_strategy": "hungarian", "threshold": 0.0},
        "recall_at_gt": 0.3401, "gt_coverage": 1.0,
        "micro_precision": 0.3524, "micro_recall": 0.5254, "micro_f1": 0.4219,
        "macro_precision": 0.3356, "macro_recall": 0.5664, "macro_f1": 0.3831,
        "overall_micro": {"precision": 0.3524, "recall": 0.5254, "f1": 0.4219},
        "overall_macro": {"precision": 0.3356, "recall": 0.5664, "f1": 0.3831},
        "n_pairs": 550, "n_skipped": 0,
        "n_gt_covered": 1700, "n_gt_total": 1700,
        "per_source": {}, "per_noise_type": {},
    }))

    args = [
        "--model", "turl", "--task", "schema_matching",
        "--dataset", "valentine", "--setting", "column_mean",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command",
                    return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0
    envelope_path = out_dir / "turl_valentine.json"
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["model"]   == "turl"
    assert env["dataset"] == "valentine"
    assert env["task"]    == "schema_matching"
    # hyperparameters mirror the reference.
    assert env["hyperparameters"]["matching_strategy"] == "hungarian"
    assert env["hyperparameters"]["threshold"]         == 0.0
    # Flat metric keys preserved verbatim.
    assert env["micro_precision"] == 0.3524
    assert env["micro_recall"]    == 0.5254
    assert env["micro_f1"]        == 0.4219
    assert env["macro_precision"] == 0.3356
    assert env["macro_recall"]    == 0.5664
    assert env["macro_f1"]        == 0.3831
    # Map runner's n_pairs -> reference's pairs_evaluated.
    assert env["pairs_evaluated"] == 550
    assert env["pairs_skipped"]   == 0
    assert env["recall_at_gt"]    == 0.3401
    assert env["gt_coverage"]     == 1.0
    assert env["status"] == "completed"


def test_union_search_envelope_parses_stdout_map_p_r(tmp_path):
    """union_search envelope reads ``stage_run.log`` (captured stdout) and
    extracts ``map_at_k`` / ``precision_at_k`` / ``recall_at_k`` plus the
    {method, k, threshold, ef, N} hyperparameter block.
    """
    synthetic_stdout = (
        b"  MAP@10  = 0.8908\n"
        b"  P@10    = 0.8908\n"
        b"  R@10    = 0.6241\n"
    )
    args = [
        "--model", "gte", "--task", "union_search",
        "--dataset", "santos", "--setting", "column_mean",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command",
                    return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run",
                    side_effect=_stdout_writing_subprocess_mock(synthetic_stdout)):
        rc = run_mod.main(args)
    assert rc == 0
    envelope_path = (tmp_path / "evaluation" / "union_search" / "gte"
                     / "gte_santos.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["model"]   == "gte"
    assert env["dataset"] == "santos"
    assert env["task"]    == "union_search"
    # hyperparameters block from registry.
    assert env["hyperparameters"]["method"]    == "linear"
    assert env["hyperparameters"]["k"]         == 10
    assert env["hyperparameters"]["threshold"] == 0.7
    assert env["hyperparameters"]["ef"]        == 100
    assert env["hyperparameters"]["N"]         == 100
    # Parsed metrics.
    assert env["map_at_k"]       == 0.8908
    assert env["precision_at_k"] == 0.8908
    assert env["recall_at_k"]    == 0.6241
    assert env["status"] == "completed"


def test_join_search_envelope_parses_stdout_col_metrics(tmp_path):
    """join_search envelope reads ``stage_run.log`` (captured stdout) and
    extracts column-level metrics (``col_recall_at_{10,20,50}``,
    ``col_precision_at_{10,20,50}``, ``col_f1_at_{10,20,50}``, ``col_map``)
    plus {k} hyperparameters.
    """
    synthetic_stdout = (
        b"  COL Precision@10: 0.2400\n"
        b"  COL Recall@10:    0.2284\n"
        b"  COL F1@10:        0.2073\n"
        b"  COL Precision@20: 0.1416\n"
        b"  COL Recall@20:    0.2605\n"
        b"  COL F1@20:        0.1674\n"
        b"  COL Precision@50: 0.0673\n"
        b"  COL Recall@50:    0.2986\n"
        b"  COL F1@50:        0.1044\n"
        b"  COL MAP: 0.1885\n"
    )
    args = [
        "--model", "tabbie", "--task", "join_search",
        "--dataset", "opendata", "--setting", "column_mean",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command",
                    return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run",
                    side_effect=_stdout_writing_subprocess_mock(synthetic_stdout)):
        rc = run_mod.main(args)
    assert rc == 0
    envelope_path = (tmp_path / "evaluation" / "join_search" / "tabbie"
                     / "tabbie_opendata.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["model"]   == "tabbie"
    assert env["dataset"] == "opendata"
    assert env["task"]    == "join_search"
    assert env["hyperparameters"] == {"k": 50}
    # Per-K column-level metrics.
    assert env["col_recall_at_10"]    == 0.2284
    assert env["col_precision_at_10"] == 0.2400
    assert env["col_f1_at_10"]        == 0.2073
    assert env["col_recall_at_20"]    == 0.2605
    assert env["col_precision_at_20"] == 0.1416
    assert env["col_f1_at_20"]        == 0.1674
    assert env["col_recall_at_50"]    == 0.2986
    assert env["col_precision_at_50"] == 0.0673
    assert env["col_f1_at_50"]        == 0.1044
    assert env["col_map"]             == 0.1885
    assert env["status"] == "completed"


def test_join_search_learned_envelope_parses_stdout_col_metrics(tmp_path):
    """join_search_learned writes a reference envelope just like join_search.

    The learned-projection variant (run_learned_search.py) tees the SAME COL
    metric banner to ``stage_run.log`` and reuses the ``join_search_flat``
    wrapper. Unlike join_search it is SEEDED (a trained projection head), so it
    is NOT in ``DETERMINISTIC_TASKS`` and its envelope keeps the seed suffix:

        evaluation/join_search_learned/<model>/<model>_<dataset>_seed<S>.json

    It is also a probe-less PROBE_TASK (no ``--probe`` axis), so this guards the
    envelope-writing gate against silently skipping the write when
    ``args.probe`` is None -- without it the one-command fresh-clone path runs
    the stage but produces no reference envelope.
    Hyperparameters mirror the registry block: {num_layers, batch_size,
    max_epochs, learning_rate, k}.
    """
    synthetic_stdout = (
        b"  COL Precision@10: 0.3251\n"
        b"  COL Recall@10:    0.2961\n"
        b"  COL F1@10:        0.2782\n"
        b"  COL Precision@20: 0.2684\n"
        b"  COL Recall@20:    0.4431\n"
        b"  COL F1@20:        0.3109\n"
        b"  COL Precision@50: 0.1670\n"
        b"  COL Recall@50:    0.6391\n"
        b"  COL F1@50:        0.2550\n"
        b"  COL MAP: 0.2942\n"
    )
    args = [
        "--model", "tabert", "--task", "join_search_learned",
        "--dataset", "opendata_uk_sg", "--setting", "cls_embedding",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "col.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command",
                    return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run",
                    side_effect=_stdout_writing_subprocess_mock(synthetic_stdout)):
        rc = run_mod.main(args)
    assert rc == 0
    # SEEDED task -> seed-suffixed envelope filename (not the flat join_search
    # ``<model>_<dataset>.json``).
    envelope_path = (tmp_path / "evaluation" / "join_search_learned" / "tabert"
                     / "tabert_opendata_uk_sg_seed42.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["model"]   == "tabert"
    assert env["dataset"] == "opendata_uk_sg"
    assert env["task"]    == "join_search_learned"
    assert env["hyperparameters"] == {
        "num_layers": 1, "batch_size": 512, "max_epochs": 10,
        "learning_rate": 1e-3, "k": 50,
    }
    assert env["col_map"]          == 0.2942
    assert env["col_recall_at_50"] == 0.6391
    assert env["status"] == "completed"


def test_dlte_merge_envelope_preserves_splits_block(tmp_path):
    """dlte_merge envelope passes the runner's ``end_to_end.json`` ``splits``
    block through verbatim and stamps the col/row/table model triple at the
    top level (mirroring the reference metrics/<col>__<row>__<table>/
    end_to_end.json shape).

    Path layout: ``step10_row_matching`` collapses table_m==col_m to a
    single token in the directory key (see ``derive_stage2_key``), so the
    diagonal ``bert × bert × bert`` case writes under
    ``metrics/bert__bert/`` (2-token), NOT ``metrics/bert__bert__bert/``.
    The dispatcher's path resolver in ``run.py`` mirrors this collapse rule.
    """
    raw_dir = (tmp_path / "evaluation" / "dlte" / "metrics" / "bert__bert")
    raw_dir.mkdir(parents=True)
    (raw_dir / "end_to_end.json").write_text(json.dumps({
        "col_model":   "bert",
        "row_model":   "bert",
        "table_model": "bert",
        "splits": {
            "test": {
                "n_queries": 1380, "cell_f1": 0.599,
                "cell_precision": 0.883, "cell_recall": 0.466,
                "parent_row_recall": 0.845, "parent_col_recall": 0.566,
                "region_recall": {"core_core": 0.982, "union_region": 0.637,
                                  "join_region": 0.114, "hard_region": 0.126},
                "uj_h": 0.139,
            }
        }
    }))
    args = [
        "--model", "bert", "--task", "dlte_merge",
        "--dataset", "dlte_v1", "--setting", "diagonal",
        "--seed", "42",
        "--embeddings-path", str(tmp_path / "row.pkl"),
        "--labels-path",     str(tmp_path / "labels.json"),
        "--results-dir",     str(tmp_path),
    ]
    with mock.patch("trl_bench.run.build_command", return_value=[["python", "a"]]), \
         mock.patch("trl_bench.run.subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0)
        rc = run_mod.main(args)
    assert rc == 0
    envelope_path = (tmp_path / "evaluation" / "dlte_merge" / "bert"
                     / "diagonal" / "bert_dlte_v1_seed42.json")
    assert envelope_path.exists()
    env = json.loads(envelope_path.read_text())
    assert env["col_model"]   == "bert"
    assert env["row_model"]   == "bert"
    assert env["table_model"] == "bert"
    assert env["splits"]["test"]["cell_f1"] == 0.599
    assert env["splits"]["test"]["uj_h"]    == 0.139
    assert env["status"] == "completed"


# == table_retrieval query-side auto-extract ================================
# These verify _auto_extract_queries dispatches the right subprocess command
# and is idempotent (skips per-split when the pickle already exists).

def test_auto_extract_queries_calls_query_encoder_for_both_splits(tmp_path):
    """Auto-extract should run the query encoder once for train + once for dev."""
    from argparse import Namespace
    from pathlib import Path
    labels = tmp_path / "nq_tables" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    (labels.parent / "train.json").write_text("[]")
    (labels.parent / "dev.json").write_text("[]")
    table_pkl = tmp_path / "embeddings" / "table" / "bert" / "nq_tables.pkl"
    table_pkl.parent.mkdir(parents=True)
    table_pkl.write_bytes(b"")
    # cross-encoder: table model = bert, query encoder (from --probe) = mpnet
    args = Namespace(
        model="bert", probe="mpnet", dataset="nq_tables", task="table_retrieval",
        extract_device=None,
    )
    with mock.patch("trl_bench.run.subprocess.run") as sp:
        # Have subprocess.run write the expected file so the function sees
        # the queries pickle "succeed".
        def fake_run(cmd, check=False):
            argd = dict(zip(cmd[3::2], cmd[4::2]))
            Path(argd["--output"]).write_bytes(b"queries")
            return mock.Mock(returncode=0)
        sp.side_effect = fake_run
        ok = run_mod._auto_extract_queries(
            args, labels_path=labels, table_pkl=table_pkl,
        )
    assert ok is True
    assert sp.call_count == 2  # train + dev
    cmds = [c.args[0] for c in sp.call_args_list]
    runners = [c[2] for c in cmds]
    # the ENCODER (mpnet from --probe) runs, NOT the bert table model
    assert all(r == "trl_bench.models.mpnet.generate_text_embeddings" for r in runners)
    outputs = sorted(dict(zip(c[3::2], c[4::2]))["--output"] for c in cmds)
    assert outputs[0].endswith("queries_dev.pkl")
    assert outputs[1].endswith("queries_train.pkl")
    # files persisted under <emb_root>/table_retrieval/<encoder>/
    qdir = table_pkl.parent.parent.parent / "table_retrieval" / "mpnet"
    assert (qdir / "queries_train.pkl").exists()
    assert (qdir / "queries_dev.pkl").exists()


def test_auto_extract_queries_idempotent_when_pickle_already_exists(tmp_path):
    """Pre-existing queries_*.pkl files cause the encoder to be skipped."""
    from argparse import Namespace
    labels = tmp_path / "nq_tables" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    (labels.parent / "train.json").write_text("[]")
    (labels.parent / "dev.json").write_text("[]")
    table_pkl = tmp_path / "embeddings" / "table" / "bert" / "nq_tables.pkl"
    table_pkl.parent.mkdir(parents=True)
    table_pkl.write_bytes(b"")
    # Pre-existing encoder queries under <emb_root>/table_retrieval/<encoder>/.
    qdir = table_pkl.parent.parent.parent / "table_retrieval" / "mpnet"
    qdir.mkdir(parents=True)
    (qdir / "queries_train.pkl").write_bytes(b"cached")
    (qdir / "queries_dev.pkl").write_bytes(b"cached")
    args = Namespace(
        model="bert", probe="mpnet", dataset="nq_tables", task="table_retrieval",
        extract_device=None,
    )
    with mock.patch("trl_bench.run.subprocess.run") as sp:
        ok = run_mod._auto_extract_queries(
            args, labels_path=labels, table_pkl=table_pkl,
        )
    assert ok is True
    assert sp.call_count == 0  # both splits cached -> no subprocess


def test_auto_extract_queries_fails_for_unwired_encoder(tmp_path):
    """A --probe query encoder that isn't wired logs an error and returns False."""
    from argparse import Namespace
    labels = tmp_path / "nq_tables" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    (labels.parent / "train.json").write_text("[]")
    (labels.parent / "dev.json").write_text("[]")
    table_pkl = tmp_path / "embeddings" / "table" / "bert" / "nq_tables.pkl"
    table_pkl.parent.mkdir(parents=True)
    table_pkl.write_bytes(b"")
    # tapas is a valid table model but NOT a wired query encoder
    args = Namespace(
        model="bert", probe="tapas", dataset="nq_tables", task="table_retrieval",
        extract_device=None,
    )
    ok = run_mod._auto_extract_queries(
        args, labels_path=labels, table_pkl=table_pkl,
    )
    assert ok is False


def test_resolve_embeddings_path_triggers_query_auto_extract_for_retrieval(tmp_path):
    """_resolve_embeddings_path should call _auto_extract_queries when task is
    table_retrieval AND the table pickle resolves."""
    from argparse import Namespace
    labels = tmp_path / "nq_tables" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    (labels.parent / "train.json").write_text("[]")
    (labels.parent / "dev.json").write_text("[]")
    # Pre-create the table pickle so resolution short-circuits at step 2.
    table_pkl = tmp_path / "embeddings" / "table" / "mpnet" / "nq_tables.pkl"
    table_pkl.parent.mkdir(parents=True)
    table_pkl.write_bytes(b"cached")
    args = Namespace(
        model="mpnet", dataset="nq_tables", task="table_retrieval",
        setting="cls_embedding", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=None,
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )
    with mock.patch("trl_bench.run._auto_extract_queries", return_value=True) as fae:
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)
    assert result == table_pkl
    assert fae.call_count == 1


def test_resolve_embeddings_path_tuta_table_uses_native_runner_not_stage2(tmp_path):
    """tuta TABLE-level extraction goes through the NATIVE table runner that
    writes a populated cls_embedding, NOT the column-extract + Stage-2 path.

    Root cause this guards: tuta's column pickle holds only row embeddings, so
    the Stage-2 aggregator produced an all-None table_embedding dict and every
    cls cell crashed. The native runner emits the table pickle directly.
    """
    from argparse import Namespace
    labels = tmp_path / "spider_join" / "labels.json"
    (labels.parent / "tables_all").mkdir(parents=True)
    labels.write_text("{}")
    args = Namespace(
        model="tuta", dataset="spider_join", task="join_classification",
        setting="cls_embedding", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=str(tmp_path / "ckpts"),
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )
    # Provide the gated checkpoint so build_extractor_command resolves it.
    ckpt = tmp_path / "ckpts" / "tuta" / "tuta.bin"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    table_pkl = tmp_path / "embeddings" / "table" / "tuta" / "spider_join.pkl"

    runners_seen = []

    def fake_run(cmd, check=False):
        # Record the runner module each subprocess invokes, and emulate the
        # native runner writing its table pickle so resolution succeeds.
        if "-m" in cmd:
            runner = cmd[cmd.index("-m") + 1]
            runners_seen.append(runner)
            argd = dict(zip(cmd[3::2], cmd[4::2]))
            out = argd.get("--output_path")
            if out:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_bytes(b"native")
        return mock.Mock(returncode=0)

    from pathlib import Path
    with mock.patch("trl_bench.run.subprocess.run", side_effect=fake_run):
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)

    assert result == table_pkl, f"expected table pkl, got {result}"
    # The native table runner must have been invoked...
    assert "trl_bench.models.tuta.generate_table_embeddings_native" in runners_seen, (
        f"native runner not invoked; runners seen: {runners_seen}"
    )
    # ...and the broken column+Stage-2 path must NOT have run for tuta.
    assert "trl_bench.scripts.generate_table_embeddings" not in runners_seen, (
        f"Stage-2 aggregator was invoked for tuta (should be bypassed); "
        f"runners seen: {runners_seen}"
    )
    assert "trl_bench.models.tuta.generate_embeddings_directory" not in runners_seen, (
        f"row-only column runner was invoked for tuta table extraction; "
        f"runners seen: {runners_seen}"
    )


def test_resolve_embeddings_path_skips_query_extract_for_non_retrieval(tmp_path):
    """Non-retrieval tasks must NOT trigger query auto-extract."""
    from argparse import Namespace
    labels = tmp_path / "spider_join" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    table_pkl = tmp_path / "embeddings" / "table" / "bert" / "spider_join.pkl"
    table_pkl.parent.mkdir(parents=True)
    table_pkl.write_bytes(b"cached")
    args = Namespace(
        model="bert", dataset="spider_join", task="join_classification",
        setting="cls_embedding", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=None,
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )
    with mock.patch("trl_bench.run._auto_extract_queries") as fae:
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)
    assert result == table_pkl
    assert fae.call_count == 0  # never called for non-retrieval tasks


# --- semantic_parsing question-side auto-extract (bug #6) -------------------
# MAPO needs per-token question embeddings at <col_pkl.parent>/<setting>/
# questions_{train,dev,test}.pkl; run.py must generate them, not just columns.

def test_auto_extract_questions_encodes_all_three_splits(tmp_path):
    """semantic_parsing auto-extract runs the TOKEN-mode question encoder once
    per split (train/dev/test), writing questions_<split>.pkl under
    <column_pkl.parent>/<setting>/ -- exactly where the MAPO decoder reads."""
    from argparse import Namespace
    from pathlib import Path
    labels = tmp_path / "wiki_table_questions" / "labels.json"
    (labels.parent / "data_split_1").mkdir(parents=True)
    labels.write_text("{}")
    for split in ("train", "dev", "test"):
        (labels.parent / "data_split_1" / f"{split}_split.jsonl").write_text("{}\n")
    column_pkl = tmp_path / "embeddings" / "column" / "bert" / "wiki_table_questions.pkl"
    column_pkl.parent.mkdir(parents=True)
    column_pkl.write_bytes(b"")
    args = Namespace(
        model="bert", dataset="wiki_table_questions", task="semantic_parsing",
        setting="sentence_t5", extract_device=None,
    )
    with mock.patch("trl_bench.run.subprocess.run") as sp:
        def fake_run(cmd, check=False):
            argd = dict(zip(cmd[3::2], cmd[4::2]))
            Path(argd["--output"]).write_bytes(b"q")
            return mock.Mock(returncode=0)
        sp.side_effect = fake_run
        ok = run_mod._auto_extract_questions(
            args, labels_path=labels, column_pkl=column_pkl,
        )
    assert ok is True
    assert sp.call_count == 3  # train + dev + test
    for c in sp.call_args_list:
        cmd = c.args[0]
        assert cmd[cmd.index("--mode") + 1] == "token"
        assert cmd[cmd.index("--tokens_field") + 1] == "tokens"
        assert cmd[cmd.index("--id_field") + 1] == "id"
    qdir = column_pkl.parent / "sentence_t5"
    for split in ("train", "dev", "test"):
        assert (qdir / f"questions_{split}.pkl").exists()


def test_auto_extract_questions_idempotent_when_pickles_exist(tmp_path):
    """Pre-existing questions_*.pkl cause the encoder to be skipped."""
    from argparse import Namespace
    labels = tmp_path / "wiki_table_questions" / "labels.json"
    (labels.parent / "data_split_1").mkdir(parents=True)
    labels.write_text("{}")
    for split in ("train", "dev", "test"):
        (labels.parent / "data_split_1" / f"{split}_split.jsonl").write_text("{}\n")
    column_pkl = tmp_path / "embeddings" / "column" / "bert" / "wiki_table_questions.pkl"
    column_pkl.parent.mkdir(parents=True)
    column_pkl.write_bytes(b"")
    qdir = column_pkl.parent / "sentence_t5"
    qdir.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        (qdir / f"questions_{split}.pkl").write_bytes(b"cached")
    args = Namespace(
        model="bert", dataset="wiki_table_questions", task="semantic_parsing",
        setting="sentence_t5", extract_device=None,
    )
    with mock.patch("trl_bench.run.subprocess.run") as sp:
        ok = run_mod._auto_extract_questions(
            args, labels_path=labels, column_pkl=column_pkl,
        )
    assert ok is True
    assert sp.call_count == 0  # all splits cached -> no subprocess


def test_auto_extract_questions_fails_for_non_designated_question_encoder(tmp_path):
    """The model under test supplies columns, NOT questions: a non-designated
    --setting (token-capable bert/gte, openai, or a table model) makes
    auto-extract return False with a clear message rather than crashing the
    decoder with a multiprocess FileNotFound storm. (main() normalizes these to
    sentence_t5 upstream; this is the defense-in-depth path.)"""
    from argparse import Namespace
    labels = tmp_path / "wiki_table_questions" / "labels.json"
    (labels.parent / "data_split_1").mkdir(parents=True)
    labels.write_text("{}")
    for split in ("train", "dev", "test"):
        (labels.parent / "data_split_1" / f"{split}_split.jsonl").write_text("{}\n")
    column_pkl = tmp_path / "embeddings" / "column" / "bert" / "wiki_table_questions.pkl"
    column_pkl.parent.mkdir(parents=True)
    column_pkl.write_bytes(b"")
    args = Namespace(
        model="bert", dataset="wiki_table_questions", task="semantic_parsing",
        setting="bert", extract_device=None,
    )
    ok = run_mod._auto_extract_questions(
        args, labels_path=labels, column_pkl=column_pkl,
    )
    assert ok is False


def test_resolve_embeddings_path_triggers_question_auto_extract_for_semparse(tmp_path):
    """_resolve_embeddings_path must call _auto_extract_questions when task is
    semantic_parsing AND the column pickle resolves."""
    from argparse import Namespace
    labels = tmp_path / "wiki_table_questions" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    column_pkl = tmp_path / "embeddings" / "column" / "bert" / "wiki_table_questions.pkl"
    column_pkl.parent.mkdir(parents=True)
    column_pkl.write_bytes(b"cached")
    args = Namespace(
        model="bert", dataset="wiki_table_questions", task="semantic_parsing",
        setting="sentence_t5", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=None,
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )
    with mock.patch("trl_bench.run._auto_extract_questions", return_value=True) as fae:
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)
    assert result == column_pkl
    assert fae.call_count == 1


def test_resolve_question_encoder_defaults_semparse_to_sentence_t5():
    """For semantic_parsing the setting axis IS the question encoder, restricted
    to {sentence_t5, mpnet} (default sentence_t5); the model supplies columns.
    The gen_grid_cells diagonal (setting=model) and table-agg modes normalize to
    the default. Non-semparse tasks pass through unchanged."""
    from trl_bench.run import _resolve_question_encoder
    assert _resolve_question_encoder("join_classification", "cls_embedding") == "cls_embedding"
    assert _resolve_question_encoder("semantic_parsing", "mpnet") == "mpnet"
    assert _resolve_question_encoder("semantic_parsing", "sentence_t5") == "sentence_t5"
    assert _resolve_question_encoder("semantic_parsing", "bert") == "sentence_t5"
    assert _resolve_question_encoder("semantic_parsing", "cls_embedding") == "sentence_t5"


@pytest.mark.parametrize("model,expected_runner,expected_model_id", [
    ("mpnet",
     "trl_bench.utils.generate_table_embeddings_text_encoder",
     "sentence-transformers/all-mpnet-base-v2"),
    ("sentence_t5",
     "trl_bench.utils.generate_table_embeddings_text_encoder",
     "sentence-transformers/sentence-t5-base"),
    ("tapex",
     "trl_bench.models.tapex.generate_table_embeddings",
     "microsoft/tapex-base"),
])
def test_run_resolves_table_pickle_via_table_encoder(
    model, expected_runner, expected_model_id, tmp_path,
):
    """For models in ``_TABLE_ENCODERS``, ``_resolve_embeddings_path`` must
    take the table-DIRECT branch -- a single subprocess.run that invokes the
    table-encoder runner. It must NOT call the column-extractor +
    Stage-2-aggregator path.
    """
    from argparse import Namespace
    # Stage-0 staged layout: labels.json + sibling tables_all/.
    labels = tmp_path / "ds_xy" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    tables_all = labels.parent / "tables_all"
    tables_all.mkdir()
    expected_table_pkl = tmp_path / "embeddings" / "table" / model / "ds_xy.pkl"
    args = Namespace(
        # task=join_classification → derived_pkl resolves to the table path;
        # this test exercises the table-direct (_TABLE_ENCODERS) branch, not
        # the col-task-routing branch. column_type_prediction (which would
        # route to embeddings/column/) is covered separately by
        # test_resolve_embeddings_path_routes_col_probe_tasks_to_column_pickle.
        model=model, dataset="ds_xy", task="join_classification",
        setting="cls_embedding", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=None,
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )

    captured_cmds: list[list[str]] = []

    def fake_subprocess_run(cmd, check=False, **kwargs):
        captured_cmds.append(list(cmd))
        # Pretend the runner produced the table pickle.
        expected_table_pkl.parent.mkdir(parents=True, exist_ok=True)
        expected_table_pkl.write_bytes(b"fake-table-pkl")
        rc = mock.Mock()
        rc.returncode = 0
        return rc

    with mock.patch("trl_bench.run.subprocess.run", side_effect=fake_subprocess_run):
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)

    assert result == expected_table_pkl, "expected the table pickle path back"
    # Exactly one subprocess call: the table-encoder runner. The
    # column-extractor + Stage-2 aggregator branch must not fire.
    assert len(captured_cmds) == 1, (
        f"expected ONE subprocess call (table-encoder), got "
        f"{len(captured_cmds)}: {captured_cmds}"
    )
    cmd = captured_cmds[0]
    assert cmd[:3] == [sys.executable, "-m", expected_runner]
    # Verify the table-direct CLI shape.
    args_dict = dict(zip(cmd[3::2], cmd[4::2]))
    assert args_dict["--input_dir"] == str(tables_all)
    assert args_dict["--output_path"] == str(expected_table_pkl)
    assert args_dict["--model"] == expected_model_id
    # The Stage-2 aggregator (``trl_bench.scripts.generate_table_embeddings``)
    # must NOT appear -- that would mean we accidentally took the column-
    # extractor branch.
    assert all(
        "trl_bench.scripts.generate_table_embeddings" not in part
        for part in cmd
    )


def test_run_table_encoder_short_circuits_when_pickle_exists(tmp_path):
    """If ``<embeddings-dir>/table/<model>/<dataset>.pkl`` already exists,
    the table-encoder branch must NOT fire -- the cached pickle wins.
    """
    from argparse import Namespace
    labels = tmp_path / "ds_xy" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    (labels.parent / "tables_all").mkdir()
    table_pkl = tmp_path / "embeddings" / "table" / "tapex" / "ds_xy.pkl"
    table_pkl.parent.mkdir(parents=True)
    table_pkl.write_bytes(b"cached")
    args = Namespace(
        model="tapex", dataset="ds_xy", task="join_classification",
        setting="cls_embedding", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=None,
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)
    assert result == table_pkl
    assert srun.call_count == 0, "cached table pickle should short-circuit"


# == Column-pickle dispatch for learned col-granularity tasks ==================
# Regression guard: column_type_prediction (CTA), column_relation_prediction
# (CRA), and semantic_parsing all consume column-level pickles (with the
# 'column_embeddings' key), not table-level. _resolve_embeddings_path must
# route them to embeddings/column/<m>/<ds>.pkl. Pre-fix the default else-branch
# routed them to embeddings/table/, causing KeyError('column_embeddings') at
# runtime when the probe runner loaded the wrong pickle. See run.py:909-918.
@pytest.mark.parametrize("task", [
    "column_type_prediction",
    "column_relation_prediction",
    "semantic_parsing",
    "join_containment",
])
def test_resolve_embeddings_path_routes_col_probe_tasks_to_column_pickle(
    task, tmp_path,
):
    from argparse import Namespace
    labels = tmp_path / "ds_xy" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    col_pkl = tmp_path / "embeddings" / "column" / "bert" / "ds_xy.pkl"
    col_pkl.parent.mkdir(parents=True)
    col_pkl.write_bytes(b"cached")
    args = Namespace(
        model="bert", dataset="ds_xy", task=task,
        setting="cls_embedding", probe="linear", seed=42,
        embeddings_path=None, labels_path=str(labels),
        no_auto_stage=True, embeddings_dir=str(tmp_path / "embeddings"),
        extract_device=None, checkpoint_root=None,
        results_dir=str(tmp_path / "results"),
        data_root=str(tmp_path),
        configs_root="configs/downstream",
    )
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        result = run_mod._resolve_embeddings_path(args, labels_path=labels)
    assert result == col_pkl, (
        f"{task} must resolve to embeddings/column/<m>/<ds>.pkl, got {result}"
    )
    assert srun.call_count == 0, "cached column pickle should short-circuit"


def test_build_column_repair_command_fills_tabbie_capped_columns():
    """tabbie's 20-col grid caps wide tables; run.py must run the embedding-repair
    pass the paper's generation pipeline used (generate_column_embeddings then
    embedding_repair scan+repair, --chunk_size 64). The CLI must target
    --action repair on the column pickle with the paper's chunk_size/max_rows."""
    cmd = run_mod._build_column_repair_command(
        "tabbie", "valentine", "/emb/column/tabbie/valentine.pkl",
        "/ck/tabbie/weights.pt", device="cuda:0", chunk_size=64, max_rows=30,
    )
    assert cmd[:3] == [sys.executable, "-m", "trl_bench.utils.embedding_repair.cli"]
    assert cmd[cmd.index("--model") + 1] == "tabbie"
    assert cmd[cmd.index("--dataset") + 1] == "valentine"
    assert cmd[cmd.index("--action") + 1] == "repair"
    assert cmd[cmd.index("--embeddings") + 1] == "/emb/column/tabbie/valentine.pkl"
    assert cmd[cmd.index("--checkpoint") + 1] == "/ck/tabbie/weights.pt"
    assert cmd[cmd.index("--chunk_size") + 1] == "64"
    assert cmd[cmd.index("--max_rows") + 1] == "30"
    assert cmd[cmd.index("--device") + 1] == "cuda:0"
    # max_rows=None / checkpoint=None omit those flags (tabsketchfm / HF models)
    cmd2 = run_mod._build_column_repair_command(
        "tabsketchfm", "valentine", "/p.pkl", None, chunk_size=64, max_rows=None)
    assert "--max_rows" not in cmd2 and "--checkpoint" not in cmd2
    assert cmd2[cmd2.index("--chunk_size") + 1] == "64"


@pytest.mark.parametrize("model,exp_max_rows,exp_chunk", [
    ("tabbie", "30", "64"),
    ("bert", "100", "64"),
    ("tapas", "100", "64"),
    ("tabert", "100", "16"),    # TaBERT used chunk_size 16 in the paper
    ("starmie", "1000", "64"),
])
def test_maybe_repair_columns_uses_paper_per_model_params(model, exp_max_rows, exp_chunk, tmp_path):
    """The repair's max_rows/chunk_size must match each model's extraction (the
    paper's round-gen values) -- else re-embedded columns use different rows."""
    import argparse
    args = argparse.Namespace(model=model, dataset="valentine", extract_device="cuda:0")
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        srun.return_value.returncode = 0
        run_mod._maybe_repair_columns(args, tmp_path / "valentine.pkl", "/ck/root")
    cmd = srun.call_args[0][0]
    assert cmd[cmd.index("--model") + 1] == model
    assert cmd[cmd.index("--max_rows") + 1] == exp_max_rows
    assert cmd[cmd.index("--chunk_size") + 1] == exp_chunk


def test_maybe_repair_columns_tabsketchfm_omits_max_rows(tmp_path):
    """tabsketchfm's extractor takes no --max_rows, so the repair must omit it."""
    import argparse
    args = argparse.Namespace(model="tabsketchfm", dataset="valentine", extract_device="cuda:0")
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        srun.return_value.returncode = 0
        run_mod._maybe_repair_columns(args, tmp_path / "valentine.pkl", "/ck/root")
    cmd = srun.call_args[0][0]
    assert "--max_rows" not in cmd
    assert cmd[cmd.index("--chunk_size") + 1] == "64"


def test_maybe_repair_columns_skips_model_without_repair_adapter(tmp_path):
    """Column models with no repair adapter (tuta/turl) skip the repair: no
    subprocess, returns True."""
    import argparse
    args = argparse.Namespace(model="tuta", dataset="valentine", extract_device="cuda:0")
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        ok = run_mod._maybe_repair_columns(args, tmp_path / "c.pkl", "/ck")
    assert ok is True
    assert srun.call_count == 0


def test_maybe_repair_columns_omits_checkpoint_for_hf_model(tmp_path):
    """HuggingFace column models (bert/gte/tapas) have no local checkpoint -- the
    repair runs but omits --checkpoint so the CLI resolves the model name."""
    import argparse
    args = argparse.Namespace(model="bert", dataset="valentine", extract_device="cuda:0")
    pkl = tmp_path / "valentine.pkl"
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        srun.return_value.returncode = 0
        ok = run_mod._maybe_repair_columns(args, pkl, "/ck/root")
    assert ok is True
    assert srun.call_count == 1
    cmd = srun.call_args[0][0]
    assert cmd[cmd.index("--model") + 1] == "bert"
    assert cmd[cmd.index("--action") + 1] == "repair"
    assert cmd[cmd.index("--embeddings") + 1] == str(pkl)
    assert "--checkpoint" not in cmd


def test_maybe_repair_columns_runs_repair_for_tabbie(tmp_path):
    """tabbie -> the repair subprocess runs against the column pickle, with the
    checkpoint resolved from <ckpt_root>/<extractor checkpoint_template>."""
    import argparse
    args = argparse.Namespace(model="tabbie", dataset="valentine", extract_device="cuda:0")
    pkl = tmp_path / "valentine.pkl"
    with mock.patch("trl_bench.run.subprocess.run") as srun:
        srun.return_value.returncode = 0
        ok = run_mod._maybe_repair_columns(args, pkl, "/ck/root")
    assert ok is True
    assert srun.call_count == 1
    cmd = srun.call_args[0][0]
    assert "trl_bench.utils.embedding_repair.cli" in cmd
    assert cmd[cmd.index("--action") + 1] == "repair"
    assert cmd[cmd.index("--embeddings") + 1] == str(pkl)
    assert cmd[cmd.index("--checkpoint") + 1] == "/ck/root/tabbie/weights.pt"


# == Stage-1 readiness gate (row_prediction dir vs pickle) ====================
# Regression: a prior failed run left an EMPTY row_prediction embedding dir,
# whose mere existence made the auto-extract gate skip generation, so the probe
# then failed with a misleading "Available: []" (masking the real Stage-1 error).
# The gate must treat a row_prediction dir as ready only when populated.

def test_embeddings_ready_row_prediction_empty_dir_not_ready(tmp_path):
    d = tmp_path / "rp"; d.mkdir()
    assert run_mod._embeddings_ready(d, "row_prediction") is False

def test_embeddings_ready_row_prediction_metadata_is_ready(tmp_path):
    d = tmp_path / "rp"; d.mkdir()
    (d / "metadata.json").write_text("{}")
    assert run_mod._embeddings_ready(d, "row_prediction") is True

def test_embeddings_ready_row_prediction_not_applicable_is_ready(tmp_path):
    d = tmp_path / "rp"; d.mkdir()
    (d / "not_applicable.json").write_text("{}")
    assert run_mod._embeddings_ready(d, "row_prediction") is True

def test_embeddings_ready_pickle_task_uses_existence(tmp_path):
    p = tmp_path / "x.pkl"
    assert run_mod._embeddings_ready(p, "column_type_prediction") is False
    p.write_bytes(b"")
    assert run_mod._embeddings_ready(p, "column_type_prediction") is True


# --- row_prediction Stage-1 auto-extract dispatch (unified through ------------
#     build_row_data_commands) ------------------------------------------------

def _row_prediction_resolve(tmp_path, model, *, ckpt_root):
    """Drive run.py's `_resolve_embeddings_path` for one row_prediction model.

    Stages a minimal labels.json (so the auto-extract branch is entered) and an
    empty derived dir (so it is NOT already 'ready'); mocks build_row_data_commands
    + subprocess.run (the mock writes metadata.json so the dir resolves 'ready').
    Returns (returned_path, build_row_data_commands_mock).
    """
    labels = tmp_path / "data" / model / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps({"data": "data.csv", "label_columns": ["y"]}))
    emb_dir = tmp_path / "emb"
    derived = emb_dir / "row_prediction" / model / "ds1"

    args = run_mod._parse_args([
        "--model", model, "--task", "row_prediction",
        "--dataset", "ds1", "--setting", "y", "--probe", "mlp", "--seed", "42",
        "--embeddings-dir", str(emb_dir),
        "--labels-path", str(labels),
        "--checkpoint-root", ckpt_root,
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(tmp_path / "data"),
    ])

    sentinel_cmd = [sys.executable, "-m", "fake.runner", "--x"]

    def _fake_subproc(cmd, *a, **k):
        # Mirror the real runner's side effect: populate the unified dir.
        derived.mkdir(parents=True, exist_ok=True)
        (derived / "metadata.json").write_text("{}")
        return mock.Mock(returncode=0)

    with mock.patch("trl_bench.run.build_row_data_commands",
                    return_value=[sentinel_cmd]) as brc, \
         mock.patch("trl_bench.run.subprocess.run", side_effect=_fake_subproc):
        out = run_mod._resolve_embeddings_path(args, labels_path=labels)
    return out, brc, derived


@pytest.mark.parametrize("model", ["bert", "gte", "tabbie", "tuta"])
def test_row_prediction_autoextract_routes_through_build_row_data_commands(
        model, tmp_path):
    """Every row_prediction model (incl. tabbie/tuta, the previously-broken ones)
    auto-extracts through build_row_data_commands -- the YAML-driven faithful port
    of the slurm generator -- with checkpoint_root threaded through so tabbie/tuta
    get their REQUIRED --model_path. (Before the fix, in-_ROW_DATA_RUNNERS models
    used a hardcoded inline CLI that dropped --model_path; tuta hit the SSL branch
    that also dropped it.)"""
    out, brc, derived = _row_prediction_resolve(
        tmp_path, model, ckpt_root="/ckroot")
    assert out == derived, "resolver returns the unified_row_embedding directory"
    assert brc.call_count == 1
    # checkpoint_root must be passed (positionally it's keyword-only here).
    _, kwargs = brc.call_args
    assert kwargs.get("checkpoint_root") == "/ckroot", (
        "checkpoint_root must be threaded so tabbie/tuta resolve --model_path"
    )


def test_row_prediction_autoextract_tuta_real_command_has_model_path(tmp_path):
    """End-to-end (registry-integrated) check: with build_row_data_commands NOT
    mocked, the tuta Stage-1 command that run.py would execute contains
    --model_path resolved under the passed checkpoint-root. Guards the full
    run.py -> registry -> YAML wiring for the licensed-checkpoint case."""
    labels = tmp_path / "data" / "tuta" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps({"data": "data.csv", "label_columns": ["y"]}))
    emb_dir = tmp_path / "emb"
    args = run_mod._parse_args([
        "--model", "tuta", "--task", "row_prediction",
        "--dataset", "ds1", "--setting", "y", "--probe", "mlp", "--seed", "42",
        "--embeddings-dir", str(emb_dir),
        "--labels-path", str(labels),
        "--checkpoint-root", "/ckroot",
        "--results-dir", str(tmp_path / "results"),
        "--data-root", str(tmp_path / "data"),
    ])
    captured = {}

    def _capture(cmd, *a, **k):
        captured["cmd"] = cmd
        d = emb_dir / "row_prediction" / "tuta" / "ds1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text("{}")
        return mock.Mock(returncode=0)

    with mock.patch("trl_bench.run.subprocess.run", side_effect=_capture):
        run_mod._resolve_embeddings_path(args, labels_path=labels)

    cmd = captured["cmd"]
    assert "generate_row_embeddings.py" in cmd[1]
    assert "--model_path" in cmd
    assert cmd[cmd.index("--model_path") + 1] == "/ckroot/tuta/tuta.bin"
    assert "--dataset_dir" in cmd  # tuta's native arg name
