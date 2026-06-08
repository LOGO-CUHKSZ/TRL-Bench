"""Tests for the (model, task) -> command dispatch registry.

Behavioural tests: where possible, we assert on the *content* of the
command the registry would dispatch, not just its shape — because the
shape was previously preserved while the content disagreed with what the
wrappers accept.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
from trl_bench.registry import (
    build_command, is_valid_cell, list_cells,
    SettingError, PROBE_TASKS, ROW_EMBEDDING_TASKS, TABLE_EMBEDDING_TASKS,
    DETERMINISTIC_TASKS, DLTE_TASKS,
)


# == Category-set sanity checks ==============================================

def test_probe_tasks_includes_paper_supervised_tasks():
    for t in ("join_classification", "union_classification", "union_regression",
              "column_type_prediction", "record_linkage"):
        assert t in PROBE_TASKS


def test_row_embedding_tasks_match_paper():
    assert "row_prediction" in ROW_EMBEDDING_TASKS
    assert "record_linkage" in ROW_EMBEDDING_TASKS


def test_table_embedding_tasks_match_paper():
    for t in ("table_retrieval", "table_subset", "union_search"):
        assert t in TABLE_EMBEDDING_TASKS


# == Validity of (model, task) cells =========================================

def test_is_valid_cell_supervised_models_can_do_supervised_tasks():
    assert is_valid_cell("bert", "join_classification") is True
    assert is_valid_cell("saint", "row_prediction") is True


def test_is_valid_cell_target_table_ssl_cannot_do_table_tasks():
    assert is_valid_cell("saint", "table_retrieval") is False
    assert is_valid_cell("dae", "schema_matching") is False


def test_tabbie_tuta_expose_row_granularity():
    # The paper counts 14 row models, including TABBIE + TUTA (their row
    # embeddings are synthesized via per-row mini-tables). The registry must
    # accept them on row tasks while retaining their col/table capabilities.
    for m in ("tabbie", "tuta"):
        assert is_valid_cell(m, "record_linkage") is True
        assert is_valid_cell(m, "row_prediction") is True
        assert is_valid_cell(m, "join_classification") is True  # still col/table
    assert is_valid_cell("tuta", "table_retrieval") is True


def test_is_valid_cell_matches_paper_reported_cells():
    """is_valid_cell must match the paper's reported (model, task) matrix
    (.paper_reference/ct_table.csv), not just the coarse granularity heuristic.
    Two paper-derived reconciliations (full audit 2026-06-02):

    Type-1 (granularity says valid, paper EMPTY -> exclude): tapex (table-only,
    crashes on per-column with KeyError) and tuta (column-capable but
    impractically slow on the large per-column datasets, so the paper ran it
    table-level only) are excluded from the 8 per-column tasks, but KEEP their 5
    table-level cells.

    Type-2 (paper REPORTS, heuristic too strict -> add granularity): starmie
    gains 'table' so table_subset/table_retrieval become valid."""
    per_column = ("column_type_prediction", "column_clustering",
                  "column_relation_prediction", "join_search", "join_containment",
                  "schema_matching", "union_search", "semantic_parsing")
    table_level = ("join_classification", "union_classification",
                   "union_regression", "table_subset", "table_retrieval")
    for m in ("tapex", "tuta"):
        for t in per_column:
            assert is_valid_cell(m, t) is False, f"{m} x {t} must be paper-excluded"
        for t in table_level:
            assert is_valid_cell(m, t) is True, f"{m} x {t} must stay valid"
    # Type-2: starmie's table-level cells the paper reports.
    assert is_valid_cell("starmie", "table_retrieval") is True
    assert is_valid_cell("starmie", "table_subset") is True
    # starmie still does its per-column tasks (paper reports ColType=0.6751 etc.)
    assert is_valid_cell("starmie", "column_type_prediction") is True
    # Column-extractor models are unaffected by the exclusions.
    assert is_valid_cell("bert", "semantic_parsing") is True
    assert is_valid_cell("bert", "column_type_prediction") is True


def test_list_cells_returns_at_least_50_supported():
    cells = list(list_cells())
    assert len(cells) >= 50
    assert all(isinstance(c, tuple) and len(c) == 2 for c in cells)


# == build_command for probe tasks ===========================================

def test_build_command_probe_task_dispatch_join_classification(tmp_path):
    embeddings = tmp_path / "bert_spider_join.pkl"
    embeddings.write_bytes(b"")
    labels = tmp_path / "labels.json"
    labels.write_text("{}")

    stages = build_command(
        model="bert", task="join_classification",
        dataset="spider_join", setting="cls_embedding",
        probe="linear", seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path / "configs",
    )

    # Probe is a single stage right now (Stage-3); Stage-1/2 are pre-supplied.
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]

    # Canonical invocation: python -m trl_bench.utils.downstream.run_task ...
    assert cmd[:3] == [sys.executable, "-m", "trl_bench.utils.downstream.run_task"]

    flags = cmd[3:]
    args = dict(zip(flags[::2], flags[1::2]))
    # Mirror what the canonical .sbatch passes for this cell.
    assert args["--embeddings"]    == str(embeddings)
    assert args["--labels"]        == str(labels)
    assert args["--task_name"]     == "join_spider_join"
    assert args["--task_type"]     == "classification"
    assert args["--embedding_type"] == "cls"
    assert args["--combination_method"] == "concat"
    assert args["--hidden_dim"]    == "256"
    assert args["--num_labels"]    == "2"
    assert args["--batch_size"]    == "32"
    assert args["--max_epochs"]    == "50"
    assert args["--learning_rate"] == "0.001"
    assert args["--dropout_prob"]  == "0.1"
    assert args["--seed"]          == "42"
    assert args["--head_type"]     == "linear"


def test_build_command_probe_task_requires_embeddings_and_labels(tmp_path):
    with pytest.raises(SettingError, match="embeddings-path"):
        build_command(
            model="bert", task="join_classification",
            dataset="spider_join", setting="cls_embedding",
            probe="linear", seed=42,
            results_dir=tmp_path,
        )


def test_build_command_probe_task_requires_probe(tmp_path):
    with pytest.raises(SettingError, match="probe"):
        build_command(
            model="bert", task="join_classification",
            dataset="spider_join", setting="cls_embedding",
            probe=None, seed=42,
            results_dir=tmp_path,
            embeddings_path=tmp_path / "x.pkl",
            labels_path=tmp_path / "l.json",
        )


def test_build_command_invalid_cell_raises(tmp_path):
    with pytest.raises(SettingError, match="not supported"):
        build_command(
            model="saint", task="table_retrieval",
            dataset="tus", setting="cls_embedding",
            probe="linear", seed=42,
            results_dir=tmp_path,
            embeddings_path=tmp_path / "x.pkl",
            labels_path=tmp_path / "l.json",
        )


def test_build_command_invalid_cell_raises_clearly(tmp_path):
    # Granularity mismatch yields a clear SettingError. With all 18 paper-grid
    # tasks now wired, the legacy "task not supported" surface no longer
    # fires for any task in ``_TASK_GRANULARITIES``; we instead verify the
    # invalid-cell error path, which is the only remaining loud-failure mode
    # any caller can hit. Models without ``row`` granularity (bert has row, so
    # we use ``starmie`` which is col-only) cannot run row-level tasks.
    with pytest.raises(SettingError, match="not supported"):
        build_command(
            model="starmie", task="row_prediction",
            dataset="openml_3", setting="row_embedding",
            probe="linear", seed=42,
            results_dir=tmp_path,
        )


# == Family-specific build_command coverage (CTA, CRA, retrieval) ============

def _flag_to_value(cmd, flag):
    """Find ``flag`` in ``cmd`` and return its immediate-next-token value."""
    if flag in cmd:
        return cmd[cmd.index(flag) + 1]
    return None


def test_cta_build_command_produces_expected_cli(tmp_path):
    embeddings = tmp_path / "bert_sato.pkl"
    embeddings.write_bytes(b"")
    labels = tmp_path / "sato" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")

    stages = build_command(
        model="bert", task="column_type_prediction",
        dataset="sato", setting="cls_embedding",
        probe="linear", seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )

    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]

    # Canonical CTA invocation.
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.column_type_prediction.train_ct_mode4"]

    # Flag-by-flag asserts — mirrors the canonical .sbatch.
    assert _flag_to_value(cmd, "--embeddings")    == str(embeddings)
    assert _flag_to_value(cmd, "--dataset")       == str(labels.parent)
    assert _flag_to_value(cmd, "--num_epochs")    == "10"
    assert _flag_to_value(cmd, "--batch_size")    == "20"
    assert _flag_to_value(cmd, "--learning_rate") == "0.001"
    assert _flag_to_value(cmd, "--seed")          == "42"
    assert _flag_to_value(cmd, "--head_type")     == "linear"
    # CTA does NOT take pair-task flags.
    assert "--labels"            not in cmd
    assert "--task_type"         not in cmd
    assert "--embedding_type"    not in cmd
    assert "--combination_method" not in cmd


def test_cra_build_command_produces_expected_cli(tmp_path):
    embeddings = tmp_path / "bert_SOTAB.pkl"
    embeddings.write_bytes(b"")
    labels = tmp_path / "SOTAB" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")

    stages = build_command(
        model="bert", task="column_relation_prediction",
        dataset="SOTAB", setting="cls_embedding",
        probe="linear", seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )

    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]

    # Canonical CRA invocation.
    assert cmd[:3] == [
        sys.executable, "-m",
        "trl_bench.tasks.column_relation_prediction.csv_relation_pipeline",
    ]

    # Flag-by-flag asserts — note CRA's distinct flag names from CTA.
    assert _flag_to_value(cmd, "--embeddings_file") == str(embeddings)
    assert _flag_to_value(cmd, "--dataset_dir")     == str(labels.parent)
    assert _flag_to_value(cmd, "--epochs")          == "20"
    assert _flag_to_value(cmd, "--batch_size")      == "32"
    assert _flag_to_value(cmd, "--lr")              == "0.001"
    assert _flag_to_value(cmd, "--hidden_dim")      == "256"
    assert _flag_to_value(cmd, "--seed")            == "42"
    assert _flag_to_value(cmd, "--head_type")       == "linear"
    # CRA does NOT use CTA's flag names.
    assert "--embeddings"     not in cmd
    assert "--dataset"        not in cmd
    assert "--num_epochs"     not in cmd
    assert "--learning_rate"  not in cmd


def test_retrieval_build_command_produces_two_stages(tmp_path):
    # model_only (probe=<enc>_modelonly) is a 2-stage pipeline: train then eval.
    # (The cross-encoder cell encoding is covered in detail by
    # test_table_retrieval_build_command_model_only / _hybrid.)
    embeddings = tmp_path / "embeddings" / "table" / "bert" / "nq_tables.pkl"
    embeddings.parent.mkdir(parents=True)
    embeddings.write_bytes(b"")
    labels = tmp_path / "data" / "nq_tables" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")

    stages = build_command(
        model="bert", task="table_retrieval",
        dataset="nq_tables", setting="cls_embedding",
        probe="mpnet_modelonly", seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )

    assert isinstance(stages, list) and len(stages) == 2
    train_cmd, eval_cmd = stages

    assert train_cmd[:3] == [sys.executable, "-m",
                             "trl_bench.tasks.table_retrieval.train"]
    assert _flag_to_value(train_cmd, "--table_embeddings") == str(embeddings)
    # query embeddings come from the encoder, not alongside the table pickle
    assert _flag_to_value(train_cmd, "--train_query_embeddings").endswith(
        "table_retrieval/mpnet/queries_train.pkl")
    assert _flag_to_value(train_cmd, "--epochs")            == "60"
    assert _flag_to_value(train_cmd, "--batch_size")        == "512"
    assert _flag_to_value(train_cmd, "--projection_dim")    == "256"
    assert _flag_to_value(train_cmd, "--hidden_dim")        == "256"
    assert _flag_to_value(train_cmd, "--embedding_variant") == "cls_embedding"
    assert _flag_to_value(train_cmd, "--seed")              == "42"
    # train.py output_dir == eval.py projection_head's parent
    train_output_dir = _flag_to_value(train_cmd, "--output_dir")
    assert train_output_dir is not None

    assert eval_cmd[:3] == [sys.executable, "-m",
                            "trl_bench.tasks.table_retrieval.evaluate"]
    assert _flag_to_value(eval_cmd, "--table_embeddings") == str(embeddings)
    assert _flag_to_value(eval_cmd, "--projection_head") == \
        str(Path(train_output_dir) / "best_model.pt")
    assert _flag_to_value(eval_cmd, "--output_path") == \
        str(Path(train_output_dir) / "results.json")
    assert _flag_to_value(eval_cmd, "--embedding_type") == "cls_embedding"


def test_existing_pair_task_dispatch_unchanged(tmp_path):
    """Refactoring ProbeConfig + _probe_command must NOT change pair-task CLI."""
    embeddings = tmp_path / "e.pkl"
    embeddings.write_bytes(b"")
    labels = tmp_path / "l.json"
    labels.write_text("{}")

    stages = build_command(
        model="bert", task="union_classification",
        dataset="ckan_subset", setting="column_mean",
        probe="mlp", seed=72,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m", "trl_bench.utils.downstream.run_task"]
    assert _flag_to_value(cmd, "--task_name")     == "union_classification_ckan_subset"
    assert _flag_to_value(cmd, "--head_type")     == "mlp"
    assert _flag_to_value(cmd, "--embedding_type") == "column_mean"
    assert _flag_to_value(cmd, "--task_type")     == "classification"


def test_row_prediction_build_command_produces_expected_cli(tmp_path):
    """row_prediction dispatch builds the train_downstream.py CLI with the
    per-cell knobs (--seed/--model/--dataset/--head_type/--label_column).

    The setting axis carries the label column name. ``embeddings_path`` is a
    DIRECTORY of .npy files (--embedding_dir), not a pickle. The path layout
    is ``<task>/<model>/<dataset>/seed<S>/<label_col>/``.
    """
    embedding_dir = tmp_path / "row_prediction" / "saint" / "openml_3"
    embedding_dir.mkdir(parents=True)
    labels = tmp_path / "openml_3" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")

    stages = build_command(
        model="saint", task="row_prediction",
        dataset="openml_3", setting="class",
        probe="mlp", seed=42,
        results_dir=tmp_path,
        embeddings_path=embedding_dir,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]

    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.row_prediction.train_downstream"]
    assert _flag_to_value(cmd, "--embedding_dir") == str(embedding_dir)
    assert _flag_to_value(cmd, "--seed")          == "42"
    assert _flag_to_value(cmd, "--model")         == "saint"
    assert _flag_to_value(cmd, "--dataset")       == "openml_3"
    assert _flag_to_value(cmd, "--head_type")     == "mlp"
    assert _flag_to_value(cmd, "--label_column")  == "class"

    # Path layout: --output_dir is <task>/<model>/<dataset>/seed<S>/ (the
    # runner APPENDS <label_col>/ itself when --label_column is passed, so
    # the dispatcher must NOT include the label suffix).
    output_dir = _flag_to_value(cmd, "--output_dir")
    assert output_dir is not None
    assert output_dir.endswith("/evaluation/row_prediction/saint/openml_3/seed42")
    assert not output_dir.endswith("class")

    # row_prediction does NOT use pair-task flag names.
    assert "--embeddings"          not in cmd
    assert "--labels"              not in cmd
    assert "--task_type"           not in cmd
    assert "--embedding_type"      not in cmd
    assert "--combination_method"  not in cmd


def test_semantic_parsing_build_command_produces_expected_cli(tmp_path):
    """semantic_parsing dispatch builds the train-then-test pair invocation
    with DASH-separated flags (--column-pkl, --question-pkls, --dataset-path,
    --output-dir, --config <json>, --seed). Two stages:
        1) run_training.py writes ``model.best.bin`` under --output-dir
        2) run_test.py loads it and writes ``test.log`` (JSON)
    Path layout: ``<task>/<model>/<setting>/seed<S>/`` (NO probe sub-dir).
    """
    # Stage-0 inputs: column pickle + per-encoder question pickles + dataset.
    column_pkl = tmp_path / "embeddings" / "tapas" / "semantic_parsing.pkl"
    column_pkl.parent.mkdir(parents=True)
    column_pkl.write_bytes(b"")
    # Question pickles live under <embeddings_path.parent>/<setting>/.
    encoder_dir = column_pkl.parent / "mpnet"
    encoder_dir.mkdir()
    for fn in ("questions_train.pkl", "questions_dev.pkl", "questions_test.pkl"):
        (encoder_dir / fn).write_bytes(b"")
    # Dataset layout: <dataset_dir>/{tables.jsonl, data_split_1/{...}.jsonl}.
    dataset_dir = tmp_path / "datasets" / "wiki_table_questions"
    (dataset_dir / "data_split_1").mkdir(parents=True)
    (dataset_dir / "tables.jsonl").write_text("")
    (dataset_dir / "data_split_1" / "test_split.jsonl").write_text("")
    labels = dataset_dir / "labels.json"
    labels.write_text("{}")

    stages = build_command(
        model="tapas", task="semantic_parsing",
        dataset="wiki_table_questions", setting="mpnet",
        probe=None,       # semantic_parsing has no probe axis
        seed=72,
        results_dir=tmp_path,
        embeddings_path=column_pkl,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 2

    # ---- Stage 1: run_training.py ------------------------------------------
    train = stages[0]
    assert train[:3] == [sys.executable, "-m",
                         "trl_bench.tasks.semantic_parsing.run_training"]
    assert "--column-pkl"    in train
    assert "--question-pkls" in train
    assert "--dataset-path"  in train
    assert "--output-dir"    in train
    assert "--config"        in train
    assert "--seed"          in train
    assert _flag_to_value(train, "--column-pkl")   == str(column_pkl)
    assert _flag_to_value(train, "--dataset-path") == str(dataset_dir)
    assert _flag_to_value(train, "--seed")         == "72"
    assert _flag_to_value(train, "--task")         == "wiki_table_questions"
    assert _flag_to_value(train, "--decoder")      == "mapo"
    # --config points to a JSON file (mapo.json), not a YAML.
    cfg_arg = _flag_to_value(train, "--config")
    assert cfg_arg is not None and cfg_arg.endswith("mapo.json")
    # --output-dir: <task>/<model>/<setting>/seed<S>/ (no probe sub-dir).
    out_dir = _flag_to_value(train, "--output-dir")
    assert out_dir is not None
    assert out_dir.endswith("/evaluation/semantic_parsing/tapas/mpnet/seed72")
    # --question-pkls is nargs=+ : two paths immediately after the flag.
    qp_idx = train.index("--question-pkls")
    assert train[qp_idx + 1].endswith("/mpnet/questions_train.pkl")
    assert train[qp_idx + 2].endswith("/mpnet/questions_dev.pkl")

    # ---- Stage 2: run_test.py ----------------------------------------------
    test_cmd = stages[1]
    assert test_cmd[:3] == [sys.executable, "-m",
                            "trl_bench.tasks.semantic_parsing.run_test"]
    assert _flag_to_value(test_cmd, "--model").endswith("/seed72/model.best.bin")
    assert _flag_to_value(test_cmd, "--column-pkl") == str(column_pkl)
    assert (_flag_to_value(test_cmd, "--test-file")
            == str(dataset_dir / "data_split_1" / "test_split.jsonl"))
    assert (_flag_to_value(test_cmd, "--table-file")
            == str(dataset_dir / "tables.jsonl"))
    # --question-pkls in test stage: just the test pickle.
    qp_idx = test_cmd.index("--question-pkls")
    assert test_cmd[qp_idx + 1].endswith("/mpnet/questions_test.pkl")

    # semantic_parsing does NOT use pair-task / probe flag names.
    for stage in stages:
        assert "--embeddings"         not in stage
        assert "--labels"             not in stage
        assert "--task_type"          not in stage
        assert "--head_type"          not in stage
        assert "--combination_method" not in stage


def test_semantic_parsing_does_not_require_probe(tmp_path):
    """semantic_parsing has no probe-head axis; build_command must accept
    probe=None (unlike all other probe tasks).
    """
    column_pkl = tmp_path / "col.pkl"; column_pkl.write_bytes(b"")
    labels = tmp_path / "labels.json"; labels.write_text("{}")
    # Should not raise about a missing --probe.
    stages = build_command(
        model="tapas", task="semantic_parsing",
        dataset="wiki_table_questions", setting="mpnet",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=column_pkl,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert len(stages) == 2


# == DLTE: dispatch coverage =================================================
# DLTE wires three tasks (dlte_retrieval / dlte_alignment / dlte_merge) to
# the step8 / step9 / step10 runners. Each test asserts the runner module and
# the per-cell CLI knobs the dispatcher passes, plus the
# diagonal-default convention for the setting axis.


def _make_dlte_paths(tmp_path):
    """Build a plausible (embeddings, labels) pair for DLTE dispatch tests.

    The dispatcher derives ``--embeddings_root`` from
    ``embeddings_path.parent.parent.parent`` and ``--project_root`` from
    ``labels_path.parent.parent.parent``, mirroring the staged layout.
    """
    # Embeddings root: <emb_root>/table/<model>/<dataset>.pkl
    emb_root = tmp_path / "embeddings"
    embeddings = emb_root / "table" / "bert" / "dlte_v1.pkl"
    embeddings.parent.mkdir(parents=True)
    embeddings.write_bytes(b"")
    # Project root: <project_root>/datasets/dlte_v1/labels.json
    project_root = tmp_path / "data" / "dlte_v1"
    labels = project_root / "datasets" / "dlte_v1" / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    return embeddings, labels, emb_root, project_root


def test_dlte_retrieval_build_command_produces_expected_cli(tmp_path):
    """dlte_retrieval dispatch invokes step8_faiss_retrieval with --models
    set to the (single) table_model. Setting axis defaults to ``diagonal``;
    no --table_variant flag is emitted in that case.
    """
    embeddings, labels, emb_root, project_root = _make_dlte_paths(tmp_path)

    stages = build_command(
        model="bert", task="dlte_retrieval",
        dataset="dlte_v1", setting="diagonal",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.dlte.scripts.step8_faiss_retrieval"]
    assert _flag_to_value(cmd, "--models")          == "bert"
    assert _flag_to_value(cmd, "--embeddings_root") == str(emb_root)
    assert _flag_to_value(cmd, "--project_root")    == str(project_root)
    # Diagonal setting -> no --table_variant override emitted.
    assert "--table_variant" not in cmd
    # The output_root must point at <results>/evaluation/dlte/.
    out_root = _flag_to_value(cmd, "--output_root")
    assert out_root is not None and out_root.endswith("/evaluation/dlte")


def test_dlte_retrieval_setting_carries_table_variant(tmp_path):
    """Non-diagonal setting on dlte_retrieval is interpreted as the table-
    embedding variant (cls_embedding / column_mean / token_mean /
    table_embedding) and passed to the runner as ``--table_variant``.
    """
    embeddings, labels, *_ = _make_dlte_paths(tmp_path)
    stages = build_command(
        model="bert", task="dlte_retrieval",
        dataset="dlte_v1", setting="cls_embedding",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    cmd = stages[0]
    assert _flag_to_value(cmd, "--table_variant") == "cls_embedding"


def test_dlte_alignment_build_command_produces_expected_cli(tmp_path):
    """dlte_alignment dispatch invokes step9_column_alignment with --models
    set to the col_model. Diagonal default -> no --table_model emitted.
    """
    embeddings, labels, emb_root, project_root = _make_dlte_paths(tmp_path)

    stages = build_command(
        model="bert", task="dlte_alignment",
        dataset="dlte_v1", setting="diagonal",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.dlte.scripts.step9_column_alignment"]
    assert _flag_to_value(cmd, "--models")          == "bert"
    assert _flag_to_value(cmd, "--topk")            == "100"
    assert _flag_to_value(cmd, "--embeddings_root") == str(emb_root)
    assert _flag_to_value(cmd, "--project_root")    == str(project_root)
    # Diagonal default -> no --table_model emitted.
    assert "--table_model" not in cmd


def test_dlte_alignment_setting_carries_table_model(tmp_path):
    """Non-diagonal setting on dlte_alignment is the upstream table_model
    (carried separately from the col_model) and emitted as --table_model.
    """
    embeddings, labels, *_ = _make_dlte_paths(tmp_path)
    stages = build_command(
        model="bert", task="dlte_alignment",
        dataset="dlte_v1", setting="gte",     # table_model = gte, col_model = bert
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    cmd = stages[0]
    assert _flag_to_value(cmd, "--models")      == "bert"
    assert _flag_to_value(cmd, "--table_model") == "gte"


def test_dlte_merge_build_command_produces_expected_cli(tmp_path):
    """dlte_merge dispatch invokes step10_row_matching with --row_models set
    to model and --col_models from setting (or model when diagonal). The
    diagonal default means model fills the col / row / table axes.
    """
    embeddings, labels, emb_root, project_root = _make_dlte_paths(tmp_path)

    stages = build_command(
        model="bert", task="dlte_merge",
        dataset="dlte_v1", setting="diagonal",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.dlte.scripts.step10_row_matching"]
    assert _flag_to_value(cmd, "--row_models")      == "bert"
    assert _flag_to_value(cmd, "--col_models")      == "bert"
    assert _flag_to_value(cmd, "--embeddings_root") == str(emb_root)
    assert _flag_to_value(cmd, "--project_root")    == str(project_root)
    # Diagonal -> no --table_model.
    assert "--table_model" not in cmd


def test_dlte_merge_setting_packs_col_and_table_models(tmp_path):
    """Setting = ``<col>__<table>`` carries both auxiliary axes for
    dlte_merge: col_model and table_model. The runner accepts only single-
    string nargs+ values, so we pass them as single tokens here.
    """
    embeddings, labels, *_ = _make_dlte_paths(tmp_path)
    stages = build_command(
        model="tabicl", task="dlte_merge",
        dataset="dlte_v1", setting="bert__gte",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    cmd = stages[0]
    assert _flag_to_value(cmd, "--row_models") == "tabicl"
    assert _flag_to_value(cmd, "--col_models") == "bert"
    assert _flag_to_value(cmd, "--table_model") == "gte"


def test_dlte_merge_rejects_malformed_setting(tmp_path):
    """An unexpected setting token count must raise SettingError so the user
    sees a clear message about the dlte_merge axis-packing convention.
    """
    embeddings, labels, *_ = _make_dlte_paths(tmp_path)
    with pytest.raises(SettingError, match="dlte setting"):
        build_command(
            model="bert", task="dlte_merge",
            dataset="dlte_v1", setting="only_one_token",
            probe=None, seed=42,
            results_dir=tmp_path,
            embeddings_path=embeddings,
            labels_path=labels,
            configs_root=tmp_path,
        )


def test_dlte_does_not_require_probe(tmp_path):
    """DLTE has no probe-head axis (mirrors semantic_parsing exemption)."""
    embeddings, labels, *_ = _make_dlte_paths(tmp_path)
    for task in ("dlte_retrieval", "dlte_alignment", "dlte_merge"):
        stages = build_command(
            model="bert", task=task,
            dataset="dlte_v1", setting="diagonal",
            probe=None, seed=42,
            results_dir=tmp_path,
            embeddings_path=embeddings,
            labels_path=labels,
            configs_root=tmp_path,
        )
        assert len(stages) == 1


def test_dlte_validity_uses_granularity_table_col_row(tmp_path):
    """dlte_retrieval requires 'table' granularity, dlte_alignment 'col',
    dlte_merge 'row'. Models that lack the granularity must fail is_valid_cell.
    """
    from trl_bench.registry import is_valid_cell
    # bert exports all three (table / col / row) -> all three dlte tasks valid.
    assert is_valid_cell("bert", "dlte_retrieval") is True
    assert is_valid_cell("bert", "dlte_alignment") is True
    assert is_valid_cell("bert", "dlte_merge")     is True
    # starmie exports col+table (the paper reports its table_subset/table_retrieval,
    # and DLTE uses its aggregated table embeddings for retrieval) -> retrieval +
    # alignment valid, merge (needs 'row') invalid.
    assert is_valid_cell("starmie", "dlte_retrieval") is True
    assert is_valid_cell("starmie", "dlte_alignment") is True
    assert is_valid_cell("starmie", "dlte_merge")     is False
    # tabicl is row-only -> only merge valid.
    assert is_valid_cell("tabicl", "dlte_retrieval") is False
    assert is_valid_cell("tabicl", "dlte_alignment") is False
    assert is_valid_cell("tabicl", "dlte_merge")     is True


# == Deterministic training-free tasks: dispatch coverage ====================
# These tasks have no probe head, no seed dimension, and the runners are
# training-free (column_clustering / schema_matching / union_search /
# join_search). Each test asserts the runner module + the per-cell CLI knobs
# the dispatcher emits, mirroring the canonical .sbatch invocations
# preserved at <reference>/results/round{4,5}/downstream/<task>/<cell>.sbatch.


def _make_deterministic_paths(tmp_path, dataset_name):
    """Build a plausible (embeddings, labels) pair for deterministic-task
    dispatch tests. The dispatcher derives auxiliary file paths
    (pairs.json, ground_truth.csv, groundtruth.pickle, queries.csv) from
    ``labels_path.parent``.
    """
    embeddings = tmp_path / "embeddings" / "column" / "bert" / f"{dataset_name}.pkl"
    embeddings.parent.mkdir(parents=True)
    embeddings.write_bytes(b"")
    labels = tmp_path / "data" / dataset_name / "labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text("{}")
    return embeddings, labels


def test_column_clustering_build_command_produces_expected_cli(tmp_path):
    """column_clustering dispatch invokes evaluate_clustering with reference
    hyperparameters {k=20, target_avg_size=50, batch_size=4096} and the
    staged dataset directory as --dataset.
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, "sato")
    stages = build_command(
        model="bert", task="column_clustering",
        dataset="sato", setting="column_mean",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.column_clustering.evaluate_clustering"]
    assert _flag_to_value(cmd, "--embeddings")      == str(embeddings)
    # --dataset points at the staged directory containing all.csv (parent of
    # labels_path by convention).
    assert _flag_to_value(cmd, "--dataset")         == str(labels.parent)
    assert _flag_to_value(cmd, "--k")               == "20"
    assert _flag_to_value(cmd, "--target_avg_size") == "50"
    assert _flag_to_value(cmd, "--batch_size")      == "4096"
    # No --head_type — column_clustering is training-free, no probe axis.
    assert "--head_type" not in cmd


def test_schema_matching_build_command_produces_expected_cli(tmp_path):
    """schema_matching dispatch invokes run_schema_matching with reference
    hyperparameters {matching_strategy=hungarian, threshold=0.0} and the
    aux paths derived from labels_path.parent.
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, "valentine")
    stages = build_command(
        model="turl", task="schema_matching",
        dataset="valentine", setting="column_mean",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.schema_matching.run_schema_matching"]
    assert _flag_to_value(cmd, "--embeddings")        == str(embeddings)
    assert _flag_to_value(cmd, "--pairs")             == str(labels.parent / "pairs.json")
    assert _flag_to_value(cmd, "--ground_truth")      == str(labels.parent / "ground_truth.csv")
    assert _flag_to_value(cmd, "--tables_dir")        == str(labels.parent / "tables")
    assert _flag_to_value(cmd, "--matching_strategy") == "hungarian"
    assert _flag_to_value(cmd, "--threshold")         == "0.0"
    # --output_dir set to the deterministic flat-layout output dir.
    out_dir = _flag_to_value(cmd, "--output_dir")
    assert out_dir is not None
    assert out_dir.endswith("/evaluation/schema_matching/turl")


def test_union_search_build_command_produces_expected_cli(tmp_path):
    """union_search dispatch invokes run_search with reference hyperparameters
    {method=linear, K=10, threshold=0.7, ef=100, N=100} and the same column
    pickle for query and datalake (canonical convention).
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, "santos")
    stages = build_command(
        model="gte", task="union_search",
        dataset="santos", setting="column_mean",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.union_search.run_search"]
    # Same pickle for query and datalake.
    assert _flag_to_value(cmd, "--query_embeddings")    == str(embeddings)
    assert _flag_to_value(cmd, "--datalake_embeddings") == str(embeddings)
    assert _flag_to_value(cmd, "--groundtruth")         == str(labels.parent / "groundtruth.pickle")
    # Hyperparameters mirror the reference.
    assert _flag_to_value(cmd, "--method")    == "linear"
    assert _flag_to_value(cmd, "--K")         == "10"
    assert _flag_to_value(cmd, "--threshold") == "0.7"
    assert _flag_to_value(cmd, "--ef")        == "100"
    assert _flag_to_value(cmd, "--N")         == "100"
    # No --head_type — training-free task.
    assert "--head_type" not in cmd


@pytest.mark.parametrize("dataset", ["tus", "tus_hard"])
def test_union_search_build_command_tus_uses_hnsw_per_dataset_config(tmp_path, dataset):
    """tus / tus_hard are large, low-similarity benchmarks: the canonical
    generated .sbatch runs them with a *different* per-dataset
    search config than the linear/K=10 default used by santos/ugen::

        --method hnsw --K 60 --threshold 0.1 --ef 100 --N 500

    tus / tus_hard need the HNSW operating point: the linear/K=10 default has a
    low recall ceiling (~0.06), so the dispatcher overrides per-dataset for
    these two.
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, dataset)
    stages = build_command(
        model="bert", task="union_search",
        dataset=dataset, setting="column_mean",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    cmd = stages[0]
    assert _flag_to_value(cmd, "--method")    == "hnsw"
    assert _flag_to_value(cmd, "--K")         == "60"
    assert _flag_to_value(cmd, "--threshold") == "0.1"
    assert _flag_to_value(cmd, "--ef")        == "100"
    assert _flag_to_value(cmd, "--N")         == "500"


def test_join_search_learned_build_command_produces_expected_cli(tmp_path):
    """join_search_learned is the learned-projection (InfoNCE) variant of
    join_search: a seeded training task with its own runner
    (``run_learned_search.py``). The dispatcher must produce the canonical CLI
    (num_layers=1, batch_size=512, max_epochs=10, lr=1e-3, k=50) and derive
    query_list/ground_truth/split_dir from labels_path.parent.
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, "opendata_can")
    stages = build_command(
        model="bert", task="join_search_learned",
        dataset="opendata_can", setting="cls_embedding",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.join_search.run_learned_search"]
    assert _flag_to_value(cmd, "--query_emb")    == str(embeddings)
    assert _flag_to_value(cmd, "--datalake_emb") == str(embeddings)
    assert _flag_to_value(cmd, "--query_list")   == str(labels.parent / "queries.csv")
    assert _flag_to_value(cmd, "--ground_truth") == str(labels.parent / "ground_truth.csv")
    assert _flag_to_value(cmd, "--split_dir")    == str(labels.parent / "splits" / "join_search")
    assert _flag_to_value(cmd, "--num_layers")   == "1"
    assert _flag_to_value(cmd, "--k")            == "50"
    assert _flag_to_value(cmd, "--seed")         == "42"
    # training-free head -> no --head_type / --probe
    assert "--head_type" not in cmd


def _make_table_retrieval_paths(tmp_path, model, dataset):
    emb = tmp_path / "embeddings" / "table" / model / f"{dataset}.pkl"
    emb.parent.mkdir(parents=True); emb.write_bytes(b"")
    lab = tmp_path / "data" / dataset / "labels.json"
    lab.parent.mkdir(parents=True); lab.write_text("{}")
    return emb, lab


def test_table_retrieval_build_command_model_only(tmp_path):
    """table_retrieval is a cross-encoder: setting=pooling, probe=<encoder>[_modelonly].
    model_only -> 2 stages (train, eval) using the raw model table embeddings +
    the QUERY-ENCODER query embeddings + the canonical training config.
    """
    emb, lab = _make_table_retrieval_paths(tmp_path, "bert", "nq_tables")
    stages = build_command(
        model="bert", task="table_retrieval", dataset="nq_tables",
        setting="cls_embedding", probe="mpnet_modelonly", seed=42,
        results_dir=tmp_path, embeddings_path=emb, labels_path=lab,
        configs_root=tmp_path,
    )
    assert len(stages) == 2          # train, eval (no hybrid pre-stage)
    train = stages[0]
    assert train[:3] == [sys.executable, "-m", "trl_bench.tasks.table_retrieval.train"]
    # canonical config
    assert _flag_to_value(train, "--num_layers")      == "1"
    assert "--no_adapter" in train
    assert _flag_to_value(train, "--similarity_fn")   == "cosine"
    assert _flag_to_value(train, "--temperature")     == "0.1"
    assert _flag_to_value(train, "--refinement_rounds") == "2"
    # query embeddings come from the ENCODER (mpnet), not the table model
    assert _flag_to_value(train, "--train_query_embeddings").endswith(
        "table_retrieval/mpnet/queries_train.pkl")
    # model_only -> raw model table embeddings
    assert _flag_to_value(train, "--table_embeddings") == str(emb)
    assert _flag_to_value(train, "--embedding_variant") == "cls_embedding"


def test_table_retrieval_build_command_hybrid(tmp_path):
    """hybrid (probe='mpnet') -> 3 stages: create_hybrid, train, eval; the train
    stage consumes the combined <model>_<encoder>_hybrid embeddings.
    """
    emb, lab = _make_table_retrieval_paths(tmp_path, "bert", "nq_tables")
    stages = build_command(
        model="bert", task="table_retrieval", dataset="nq_tables",
        setting="token_mean", probe="mpnet", seed=42,
        results_dir=tmp_path, embeddings_path=emb, labels_path=lab,
        configs_root=tmp_path,
    )
    assert len(stages) == 3          # create_hybrid, train, eval
    assert stages[0][:3] == [sys.executable, "-m",
                             "trl_bench.tasks.table_retrieval.create_hybrid_embeddings"]
    assert _flag_to_value(stages[0], "--combination_method") == "concat"
    # train consumes the hybrid embeddings
    assert "bert_mpnet_hybrid" in _flag_to_value(stages[1], "--table_embeddings")


def test_join_search_build_command_produces_expected_cli(tmp_path):
    """join_search dispatch invokes run_search_and_evaluate with reference
    hyperparameter {k=50}, same column pickle for query and datalake (canonical
    convention), and aux paths derived from labels_path.parent.
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, "opendata")
    stages = build_command(
        model="tabbie", task="join_search",
        dataset="opendata", setting="column_mean",
        probe=None, seed=42,
        results_dir=tmp_path,
        embeddings_path=embeddings,
        labels_path=labels,
        configs_root=tmp_path,
    )
    assert isinstance(stages, list) and len(stages) == 1
    cmd = stages[0]
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.tasks.join_search.run_search_and_evaluate"]
    assert _flag_to_value(cmd, "--query_emb")    == str(embeddings)
    assert _flag_to_value(cmd, "--datalake_emb") == str(embeddings)
    assert _flag_to_value(cmd, "--query_list")   == str(labels.parent / "queries.csv")
    assert _flag_to_value(cmd, "--ground_truth") == str(labels.parent / "ground_truth.csv")
    assert _flag_to_value(cmd, "--k")            == "50"
    # --output goes to a CSV inside the deterministic flat-layout output dir.
    output = _flag_to_value(cmd, "--output")
    assert output is not None
    assert output.endswith("/evaluation/join_search/tabbie/results.csv")


def test_deterministic_tasks_do_not_require_probe(tmp_path):
    """The 4 deterministic tasks accept ``probe=None`` without raising
    (they have no probe-head axis). Mirrors the semantic_parsing / DLTE
    exemption pattern.
    """
    embeddings, labels = _make_deterministic_paths(tmp_path, "x")
    for task in ("column_clustering", "schema_matching",
                 "union_search", "join_search"):
        stages = build_command(
            model="bert", task=task,
            dataset="x", setting="column_mean",
            probe=None, seed=42,
            results_dir=tmp_path,
            embeddings_path=embeddings,
            labels_path=labels,
            configs_root=tmp_path,
        )
        assert isinstance(stages, list) and len(stages) >= 1


def test_eighteen_paper_grid_tasks_dispatch(tmp_path):
    """End-to-end sanity check: every task in ``_TASK_GRANULARITIES`` resolves
    to a dispatch path that does NOT raise the 'not supported' SettingError
    (i.e. all 18 paper-grid tasks are now wired). For tasks that need
    additional auxiliary files (DLTE / row_prediction / retrieval / etc.) we
    pass plausible placeholder paths; the assertion is that ``build_command``
    returns a non-empty stage list.
    """
    from trl_bench.registry import _TASK_GRANULARITIES

    placeholder_emb = tmp_path / "emb.pkl"
    placeholder_emb.write_bytes(b"")
    placeholder_labels = tmp_path / "labels.json"
    placeholder_labels.write_text("{}")

    # Map task -> a model with the right granularity. ``bert`` exports all 3,
    # but some tasks restrict the model differently — pick a model that's a
    # valid cell for each. For granularity 'col': bert; 'table': bert; 'row':
    # bert (it exports row too). semantic_parsing uses bert + mpnet setting.
    for task, granularity in _TASK_GRANULARITIES.items():
        # All 18 grid tasks should be reachable from bert (col + row + table).
        if not is_valid_cell("bert", task):
            continue
        # Pick a setting consistent with task family:
        #   - DLTE/deterministic: setting="diagonal"/"column_mean"
        #   - row_prediction: setting=label_column placeholder
        #   - semantic_parsing: setting=query_encoder model
        #   - others: setting="cls_embedding" (a valid pair-task setting)
        if task.startswith("dlte_"):
            setting = "diagonal"
        elif task in DETERMINISTIC_TASKS:
            setting = "column_mean"
        elif task == "row_prediction":
            setting = "log_total_assets"
        elif task == "semantic_parsing":
            setting = "mpnet"
        else:
            setting = "cls_embedding"

        # Some tasks need probe (PROBE_TASKS minus the exempt ones).
        from trl_bench.registry import PROBE_TASKS
        needs_probe = task in PROBE_TASKS and task not in (
            "semantic_parsing",
            "dlte_retrieval", "dlte_alignment", "dlte_merge",
        )
        probe = "linear" if needs_probe else None

        stages = build_command(
            model="bert", task=task,
            dataset="ds", setting=setting,
            probe=probe, seed=42,
            results_dir=tmp_path,
            embeddings_path=placeholder_emb,
            labels_path=placeholder_labels,
            configs_root=tmp_path,
        )
        assert isinstance(stages, list) and len(stages) >= 1, (
            f"task {task!r} did not return a non-empty stage list"
        )
