"""Tests for the Stage-0 HF -> on-disk stager.

Hermetic unit tests use a fake ``datasets`` library to verify the per-task
field mapping and layout.
"""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path

import pytest

import trl_bench.data.stage as stage_mod


@pytest.fixture(autouse=True)
def _clear_tables_cache():
    """Tables-config dicts are memoized per-process; reset between tests so
    one test's fake doesn't leak into the next."""
    stage_mod._CTBENCH_TABLES_CACHE.clear()
    yield
    stage_mod._CTBENCH_TABLES_CACHE.clear()


# == hermetic tests using injected fake load_dataset =========================

def _fake_dataset(rows):
    """Return an iterable that mimics the subset of HF dataset API used."""
    class _DS:
        def __iter__(self): return iter(rows)
        def __len__(self):  return len(rows)
    return _DS()


def test_stage_ctbench_join_classification_writes_labels_and_tables(tmp_path, monkeypatch):
    """Verify Stage-0 writes labels.json with the source schema and CSVs."""
    train_rows = [{
        "table_a_id": "positive/dir1/t_a1.csv", "table_a_csv": "x,y\n1,2",
        "table_b_id": "positive/dir1/t_b1.csv", "table_b_csv": "x,y\n3,4",
        "label": 1, "join_col_a": "x", "join_col_b": "x",
    }]
    valid_rows = [{
        "table_a_id": "positive/dir2/t_a2.csv", "table_a_csv": "p,q\n5,6",
        "table_b_id": "positive/dir2/t_b2.csv", "table_b_csv": "p,q\n7,8",
        "label": 0, "join_col_a": "p", "join_col_b": "p",
    }]
    test_rows = [{
        "table_a_id": "negative/dir3/t_a3.csv", "table_a_csv": "u,v\n9,10",
        "table_b_id": "negative/dir3/t_b3.csv", "table_b_csv": "u,v\n11,12",
        "label": 1, "join_col_a": "u", "join_col_b": "u",
    }]
    by_split = {"train": train_rows, "validation": valid_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-ctbench"
        assert name == "spider_join"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # spider_join also fetches table-disjoint "strict" labels from HF; stub it
    # so this hermetic test stays offline (real fetch + byte-match is @slow).
    def fake_fetch_strict(*, repo, dataset, dst_path, revision=None):
        assert dataset == "spider_join"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text('{"train": [], "valid": [], "test": []}')
    monkeypatch.setattr(stage_mod, "_fetch_labels_strict", fake_fetch_strict, raising=False)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="join_classification", dataset="spider_join",
        data_root=tmp_path,
    )
    assert base == tmp_path / "spider_join" / "spider-join"
    assert (base / ".staged_ok").exists()
    # Strict labels fetched next to labels.json (inner spider-join/ dir).
    assert (base / "labels_strict.json").exists()

    labels = json.loads((base / "labels.json").read_text())
    assert sorted(labels) == ["test", "train", "valid"]
    assert len(labels["train"]) == 1
    assert len(labels["valid"]) == 1
    assert len(labels["test"])  == 1
    entry = labels["train"][0]
    assert entry["table1"] == {"filename": "positive/dir1/t_a1.csv"}
    assert entry["table2"] == {"filename": "positive/dir1/t_b1.csv"}
    assert entry["label"]            == 1
    assert entry["join_col_table1"] == "x"
    assert entry["join_col_table2"] == "x"

    # CSVs flattened to tables_all with basename names.
    assert (base / "tables_all" / "t_a1.csv").read_text() == "x,y\n1,2"
    assert (base / "tables_all" / "t_b1.csv").read_text() == "x,y\n3,4"
    assert (base / "tables_all" / "t_a2.csv").read_text() == "p,q\n5,6"
    assert (base / "tables_all" / "t_a3.csv").read_text() == "u,v\n9,10"


def test_stage_basename_collision_raises(tmp_path, monkeypatch):
    """If two label rows reference the same basename with different content,
    staging should refuse rather than silently overwrite."""
    rows = [{
        "table_a_id": "positive/dirA/dup.csv", "table_a_csv": "content_one",
        "table_b_id": "positive/dirB/other.csv", "table_b_csv": "x",
        "label": 1, "join_col_a": "k", "join_col_b": "k",
    }, {
        "table_a_id": "negative/dirC/dup.csv", "table_a_csv": "content_two",
        "table_b_id": "positive/dirB/other2.csv", "table_b_csv": "y",
        "label": 0, "join_col_a": "k", "join_col_b": "k",
    }]
    by_split = {"train": rows, "validation": [], "test": []}
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda repo, name, split, revision=None: _fake_dataset(by_split[split])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="basename collision"):
        stage_mod.stage_dataset(
            suite="ctbench", task="join_classification", dataset="spider_join",
            data_root=tmp_path,
        )


def test_stage_idempotent_when_sentinel_exists(tmp_path):
    base = tmp_path / "spider_join" / "spider-join"
    base.mkdir(parents=True)
    (base / ".staged_ok").write_text("ok\n")

    # Should not raise (no fake load_dataset) — never calls datasets.
    result = stage_mod.stage_dataset(
        suite="ctbench", task="join_classification", dataset="spider_join",
        data_root=tmp_path,
    )
    assert result == base


def test_stage_rejects_unknown_suite(tmp_path):
    with pytest.raises(ValueError, match="unknown suite"):
        stage_mod.stage_dataset(
            suite="invalid", task="x", dataset="y", data_root=tmp_path,
        )


def test_stage_unported_task_raises_not_implemented(tmp_path):
    # All ctbench task families ARE wired now. To keep this assertion alive,
    # exercise the NotImplementedError that fires for an unrecognized task
    # name (e.g. a typo).
    with pytest.raises(NotImplementedError, match="not supported"):
        stage_mod.stage_dataset(
            suite="ctbench", task="not_a_real_task_family", dataset="wtq",
            data_root=tmp_path,
        )


# == multi-config (separate *_tables config) path ============================

def test_stage_ctbench_multi_config_table_subset_uses_lookup(tmp_path, monkeypatch):
    """Verify the multi-config code path: pair rows carry only ids, CSV
    content is loaded from a separate ``<dataset>_tables`` HF config and
    looked up by table_id (with the documented ``.bz2`` ext strip)."""
    pair_train = [{
        "table_a_id": "foo.csv.part1.csv.bz2",  # tables row id is foo.csv.part1.csv
        "table_b_id": "foo.csv.1.neg.csv.bz2",
        "label": 0.0,  # HF gives float; reference labels.json wants int
        "join_col_a": "", "join_col_b": "",
    }]
    pair_valid = [{
        "table_a_id": "bar.csv.part0.csv.bz2",
        "table_b_id": "bar.csv.0.neg.csv.bz2",
        "label": 1.0,
        "join_col_a": "", "join_col_b": "",
    }]
    pair_test = [{
        "table_a_id": "baz.csv.part2.csv.bz2",
        "table_b_id": "baz.csv.2.neg.csv.bz2",
        "label": 0.0,
        "join_col_a": "", "join_col_b": "",
    }]
    pair_by_split = {"train": pair_train, "validation": pair_valid, "test": pair_test}

    tables_rows = [
        {"table_id": "foo.csv.part1.csv", "csv_text": "x,y\n1,2", "n_rows": 1, "n_cols": 2},
        {"table_id": "foo.csv.1.neg.csv", "csv_text": "x,y\n3,4", "n_rows": 1, "n_cols": 2},
        {"table_id": "bar.csv.part0.csv", "csv_text": "p,q\n5,6", "n_rows": 1, "n_cols": 2},
        {"table_id": "bar.csv.0.neg.csv", "csv_text": "p,q\n7,8", "n_rows": 1, "n_cols": 2},
        {"table_id": "baz.csv.part2.csv", "csv_text": "u,v\n9,10",  "n_rows": 1, "n_cols": 2},
        {"table_id": "baz.csv.2.neg.csv", "csv_text": "u,v\n11,12", "n_rows": 1, "n_cols": 2},
    ]

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-ctbench"
        if name == "ckan_subset":
            return _fake_dataset(pair_by_split[split])
        if name == "ckan_subset_tables":
            assert split == "train"          # tables configs use train-only
            return _fake_dataset(tables_rows)
        raise AssertionError(f"unexpected config name: {name!r}")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="table_subset", dataset="ckan_subset",
        data_root=tmp_path,
    )
    assert base == tmp_path / "ckan_subset"
    assert (base / ".staged_ok").exists()

    labels = json.loads((base / "labels.json").read_text())
    # ckan_subset reference split order is (train, test, valid), not the default.
    assert list(labels) == ["train", "test", "valid"]
    # Label coerced to int (HF gave 0.0, reference expects 0).
    train_entry = labels["train"][0]
    assert train_entry["label"] == 0 and isinstance(train_entry["label"], int)
    assert labels["valid"][0]["label"] == 1 and isinstance(labels["valid"][0]["label"], int)
    # Pair-id round-trip into labels.json: original ``.csv.bz2`` preserved.
    assert train_entry["table1"] == {"filename": "foo.csv.part1.csv.bz2"}
    assert train_entry["table2"] == {"filename": "foo.csv.1.neg.csv.bz2"}
    # No metadata for table_subset (verified field map).
    assert set(train_entry) == {"table1", "table2", "label"}

    # CSV content written under the tables-config basename (``.csv``, no bz2).
    assert (base / "tables_all" / "foo.csv.part1.csv").read_text() == "x,y\n1,2"
    assert (base / "tables_all" / "foo.csv.1.neg.csv").read_text() == "x,y\n3,4"
    assert (base / "tables_all" / "baz.csv.2.neg.csv").read_text() == "u,v\n11,12"

    # labels.json formatting: ckan_subset is the documented single-line outlier.
    blob = (base / "labels.json").read_bytes()
    assert b"\n" not in blob


def test_stage_ctbench_multi_config_missing_table_raises(tmp_path, monkeypatch):
    """If a pair row references a table_id absent from the tables config,
    staging should raise a clear error rather than silently emit empty CSVs."""
    pair_rows = [{
        "table_a_id": "present.csv.bz2", "table_b_id": "missing.csv.bz2",
        "label": 0.0, "join_col_a": "", "join_col_b": "",
    }]
    tables_rows = [
        {"table_id": "present.csv", "csv_text": "x\n1", "n_rows": 1, "n_cols": 1},
    ]
    by_split = {"train": pair_rows, "validation": [], "test": []}

    def fake_load(repo, name, split, revision=None):
        if name == "ckan_subset":
            return _fake_dataset(by_split[split])
        if name == "ckan_subset_tables":
            return _fake_dataset(tables_rows)
        raise AssertionError(name)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="not present in tables config"):
        stage_mod.stage_dataset(
            suite="ctbench", task="table_subset", dataset="ckan_subset",
            data_root=tmp_path,
        )


def test_stage_ctbench_multi_config_memoizes_tables_load(tmp_path, monkeypatch):
    """Each wiki tables config loads only once per process (memoized).
    wiki_union uses wiki_tables_full; wiki_containment now routes to its own
    real-header config wiki_containment_tables (the join_containment fix), so
    neither config is re-loaded redundantly and wiki_containment does NOT pull
    wiki_tables_full."""
    pair_rows = [{
        "table_a_id": "t1.csv", "table_b_id": "t2.csv",
        "label": 1.0, "join_col_a": "", "join_col_b": "",
    }]
    tables_rows = [
        {"table_id": "t1.csv", "csv_text": "a\n1", "n_rows": 1, "n_cols": 1},
        {"table_id": "t2.csv", "csv_text": "b\n2", "n_rows": 1, "n_cols": 1},
    ]

    load_calls: list[tuple[str, str]] = []

    def fake_load(repo, name, split, revision=None):
        load_calls.append((name, split))
        if name in ("wiki_union", "wiki_containment"):
            by_split = {"train": pair_rows, "validation": [], "test": []}
            return _fake_dataset(by_split[split])
        if name in ("wiki_tables_full", "wiki_containment_tables"):
            return _fake_dataset(tables_rows)
        raise AssertionError(name)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    # wiki_union + wiki_containment both fetch strict labels from HF; stub it
    # to keep this memoization test offline.
    monkeypatch.setattr(
        stage_mod, "_fetch_labels_strict",
        lambda *, repo, dataset, dst_path, revision=None: dst_path.write_text("{}"),
        raising=False,
    )

    # Stage wiki_union first.
    stage_mod.stage_dataset(
        suite="ctbench", task="union_classification", dataset="wiki_union",
        data_root=tmp_path,
    )
    # Then wiki_containment — routes to its own real-header config.
    stage_mod.stage_dataset(
        suite="ctbench", task="join_containment", dataset="wiki_containment",
        data_root=tmp_path,
    )

    wiki_tables_loads = [c for c in load_calls if c[0] == "wiki_tables_full"]
    wc_tables_loads = [c for c in load_calls if c[0] == "wiki_containment_tables"]
    assert len(wiki_tables_loads) == 1, (
        f"wiki_tables_full should load once for wiki_union; got {wiki_tables_loads}"
    )
    assert len(wc_tables_loads) == 1, (
        f"wiki_containment_tables should load once for wiki_containment; got {wc_tables_loads}"
    )


# == hermetic tests for CTA / CRA / retrieval stagers ========================

def test_stage_ctbench_cta_sato_produces_train_test_csv(tmp_path, monkeypatch):
    """CTA stager: write ``train.csv`` / ``test.csv`` + ``tables_all/`` from
    HF ``sato`` config rows (verified schema: table_id, column_id, class,
    table_csv, table_columns)."""
    train_rows = [
        {"table_id": "0", "column_id": 0, "class": "country",
         "table_csv": "col0\nUSA\nCAN", "table_columns": ["col0"]},
        {"table_id": "0", "column_id": 1, "class": "city",
         "table_csv": "col0\nUSA\nCAN", "table_columns": ["col0"]},
        {"table_id": "1", "column_id": 0, "class": "person",
         "table_csv": "name\nAlice", "table_columns": ["name"]},
    ]
    test_rows = [
        {"table_id": "2", "column_id": 0, "class": "brand",
         "table_csv": "brand\nFoo,Bar", "table_columns": ["brand"]},
    ]
    by_split = {"train": train_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-ctbench"
        assert name == "sato"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_type_prediction", dataset="sato",
        data_root=tmp_path,
    )
    assert base == tmp_path / "sato"
    assert (base / ".staged_ok").exists()
    assert (base / "labels.json").exists()

    # train.csv: 1 header + 3 data rows
    train_csv_text = (base / "train.csv").read_text().strip().splitlines()
    assert train_csv_text[0] == "table_id,column_id,class"
    assert len(train_csv_text) == 4
    assert train_csv_text[1] == "0,0,country"
    assert train_csv_text[3] == "1,0,person"

    # test.csv: 1 header + 1 data row with a CSV-special-char class
    test_csv_text = (base / "test.csv").read_text().strip().splitlines()
    assert test_csv_text[0] == "table_id,column_id,class"
    assert test_csv_text[1] == "2,0,brand"

    # tables_all materialized once per unique table_id. For numeric IDs the
    # filename gets a 'table_' prefix to match the reference convention:
    # the CTA loader (train_ct_mode4.py:84) reconstructs the lookup key as
    # f'table_{tid}' when pandas reads the labels CSV's table_id column as
    # int, so the bert column extractor (which uses the filename stem as the
    # pickle's table_id) must produce 'table_0'-style keys to match.
    assert (base / "tables_all" / "table_0.csv").read_text() == "col0\nUSA\nCAN"
    assert (base / "tables_all" / "table_1.csv").read_text() == "name\nAlice"
    assert (base / "tables_all" / "table_2.csv").read_text() == "brand\nFoo,Bar"


def test_stage_ctbench_cta_sotab_handles_csv_special_chars(tmp_path, monkeypatch):
    """CTA stager handles class names with commas/quotes by RFC4180 escaping."""
    train_rows = [
        {"table_id": "t1", "column_id": 0, "class": "has,comma",
         "table_csv": '"col0"\n"val"', "table_columns": ['"col0"']},
    ]
    test_rows = [
        {"table_id": "t2", "column_id": 0, "class": 'has"quote',
         "table_csv": '"col0"\n"v2"', "table_columns": ['"col0"']},
    ]
    by_split = {"train": train_rows, "test": test_rows}

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (
        lambda repo, name, split, revision=None: _fake_dataset(by_split[split])
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_type_prediction", dataset="sotab",
        data_root=tmp_path,
    )
    # Quoted-and-escaped comma + quote per RFC 4180
    assert (base / "train.csv").read_text().splitlines()[1] == 't1,0,"has,comma"'
    assert (base / "test.csv").read_text().splitlines()[1] == 't2,0,"has""quote"'


def test_stage_ctbench_cta_rejects_table_id_content_collision(tmp_path, monkeypatch):
    """Two rows with the same table_id but different table_csv must raise."""
    train_rows = [
        {"table_id": "t1", "column_id": 0, "class": "A",
         "table_csv": "x\n1", "table_columns": ["x"]},
        {"table_id": "t1", "column_id": 1, "class": "B",
         "table_csv": "x\n2", "table_columns": ["x"]},  # different content
    ]
    by_split = {"train": train_rows, "test": []}

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (
        lambda repo, name, split, revision=None: _fake_dataset(by_split[split])
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="table_id collision"):
        stage_mod.stage_dataset(
            suite="ctbench", task="column_type_prediction", dataset="sato",
            data_root=tmp_path,
        )


def test_stage_ctbench_cra_wikict_relation_produces_metadata_json(tmp_path, monkeypatch):
    """CRA stager: write per-split ``*_metadata.json`` with decoded
    ``relation_annotations`` field plus per-table CSVs.

    HF wikict_relation row schema (verified 2026-05-19): table_id,
    csv_filename, table_csv, headers, num_columns, num_rows,
    relation_annotations_json (str-encoded list of dicts), ...
    """
    train_annots = [
        {"column_id": 0, "relations": [], "relation_ids": [0, 0]},
        {"column_id": 1, "relations": ["foo.bar"], "relation_ids": [1, 0]},
    ]
    test_annots = [
        {"column_id": 0, "relations": [], "relation_ids": [0]},
    ]
    train_rows = [{
        "table_id": "tbl-train-1",
        "csv_filename": "table_000001_tbl-train-1.csv",
        "table_csv": "h1,h2\na,b\nc,d",
        "headers": ["h1", "h2"],
        "num_columns": 2, "num_rows": 2,
        "relation_annotations_json": json.dumps(train_annots),
    }]
    test_rows = [{
        "table_id": "tbl-test-1",
        "csv_filename": "table_000002_tbl-test-1.csv",
        "table_csv": "h\nval",
        "headers": ["h"],
        "num_columns": 1, "num_rows": 1,
        "relation_annotations_json": json.dumps(test_annots),
    }]
    by_split = {"train": train_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-ctbench"
        assert name == "wikict_relation"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_relation_prediction",
        dataset="wikict_relation", data_root=tmp_path,
    )
    assert base == tmp_path / "wikict_relation"
    assert (base / ".staged_ok").exists()
    assert (base / "labels.json").exists()

    # Per-split metadata JSON layout consumed by csv_relation_pipeline.py
    train_meta = json.loads((base / "train" / "train_metadata.json").read_text())
    assert isinstance(train_meta, list) and len(train_meta) == 1
    assert train_meta[0]["table_id"] == "tbl-train-1"
    assert train_meta[0]["relation_annotations"] == train_annots
    test_meta = json.loads((base / "test" / "test_metadata.json").read_text())
    assert test_meta[0]["table_id"] == "tbl-test-1"
    assert test_meta[0]["relation_annotations"] == test_annots

    # CSV content under tables_all/ keyed by csv_filename
    assert (base / "tables_all" / "table_000001_tbl-train-1.csv").read_text() == "h1,h2\na,b\nc,d"
    assert (base / "tables_all" / "table_000002_tbl-test-1.csv").read_text() == "h\nval"


def test_stage_ctbench_cra_wikict_relation_skips_empty_csv_content(tmp_path, monkeypatch):
    """The CRA wikict_relation stager must not write empty-content CSV files
    into tables_all/. The bert column extractor calls
    ``pd.read_csv(csv_path, nrows=max_rows, dtype=str, engine='python')``
    in src/trl_bench/models/bert/generate_column_embeddings.py:210, which
    raises ``pandas.errors.EmptyDataError`` on zero-byte or whitespace-only
    CSVs. Empirically, ~201/53768 wikict_relation rows in the HF source have
    empty ``table_csv``; left unfiltered they crash Stage-1.

    Also drop the corresponding metadata entry, so downstream consumers
    don't list a table_id with no on-disk CSV.
    """
    normal_annots = [{"column_id": 0, "relations": ["foo"], "relation_ids": [1]}]
    train_rows = [
        # Normal row: must be materialized.
        {
            "table_id": "tbl-normal",
            "csv_filename": "table_000001_tbl-normal.csv",
            "table_csv": "h1,h2\na,b\nc,d",
            "relation_annotations_json": json.dumps(normal_annots),
        },
        # Empty content: must be SKIPPED.
        {
            "table_id": "tbl-empty",
            "csv_filename": "table_000002_tbl-empty.csv",
            "table_csv": "",
            "relation_annotations_json": json.dumps([]),
        },
        # Whitespace-only content: must be SKIPPED.
        {
            "table_id": "tbl-whitespace",
            "csv_filename": "table_000003_tbl-whitespace.csv",
            "table_csv": "  \n  \n",
            "relation_annotations_json": json.dumps([]),
        },
    ]
    test_rows = [
        # A test-split empty row must also be skipped.
        {
            "table_id": "tbl-test-empty",
            "csv_filename": "table_000004_tbl-test-empty.csv",
            "table_csv": "",
            "relation_annotations_json": json.dumps([]),
        },
    ]
    by_split = {"train": train_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_relation_prediction",
        dataset="wikict_relation", data_root=tmp_path,
    )
    tables_dir = base / "tables_all"

    # Normal-content row: materialized.
    assert (tables_dir / "table_000001_tbl-normal.csv").exists(), (
        "Expected normal-content table to be staged"
    )

    # Empty-content + whitespace-only rows: NOT materialized — bert
    # column extractor would crash with EmptyDataError on pd.read_csv.
    assert not (tables_dir / "table_000002_tbl-empty.csv").exists(), (
        "Empty-content CSV must be skipped"
    )
    assert not (tables_dir / "table_000003_tbl-whitespace.csv").exists(), (
        "Whitespace-only CSV must be skipped"
    )
    assert not (tables_dir / "table_000004_tbl-test-empty.csv").exists(), (
        "Empty-content test-split CSV must be skipped"
    )

    # And the corresponding metadata entries must be dropped too, so
    # downstream task iteration doesn't reference a missing CSV.
    train_meta = json.loads((base / "train" / "train_metadata.json").read_text())
    train_tids = {entry["table_id"] for entry in train_meta}
    assert "tbl-normal" in train_tids
    assert "tbl-empty" not in train_tids
    assert "tbl-whitespace" not in train_tids

    test_meta = json.loads((base / "test" / "test_metadata.json").read_text())
    test_tids = {entry["table_id"] for entry in test_meta}
    assert "tbl-test-empty" not in test_tids


def test_stage_ctbench_cra_rejects_unwired_dataset(tmp_path, monkeypatch):
    """CRA stager raises NotImplementedError for unwired datasets, listing the
    wired set (currently ``wikict_relation`` + ``sotab`` as of 2026-05-20)."""
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **k: _fake_dataset([])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    with pytest.raises(NotImplementedError, match="CRA stager for ctbench dataset"):
        stage_mod.stage_dataset(
            suite="ctbench", task="column_relation_prediction",
            dataset="some_other_relation_dataset",
            data_root=tmp_path,
        )


def test_stage_ctbench_cra_sotab_uses_sotab_relation_hf_config(tmp_path, monkeypatch):
    """SOTAB CRA staging loads from the ``sotab_relation`` HF config and
    materializes under ``<dst_root>/sotab_cra/`` (disambiguating from the
    CTA-shape ``<dst_root>/sotab/`` layout)."""
    train_annots = [{"column_id": 0, "relations": ["title"], "relation_ids": [1, 0]}]
    train_rows = [{
        "table_id": "Book_CPA",
        "csv_filename": "Book_CPA.csv",
        "table_csv": "a,b\n1,2\n",
        "relation_annotations_json": json.dumps(train_annots),
    }]
    test_rows = [{
        "table_id": "Book2_CPA",
        "csv_filename": "Book2_CPA.csv",
        "table_csv": "c,d\n3,4\n",
        "relation_annotations_json": json.dumps([{"column_id": 0, "relations": []}]),
    }]
    by_split = {"train": train_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-ctbench"
        # CRA on sotab must resolve to the sotab_relation HF config.
        assert name == "sotab_relation"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_relation_prediction", dataset="sotab",
        data_root=tmp_path,
    )
    # On-disk path is ``sotab_cra`` so the CTA layout under ``sotab/`` can
    # coexist without collision.
    assert base == tmp_path / "sotab_cra"
    assert (base / ".staged_ok").exists()
    assert (base / "train" / "train_metadata.json").exists()
    assert (base / "test" / "test_metadata.json").exists()
    train_meta = json.loads((base / "train" / "train_metadata.json").read_text())
    assert len(train_meta) == 1
    assert train_meta[0]["table_id"] == "Book_CPA"
    assert train_meta[0]["relation_annotations"] == train_annots
    assert (base / "tables_all" / "Book_CPA.csv").read_text() == "a,b\n1,2\n"
    # Sanity: a CTA staging in the same data_root would land at ``sotab/`` —
    # disjoint paths, no collision.
    assert not (tmp_path / "sotab").exists()

    # Dispatcher contract: build_command with user-facing dataset='sotab' +
    # task='column_relation_prediction' must derive --dataset_dir from the
    # CRA labels.json (under sotab_cra/), not the would-be CTA layout.
    from trl_bench.registry import build_command
    emb = tmp_path / "embeddings" / "table" / "bert" / "sotab.pkl"
    emb.parent.mkdir(parents=True, exist_ok=True)
    emb.write_bytes(b"")
    stages = build_command(
        model="bert", task="column_relation_prediction", dataset="sotab",
        setting="cls_embedding", probe="mlp", seed=42,
        results_dir=tmp_path / "results",
        embeddings_path=emb, labels_path=base / "labels.json",
        configs_root=tmp_path,
    )
    cra_cmd = stages[0]
    ds_dir_idx = cra_cmd.index("--dataset_dir")
    assert Path(cra_cmd[ds_dir_idx + 1]) == base


def test_stage_ctbench_cta_and_cra_sotab_can_coexist(tmp_path, monkeypatch):
    """When the same dataset name (``sotab``) has both CTA + CRA layouts, the
    two stagers materialize to disjoint on-disk directories — the second
    invocation must not short-circuit on the first's ``.staged_ok``."""
    # Fake both HF configs.
    cta_rows = [{
        "table_id": "Book_CTA", "column_id": 0, "class": "Date",
        "table_csv": "a,b\n1,2\n",
        "table_columns": ["a", "b"],
    }]
    cra_train_rows = [{
        "table_id": "Book_CPA", "csv_filename": "Book_CPA.csv",
        "table_csv": "a,b\n1,2\n",
        "relation_annotations_json": json.dumps([{"column_id": 0, "relations": []}]),
    }]

    by_split_cta = {"train": cta_rows, "test": []}
    by_split_cra = {"train": cra_train_rows, "test": []}

    def fake_load(repo, name, split, revision=None):
        if name == "sotab":
            return _fake_dataset(by_split_cta[split])
        if name == "sotab_relation":
            return _fake_dataset(by_split_cra[split])
        raise AssertionError(f"unexpected name: {name!r}")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    cta_base = stage_mod.stage_dataset(
        suite="ctbench", task="column_type_prediction", dataset="sotab",
        data_root=tmp_path,
    )
    cra_base = stage_mod.stage_dataset(
        suite="ctbench", task="column_relation_prediction", dataset="sotab",
        data_root=tmp_path,
    )
    assert cta_base == tmp_path / "sotab"
    assert cra_base == tmp_path / "sotab_cra"
    # Both stagings present + intact.
    assert (cta_base / "train.csv").exists()
    assert (cra_base / "train" / "train_metadata.json").exists()


def test_stage_ctbench_retrieval_nq_tables_produces_jsons(tmp_path, monkeypatch):
    """table_retrieval stager: write ``train.json`` + ``dev.json`` (HF
    ``test`` split, which carries dev_*-prefixed question_ids) + a
    ``table_id_to_csv.json`` mapping + per-table CSVs from the
    ``nq_tables_tables`` config (split=train only).

    The mapping should follow the reference shape: keys are canonical
    query-side table_ids (with spaces), values are CSV filenames (with
    underscores + ``.csv`` suffix). See ``build_csv_to_table_id_mapping``
    in ``trl_bench.tasks.table_retrieval.utils.data_utils``.
    """
    # Canonical IDs may contain spaces; CSV filenames are space->underscore.
    train_queries = [
        {"question_id": "train_-108_0_0", "question": "q1",
         "table_id": "List of Resources_BD944C2B",
         "answers": ["Brazil"]},
        {"question_id": "train_x_0_1", "question": "q2",
         "table_id": "Some Table_HASH2", "answers": ["a", "b"]},
    ]
    test_queries = [
        {"question_id": "dev_6330_0_0", "question": "q3",
         "table_id": "Brazos River_8F7B4BA1",
         "answers": ["Llano Estacado", "Gulf of Mexico"]},
    ]
    tables_rows = [
        {"table_id": "List_of_Resources_BD944C2B.csv", "csv_text": "c\nv",
         "n_rows": 1, "n_cols": 1},
        {"table_id": "Brazos_River_8F7B4BA1.csv", "csv_text": "h\nx",
         "n_rows": 1, "n_cols": 1},
        {"table_id": "Some_Table_HASH2.csv", "csv_text": "a\nb",
         "n_rows": 1, "n_cols": 1},
        # An orphan corpus table referenced by no query
        {"table_id": "Orphan_HASH3.csv", "csv_text": "z\n9",
         "n_rows": 1, "n_cols": 1},
    ]

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-ctbench"
        if name == "nq_tables":
            by_split = {"train": train_queries, "test": test_queries}
            return _fake_dataset(by_split[split])
        if name == "nq_tables_tables":
            assert split == "train"
            return _fake_dataset(tables_rows)
        raise AssertionError(f"unexpected name: {name!r}")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # The retrieval stager also fetches the NQ corpus tables.json (a raw-file
    # artifact on HF that preserves original empty headers the lossy
    # nq_tables_tables parquet drops). Stub the fetch so this hermetic test
    # stays offline; the real fetch + byte-match vs the reference is a @slow test.
    def fake_fetch_tables_json(*, repo, dst_path, revision=None):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text('[{"table_id": "x", "title": "x", "header": [""], "rows": [["v"]]}]')
    monkeypatch.setattr(stage_mod, "_fetch_nq_tables_json", fake_fetch_tables_json, raising=False)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="table_retrieval", dataset="nq_tables",
        data_root=tmp_path,
    )
    assert base == tmp_path / "nq_tables"
    assert (base / ".staged_ok").exists()
    assert (base / "labels.json").exists()
    # Corpus tables.json fetched (stubbed above) — required by the retrieval encoder.
    assert (base / "tables.json").exists()

    # Queries side: HF train -> train.json, HF test -> dev.json
    train_json = json.loads((base / "train.json").read_text())
    assert len(train_json) == 2
    assert train_json[0]["question_id"] == "train_-108_0_0"
    assert train_json[0]["table_id"] == "List of Resources_BD944C2B"  # spaces preserved
    assert train_json[0]["answers"] == ["Brazil"]
    dev_json = json.loads((base / "dev.json").read_text())
    assert len(dev_json) == 1
    assert dev_json[0]["question_id"].startswith("dev_")

    # table_id_to_csv.json: canonical (spaces) -> csv filename (underscores)
    tid_map = json.loads((base / "table_id_to_csv.json").read_text())
    assert tid_map["List of Resources_BD944C2B"] == "List_of_Resources_BD944C2B.csv"
    assert tid_map["Brazos River_8F7B4BA1"] == "Brazos_River_8F7B4BA1.csv"
    assert tid_map["Some Table_HASH2"] == "Some_Table_HASH2.csv"
    # Orphan corpus table included under its underscored basename
    assert tid_map["Orphan_HASH3"] == "Orphan_HASH3.csv"

    # Per-table CSVs materialized under tables_all/
    assert (base / "tables_all" / "List_of_Resources_BD944C2B.csv").read_text() == "c\nv"
    assert (base / "tables_all" / "Brazos_River_8F7B4BA1.csv").read_text() == "h\nx"
    assert (base / "tables_all" / "Some_Table_HASH2.csv").read_text() == "a\nb"
    assert (base / "tables_all" / "Orphan_HASH3.csv").read_text() == "z\n9"


def test_stage_ctbench_retrieval_idempotent(tmp_path, monkeypatch):
    """Re-running retrieval staging when ``.staged_ok`` already exists is a
    no-op (doesn't call ``load_dataset`` again)."""
    base = tmp_path / "nq_tables"
    base.mkdir()
    (base / ".staged_ok").write_text("ok\n")

    # No fake load_dataset registered -> would fail if called.
    result = stage_mod.stage_dataset(
        suite="ctbench", task="table_retrieval", dataset="nq_tables",
        data_root=tmp_path,
    )
    assert result == base


# == @slow integration tests for new families ===============================

@pytest.mark.slow
def test_stage_ctbench_sato_from_hf_produces_train_test_csv(tmp_path):
    """Slow: stage ``sato`` from HuggingFace (no fixture) and verify the
    produced layout has the expected files with reasonable row counts."""
    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_type_prediction", dataset="sato",
        data_root=tmp_path,
    )
    assert (base / "train.csv").exists()
    assert (base / "test.csv").exists()
    assert (base / "tables_all").is_dir()
    assert any((base / "tables_all").iterdir())
    # train.csv should have at least a header + many data rows; same for test
    train_lines = (base / "train.csv").read_text().splitlines()
    test_lines  = (base / "test.csv").read_text().splitlines()
    assert train_lines[0] == "table_id,column_id,class"
    assert len(train_lines) > 100
    assert len(test_lines) > 10


@pytest.mark.slow
def test_stage_ctbench_wikict_relation_from_hf_produces_metadata_json(tmp_path):
    """Slow: stage ``wikict_relation`` from HuggingFace and verify per-split
    metadata JSON is well-formed."""
    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_relation_prediction",
        dataset="wikict_relation", data_root=tmp_path,
    )
    train_meta = json.loads((base / "train" / "train_metadata.json").read_text())
    test_meta  = json.loads((base / "test" / "test_metadata.json").read_text())
    assert isinstance(train_meta, list) and len(train_meta) > 0
    assert isinstance(test_meta,  list) and len(test_meta)  > 0
    sample = train_meta[0]
    assert "table_id" in sample
    assert "relation_annotations" in sample
    assert isinstance(sample["relation_annotations"], list)
    if sample["relation_annotations"]:
        col_ann = sample["relation_annotations"][0]
        assert "column_id" in col_ann
        assert "relation_ids" in col_ann


@pytest.mark.slow
def test_stage_ctbench_nq_tables_from_hf_produces_retrieval_jsons(tmp_path):
    """Slow: stage ``nq_tables`` from HuggingFace and verify the retrieval
    layout has train/dev JSONs + table_id_mapping + per-table CSVs."""
    base = stage_mod.stage_dataset(
        suite="ctbench", task="table_retrieval", dataset="nq_tables",
        data_root=tmp_path,
    )
    train_json = json.loads((base / "train.json").read_text())
    dev_json   = json.loads((base / "dev.json").read_text())
    tid_map    = json.loads((base / "table_id_to_csv.json").read_text())
    assert isinstance(train_json, list) and len(train_json) > 0
    assert isinstance(dev_json,   list) and len(dev_json)   > 0
    assert isinstance(tid_map, dict) and len(tid_map) > 0
    # Each question entry has the required keys
    sample_q = train_json[0]
    for key in ("question_id", "question", "table_id", "answers"):
        assert key in sample_q
    # The dev split contains dev_*-prefixed question_ids per HF schema
    assert any(q["question_id"].startswith("dev_") for q in dev_json)
    # tables_all/ has at least one CSV
    assert any((base / "tables_all").iterdir())


@pytest.mark.slow
def test_stage_ctbench_column_clustering_sato_from_hf(tmp_path):
    """Slow: stage ``sato`` for column_clustering from HF and verify
    ``all.csv`` (train+test concat) + ``tables_all/`` are produced.

    Reference counts (HF sato, 2026-05-20):
      train rows = 96,451; test rows = 24,158; total = 120,609.
    """
    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_clustering", dataset="sato",
        data_root=tmp_path,
    )
    assert (base / "all.csv").exists()
    assert (base / "tables_all").is_dir()
    lines = (base / "all.csv").read_text().splitlines()
    assert lines[0] == "table_id,column_id,class"
    assert len(lines) == 1 + 120_609  # header + train + test
    assert len(list((base / "tables_all").iterdir())) > 1_000


@pytest.mark.slow
def test_stage_ctbench_schema_matching_valentine_from_hf(tmp_path):
    """Slow: stage ``valentine`` for schema_matching from HF and verify
    ``pairs.json`` (550 unique pairs) + ``ground_truth.csv`` (8,681 GT
    correspondences) + ``tables/`` (1,098 unique CSVs) are produced."""
    base = stage_mod.stage_dataset(
        suite="ctbench", task="schema_matching", dataset="valentine",
        data_root=tmp_path,
    )
    pairs = json.loads((base / "pairs.json").read_text())
    assert isinstance(pairs, list) and len(pairs) == 550
    sample = pairs[0]
    for key in ("pair_id", "source", "noise_type", "table_a", "table_b"):
        assert key in sample
    gt_lines = (base / "ground_truth.csv").read_text().splitlines()
    assert gt_lines[0] == (
        "pair_id,source,noise_type,noise_param,table_a,table_b,column_a,column_b"
    )
    assert len(gt_lines) == 1 + 8_681  # header + GT correspondences
    # Tables materialized
    assert (base / "tables").is_dir()
    assert len(list((base / "tables").iterdir())) > 100


@pytest.mark.slow
def test_stage_ctbench_union_search_santos_from_hf(tmp_path):
    """Slow: stage ``santos`` for union_search from HF and verify
    ``groundtruth.pickle`` ({query_id -> unionable_with}) + tables_all/
    are produced.

    Reference counts (HF santos, 2026-05-20): 50 query tables, 550 datalake
    tables; queries are a subset of the datalake (overlap = 50), so the
    union materialized under tables_all/ is 550 unique CSVs."""
    import pickle as _pickle
    base = stage_mod.stage_dataset(
        suite="ctbench", task="union_search", dataset="santos",
        data_root=tmp_path,
    )
    gt = _pickle.loads((base / "groundtruth.pickle").read_bytes())
    assert isinstance(gt, dict) and len(gt) == 50  # santos has 50 queries
    sample_query = next(iter(gt.keys()))
    sample_gt = gt[sample_query]
    assert isinstance(sample_gt, list) and len(sample_gt) > 0
    # Queries are a subset of the datalake; the union deduplicates to 550.
    n_tables = len(list((base / "tables_all").iterdir()))
    assert n_tables == 550, f"expected 550 unique tables (queries∪datalake), got {n_tables}"


@pytest.mark.slow
def test_stage_ctbench_join_search_opendata_can_from_hf(tmp_path):
    """Slow: stage ``opendata_can`` for join_search from HF and verify
    ``queries.csv`` + ``ground_truth.csv`` (task='join' filter) +
    tables_all/ are produced.

    This is the SMALLEST opendata variant (~4K tables vs ~13K for the
    main config), kept fast enough for routine @slow CI."""
    base = stage_mod.stage_dataset(
        suite="ctbench", task="join_search", dataset="opendata_can",
        data_root=tmp_path,
    )
    queries_lines = (base / "queries.csv").read_text().splitlines()
    assert queries_lines[0] == "query_table,query_column"
    assert len(queries_lines) > 100   # ~thousands of queries; lower bound

    gt_lines = (base / "ground_truth.csv").read_text().splitlines()
    assert gt_lines[0] == (
        "query_table,candidate_table,query_column,candidate_column"
    )
    assert len(gt_lines) > 1_000      # opendata_can has thousands of GT pairs

    assert (base / "tables_all").is_dir()
    assert len(list((base / "tables_all").iterdir())) > 100


# == ctbench: column_clustering / schema_matching / union_search / join_search

def test_stage_ctbench_column_clustering_sato(tmp_path, monkeypatch):
    """column_clustering stager: write ``all.csv`` (concat of HF sato train +
    test splits) + ``tables_all/`` from per-row table_csv content."""
    train_rows = [
        {"table_id": "0", "column_id": 0, "class": "country",
         "table_csv": "c0\nUSA"},
        {"table_id": "0", "column_id": 1, "class": "city",
         "table_csv": "c0\nUSA"},
        {"table_id": "1", "column_id": 0, "class": "person",
         "table_csv": "name\nAlice"},
    ]
    test_rows = [
        {"table_id": "2", "column_id": 0, "class": "brand",
         "table_csv": "brand\nFoo,Bar"},
    ]
    by_split = {"train": train_rows, "test": test_rows}

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (
        lambda repo, name, split, revision=None: _fake_dataset(by_split[split])
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="column_clustering", dataset="sato",
        data_root=tmp_path,
    )
    assert base == tmp_path / "sato"
    assert (base / ".staged_ok_clustering").exists()
    # all.csv: header + 4 data rows (train then test order)
    all_lines = (base / "all.csv").read_text().splitlines()
    assert all_lines[0] == "table_id,column_id,class"
    assert all_lines[1] == "0,0,country"
    assert all_lines[3] == "1,0,person"
    assert all_lines[4] == "2,0,brand"
    # Tables materialized once per unique table_id
    assert (base / "tables_all" / "0.csv").read_text() == "c0\nUSA"
    assert (base / "tables_all" / "2.csv").read_text() == "brand\nFoo,Bar"


def test_stage_ctbench_column_clustering_rejects_unknown_dataset(tmp_path, monkeypatch):
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (
        lambda *a, **kw: _fake_dataset([])
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    with pytest.raises(NotImplementedError, match="column_clustering stager"):
        stage_mod.stage_dataset(
            suite="ctbench", task="column_clustering", dataset="wikict_relation",
            data_root=tmp_path,
        )


def test_stage_ctbench_cta_then_clustering_sotab_coexist(tmp_path, monkeypatch):
    """CTA and column_clustering share ``data/<ds>/`` + the same per-column
    embeddings for sato/sotab (sotab's string table_ids -> identical
    ``tables_all/`` filenames). Staging clustering AFTER CTA must NOT
    short-circuit on CTA's ``.staged_ok`` sentinel: it must still emit
    ``all.csv`` alongside CTA's ``train.csv``/``test.csv``, and the merged
    ``labels.json`` must list both task families' artifacts.

    Regression guard for the shared-sentinel bug: pre-fix, clustering reused
    the same ``.staged_ok`` as CTA, so once CTA staged sotab the clustering
    dispatch returned early and ``all.csv`` was never produced (the paper
    evaluates clustering on BOTH sato and SOTAB, appendix Table:
    "Column Clustering ... NMI ... SATO, SOTAB").
    """
    rows_train = [
        {"table_id": "Book_a_CTA", "column_id": 0, "class": "country",
         "table_csv": "c0\nUSA", "table_columns": ["c0"]},
        {"table_id": "Book_b_CTA", "column_id": 0, "class": "person",
         "table_csv": "name\nAlice", "table_columns": ["name"]},
    ]
    rows_test = [
        {"table_id": "Book_c_CTA", "column_id": 0, "class": "brand",
         "table_csv": "brand\nFoo", "table_columns": ["brand"]},
    ]
    by_split = {"train": rows_train, "test": rows_test}

    def fake_load(repo, name, split, revision=None):
        assert name == "sotab"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # 1. CTA first: writes train.csv/test.csv + .staged_ok + labels{train,test}.
    cta_base = stage_mod.stage_dataset(
        suite="ctbench", task="column_type_prediction", dataset="sotab",
        data_root=tmp_path,
    )
    assert cta_base == tmp_path / "sotab"
    assert (cta_base / "train.csv").exists()
    assert (cta_base / ".staged_ok").exists()

    # 2. Clustering SECOND: must run (not short-circuit) and emit all.csv.
    clu_base = stage_mod.stage_dataset(
        suite="ctbench", task="column_clustering", dataset="sotab",
        data_root=tmp_path,
    )
    assert clu_base == tmp_path / "sotab"          # same shared directory
    assert (clu_base / "all.csv").exists(), (
        "clustering all.csv must be emitted even though CTA already staged "
        "(shared-sentinel short-circuit regression)"
    )
    # Clustering uses its own sentinel so it neither short-circuits on, nor
    # suppresses, CTA's .staged_ok.
    assert (clu_base / ".staged_ok_clustering").exists()
    assert (clu_base / ".staged_ok").exists()       # CTA's sentinel survives

    # 3. CTA artifacts survive the clustering stage.
    assert (clu_base / "train.csv").exists()
    assert (clu_base / "test.csv").exists()

    # 4. all.csv = header + train(2) + test(1), in (train, test) order.
    all_lines = (clu_base / "all.csv").read_text().splitlines()
    assert all_lines[0] == "table_id,column_id,class"
    assert len(all_lines) == 1 + 3
    assert all_lines[1] == "Book_a_CTA,0,country"
    assert all_lines[3] == "Book_c_CTA,0,brand"

    # 5. labels.json merged: CTA keys preserved, clustering key added.
    labels = json.loads((clu_base / "labels.json").read_text())
    assert labels.get("train") == "train.csv"
    assert labels.get("test") == "test.csv"
    assert labels.get("all") == "all.csv"

    # 6. String table_ids -> shared tables_all filenames (no table_ prefix),
    #    so CTA + clustering reuse the same per-table CSV (and embeddings).
    assert (clu_base / "tables_all" / "Book_a_CTA.csv").read_text() == "c0\nUSA"


def test_stage_ctbench_schema_matching_valentine(tmp_path, monkeypatch):
    """schema_matching stager: write ``pairs.json`` + ``ground_truth.csv`` +
    ``tables/<id>.csv`` from HF valentine rows.

    HF row schema (verified 2026-05-19): pair_id, source, noise_type,
    noise_param, table_a_id, table_a_csv, table_a_columns, table_b_id,
    table_b_csv, table_b_columns, column_a, column_b. One row per GT
    correspondence; pair_id may repeat across rows.
    """
    rows = [
        {
            "pair_id": "p1", "source": "chembl", "noise_type": "joinable",
            "noise_param": "",
            "table_a_id": "a.csv", "table_a_csv": "x\n1",
            "table_a_columns": ["x"],
            "table_b_id": "b.csv", "table_b_csv": "y\n2",
            "table_b_columns": ["y"],
            "column_a": "x", "column_b": "y",
        },
        {
            "pair_id": "p1", "source": "chembl", "noise_type": "joinable",
            "noise_param": "",
            "table_a_id": "a.csv", "table_a_csv": "x\n1",
            "table_a_columns": ["x"],
            "table_b_id": "b.csv", "table_b_csv": "y\n2",
            "table_b_columns": ["y"],
            "column_a": "x2", "column_b": "y2",
        },
        {
            "pair_id": "p2", "source": "wikidata", "noise_type": "unionable",
            "noise_param": "0.5",
            "table_a_id": "c,name.csv", "table_a_csv": "h\nv",
            "table_a_columns": ["h"],
            "table_b_id": "d.csv", "table_b_csv": "g\nw",
            "table_b_columns": ["g"],
            "column_a": "h", "column_b": "g",
        },
    ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (
        lambda repo, name, split, revision=None: _fake_dataset(rows)
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="schema_matching", dataset="valentine",
        data_root=tmp_path,
    )
    assert base == tmp_path / "valentine"
    pairs = json.loads((base / "pairs.json").read_text())
    assert len(pairs) == 2     # p1 dedup'd, p2 unique
    assert pairs[0]["pair_id"] == "p1"
    assert pairs[1]["source"] == "wikidata"

    gt_lines = (base / "ground_truth.csv").read_text().splitlines()
    assert gt_lines[0] == (
        "pair_id,source,noise_type,noise_param,table_a,table_b,column_a,column_b"
    )
    assert len(gt_lines) == 4  # header + 3 GT rows
    # CSV-special char in table_a_id is RFC4180-escaped
    assert '"c,name.csv"' in gt_lines[3]
    # Tables materialized
    assert (base / "tables" / "a.csv").read_text() == "x\n1"
    assert (base / "tables" / "c,name.csv").read_text() == "h\nv"


def test_stage_ctbench_union_search_santos(tmp_path, monkeypatch):
    """union_search stager: write ``groundtruth.pickle`` + ``tables_all/``
    from HF queries+datalake splits. ``unionable_with`` field is the GT."""
    import pickle as _pickle
    queries = [
        {"table_id": "q1.csv", "csv_text": "h\n1", "n_rows": 1, "n_cols": 1,
         "unionable_with": ["q1.csv", "dl1.csv"]},
        {"table_id": "q2.csv", "csv_text": "h\n2", "n_rows": 1, "n_cols": 1,
         "unionable_with": ["q2.csv"]},
    ]
    datalake = [
        {"table_id": "dl1.csv", "csv_text": "h\nx", "n_rows": 1, "n_cols": 1,
         "unionable_with": []},
        {"table_id": "dl2.csv", "csv_text": "h\ny", "n_rows": 1, "n_cols": 1,
         "unionable_with": []},
    ]
    by_split = {"queries": queries, "datalake": datalake}

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (
        lambda repo, name, split, revision=None: _fake_dataset(by_split[split])
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="union_search", dataset="santos",
        data_root=tmp_path,
    )
    assert base == tmp_path / "santos"
    gt = _pickle.loads((base / "groundtruth.pickle").read_bytes())
    assert set(gt.keys()) == {"q1.csv", "q2.csv"}
    assert gt["q1.csv"] == ["q1.csv", "dl1.csv"]
    # All 4 tables materialized (2 query + 2 datalake)
    assert (base / "tables_all" / "dl2.csv").read_text() == "h\ny"
    assert (base / "tables_all" / "q1.csv").exists()


def test_stage_ctbench_union_search_rejects_unwired(tmp_path, monkeypatch):
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (lambda *a, **kw: _fake_dataset([]))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    with pytest.raises(NotImplementedError, match="union_search stager"):
        stage_mod.stage_dataset(
            suite="ctbench", task="union_search", dataset="opendata_main",
            data_root=tmp_path,
        )


def test_stage_ctbench_join_search_opendata_main(tmp_path, monkeypatch):
    """join_search stager: filter HF pair config by task='join', emit
    queries.csv + ground_truth.csv + tables_all/."""
    pair_rows = [
        {"query_table_id": "qt1.csv", "candidate_table_id": "ct1.csv",
         "task": "join", "query_column": "id", "candidate_column": "ref_id"},
        {"query_table_id": "qt1.csv", "candidate_table_id": "ct2.csv",
         "task": "join", "query_column": "id", "candidate_column": "ref"},
        {"query_table_id": "qt2.csv", "candidate_table_id": "ct1.csv",
         "task": "union", "query_column": "name", "candidate_column": "n"},
        {"query_table_id": "qt3.csv", "candidate_table_id": "ct3.csv",
         "task": "join", "query_column": "k", "candidate_column": "k"},
    ]
    tables_rows = [
        {"table_id": "qt1.csv", "csv_text": "id\n1", "n_rows": 1, "n_cols": 1},
        {"table_id": "qt3.csv", "csv_text": "k\nv", "n_rows": 1, "n_cols": 1},
        {"table_id": "ct1.csv", "csv_text": "ref_id\n10", "n_rows": 1, "n_cols": 1},
        {"table_id": "ct2.csv", "csv_text": "ref\n20", "n_rows": 1, "n_cols": 1},
        {"table_id": "ct3.csv", "csv_text": "k\nv", "n_rows": 1, "n_cols": 1},
    ]

    def fake_load(repo, name, split, revision=None):
        if name == "opendata_main":
            return _fake_dataset(pair_rows)
        if name == "opendata_main_tables":
            return _fake_dataset(tables_rows)
        raise AssertionError(f"unexpected name {name!r}")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # The opendata join_search stager also fetches the canonical query-disjoint
    # split from HF (splits/join_search/<variant>/). Stub the fetch with a
    # fixture so this hermetic test stays offline; the real fetch + match
    # vs the reference is covered by a @slow test below.
    def fake_fetch_split(*, repo, variant, dst_dir, revision=None):
        assert variant == "opendata"          # opendata_main -> HF variant 'opendata'
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "train_queries.csv").write_text("query_table,query_column\nqt1.csv,id\n")
        (dst_dir / "test_queries.csv").write_text("query_table,query_column\nqt3.csv,k\n")
        (dst_dir / "test_gt.csv").write_text(
            "query_table,candidate_table,query_column,candidate_column\nqt3.csv,ct3.csv,k,k\n")
        (dst_dir / "split_info.json").write_text("{}")
    monkeypatch.setattr(stage_mod, "_fetch_join_search_split", fake_fetch_split, raising=False)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="join_search", dataset="opendata_main",
        data_root=tmp_path,
    )
    # Canonical split fetched into <base>/splits/join_search/ (stubbed above).
    split_dir = base / "splits" / "join_search"
    assert (split_dir / "train_queries.csv").exists()
    assert (split_dir / "test_queries.csv").exists()
    assert (split_dir / "test_gt.csv").exists()
    assert (split_dir / "split_info.json").exists()

    queries_lines = (base / "queries.csv").read_text().splitlines()
    assert queries_lines[0] == "query_table,query_column"
    # qt1.csv,id once (two GT rows share same query), qt3.csv,k once;
    # qt2.csv is union-only -> excluded.
    assert len(queries_lines) == 3
    assert "qt1.csv,id" in queries_lines
    assert "qt3.csv,k" in queries_lines
    gt_lines = (base / "ground_truth.csv").read_text().splitlines()
    assert gt_lines[0] == (
        "query_table,candidate_table,query_column,candidate_column"
    )
    # 3 join rows
    assert len(gt_lines) == 4
    # Union row excluded
    assert all("qt2.csv" not in line for line in gt_lines[1:])
    # Tables materialized
    assert (base / "tables_all" / "qt1.csv").exists()


# == Stage-0 -> downstream path-contract regression guard =====================
# task_datasets.yaml is BOTH the generator's hard skip-gate AND the source of
# the runner's ${LABELS_PATH}/${QUERY_LIST}/... substitutions
# (generate_downstream_scripts.validate_dataset_for_task + generate_script).
# So each downstream cell's configured path MUST equal where the Stage-0 stager
# actually writes the file; a drift silently skips the cell even though the data
# is present. These guard the rename-only families that were found stale on
# 2026-05-25 (union_search pickle name, join_search flat queries, CRA _cra dir).

def _fake_load_santos(repo, name, split, revision=None):
    data = {
        "queries": [{"table_id": "q1.csv", "csv_text": "h\n1", "n_rows": 1,
                     "n_cols": 1, "unionable_with": ["q1.csv", "dl1.csv"]}],
        "datalake": [{"table_id": "dl1.csv", "csv_text": "h\nx", "n_rows": 1,
                      "n_cols": 1, "unionable_with": []}],
    }
    return _fake_dataset(data[split])


def _fake_load_opendata(repo, name, split, revision=None):
    if name == "opendata_main":
        return _fake_dataset([
            {"query_table_id": "qt1.csv", "candidate_table_id": "ct1.csv",
             "task": "join", "query_column": "id", "candidate_column": "ref_id"},
        ])
    if name == "opendata_main_tables":
        return _fake_dataset([
            {"table_id": "qt1.csv", "csv_text": "id\n1", "n_rows": 1, "n_cols": 1},
            {"table_id": "ct1.csv", "csv_text": "ref_id\n10", "n_rows": 1, "n_cols": 1},
        ])
    raise AssertionError(f"unexpected config name {name!r}")


def _fake_load_sotab_cra(repo, name, split, revision=None):
    assert name == "sotab_relation"
    data = {
        "train": [{"table_id": "Book_CPA", "csv_filename": "Book_CPA.csv",
                   "table_csv": "a,b\n1,2\n",
                   "relation_annotations_json": json.dumps(
                       [{"column_id": 0, "relations": ["title"], "relation_ids": [1, 0]}])}],
        "test": [{"table_id": "Book2_CPA", "csv_filename": "Book2_CPA.csv",
                  "table_csv": "c,d\n3,4\n",
                  "relation_annotations_json": json.dumps(
                      [{"column_id": 0, "relations": []}])}],
    }
    return _fake_dataset(data[split])


@pytest.mark.parametrize("stage_task,downstream_task,dataset,fake_load", [
    ("union_search", "union_search", "santos", _fake_load_santos),
    ("join_search", "join_search", "opendata_main", _fake_load_opendata),
    ("column_relation_prediction", "column_relation_prediction", "sotab",
     _fake_load_sotab_cra),
])
def test_task_datasets_config_path_matches_stager_output(
    stage_task, downstream_task, dataset, fake_load, tmp_path, monkeypatch,
):
    """The downstream config path for each cell must resolve to where the
    Stage-0 stager actually writes the artifact (otherwise the generator skips
    the cell / feeds the runner a bad path). Stage hermetically into <tmp>/data,
    then assert the generator's gate passes against that data_root."""
    import yaml
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    repo_root = Path(__file__).resolve().parent.parent
    data_root = tmp_path / "data"
    stage_mod.stage_dataset(
        suite="ctbench", task=stage_task, dataset=dataset, data_root=data_root,
    )

    # Import the generator (needs slurm/ on sys.path for its sibling imports).
    slurm_dir = repo_root / "slurm"
    if str(slurm_dir) not in sys.path:
        sys.path.insert(0, str(slurm_dir))
    import generate_downstream_scripts as gds

    cfg = yaml.safe_load(
        (slurm_dir / "config/downstream/task_datasets.yaml").read_text()
    )
    ok, msg = gds.validate_dataset_for_task(downstream_task, dataset, tmp_path, cfg)
    assert ok, (
        f"task_datasets.yaml path for {downstream_task}/{dataset} does not match "
        f"the Stage-0 stager output: {msg}"
    )


def test_stage_ctbench_semantic_parsing_remains_not_implemented(tmp_path, monkeypatch):
    """semantic_parsing Stage-0 staging is NOT yet wired (raw HF wtq config
    lacks the MAPO preprocessing artifacts)."""
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = (lambda *a, **kw: _fake_dataset([]))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    with pytest.raises(NotImplementedError, match="semantic_parsing"):
        stage_mod.stage_dataset(
            suite="ctbench", task="semantic_parsing", dataset="wtq",
            data_root=tmp_path,
        )


# == rbench: record_linkage Stage-0 ==========================================

def test_stage_rbench_record_linkage_pair_task(tmp_path, monkeypatch):
    """Hermetic test: HF record_linkage rows filtered by source -> on-disk
    tableA.csv / tableB.csv + labels.json + metadata.json with the canonical
    reference shape."""
    # Two pairs for source=deepmatcher_abt_buy. Each row has decoded JSON
    # for tableA / tableB record content. Both pairs share the same A record
    # to exercise deduplication on side A.
    train_rows = [
        {
            "source": "deepmatcher_abt_buy",
            "family": "deepmatcher_clean",
            "pair_id": "deepmatcher_abt_buy/train/0",
            "table_a_record_json": json.dumps({"name": "Foo", "price": "10"}),
            "table_b_record_json": json.dumps({"name": "Foo2", "price": "11"}),
            "label": 1,
        },
        {
            "source": "deepmatcher_abt_buy",
            "family": "deepmatcher_clean",
            "pair_id": "deepmatcher_abt_buy/train/1",
            "table_a_record_json": json.dumps({"name": "Foo", "price": "10"}),
            "table_b_record_json": json.dumps({"name": "Bar", "price": "99"}),
            "label": 0,
        },
        {
            # Different source - should be filtered out.
            "source": "deepmatcher_amazon_google",
            "family": "deepmatcher_clean",
            "pair_id": "deepmatcher_amazon_google/train/0",
            "table_a_record_json": json.dumps({"x": "1"}),
            "table_b_record_json": json.dumps({"x": "2"}),
            "label": 0,
        },
    ]
    valid_rows = [{
        "source": "deepmatcher_abt_buy", "family": "deepmatcher_clean",
        "pair_id": "deepmatcher_abt_buy/valid/0",
        "table_a_record_json": json.dumps({"name": "Baz", "price": "5"}),
        "table_b_record_json": json.dumps({"name": "Foo2", "price": "11"}),
        "label": 0,
    }]
    test_rows = [{
        "source": "deepmatcher_abt_buy", "family": "deepmatcher_clean",
        "pair_id": "deepmatcher_abt_buy/test/0",
        "table_a_record_json": json.dumps({"name": "Foo", "price": "10"}),
        "table_b_record_json": json.dumps({"name": "Bar", "price": "99"}),
        "label": 1,
    }]
    by_split = {"train": train_rows, "validation": valid_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-rbench"
        assert name == "record_linkage"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="rbench", task="record_linkage", dataset="deepmatcher_abt_buy",
        data_root=tmp_path,
    )
    assert base == tmp_path / "deepmatcher_abt_buy"
    assert (base / ".staged_ok").exists()

    labels = json.loads((base / "labels.json").read_text())
    assert labels["dataset_source"] == "deepmatcher_clean"
    assert labels["sub_dataset"] == "deepmatcher_abt_buy"
    assert len(labels["train"]) == 2   # the third (different source) is filtered
    assert len(labels["valid"]) == 1
    assert len(labels["test"])  == 1

    # Pair entries shape mirrors the reference labels.json.
    entry = labels["train"][0]
    assert entry["table1"]["filename"] == "tableA.csv"
    assert entry["table2"]["filename"] == "tableB.csv"
    assert "row_idx" in entry["table1"] and "row_idx" in entry["table2"]
    assert entry["label"] == 1

    # Deduplication on side A: the two A records "Foo,10" (rows 0+1) and
    # "Baz,5" (valid) and "Foo,10" again (test) collapse to just 2 unique A.
    metadata = json.loads((base / "metadata.json").read_text())
    assert metadata["tableA_rows"] == 2     # Foo + Baz
    assert metadata["tableB_rows"] == 2     # Foo2 + Bar

    # tableA.csv / tableB.csv have header + dedup'd rows
    a_lines = (base / "tables" / "tableA.csv").read_text().splitlines()
    b_lines = (base / "tables" / "tableB.csv").read_text().splitlines()
    assert a_lines[0] == "name,price"   # header from first record
    assert len(a_lines) == 1 + 2
    assert len(b_lines) == 1 + 2

    # Filtering should reject when no rows match the source.
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    with pytest.raises(RuntimeError, match="no rows for source"):
        stage_mod.stage_dataset(
            suite="rbench", task="record_linkage",
            dataset="not_a_real_source",
            data_root=tmp_path,
        )


# == rbench: row_prediction Stage-0 ==========================================

def test_stage_rbench_row_prediction_per_dataset(tmp_path, monkeypatch):
    """Hermetic test: HF row_prediction rows filtered by openml_id -> on-disk
    data.csv + dataset.json + splits.npz + labels.json manifest."""
    import numpy as np  # numpy used by the stager

    ds_metadata = {
        "schema_version": "1.0",
        "data": {"file": "data.csv", "format": "csv",
                 "n_rows": 4, "n_columns": 3},
        "label_columns": ["target"],
        "splits": {"format": "row_indices_npz", "file": "splits.npz",
                   "names": ["test", "train", "val"]},
        "labels": [{"name": "target", "role": "y1",
                    "task_type": "classification"}],
        "dataset_id": "openml:3",
        "dataset_name": "kr-vs-kp",
    }
    meta_str = json.dumps(ds_metadata)

    def _row(oid, ridx, rec, tgt):
        return {
            "openml_id": oid, "dataset_name": "kr-vs-kp", "row_idx": ridx,
            "record_json": json.dumps(rec),
            "targets_json": json.dumps(tgt),
            "target_specs_json": json.dumps([]),
            "dataset_metadata_json": meta_str,
        }

    train_rows = [
        _row(3, 0, {"a": "1", "b": "2"}, {"target": "y"}),
        _row(3, 2, {"a": "3", "b": "4"}, {"target": "n"}),
        # Different openml_id -> filtered out.
        _row(99, 0, {"a": "0", "b": "0"}, {"target": "n"}),
    ]
    valid_rows = [_row(3, 1, {"a": "5", "b": "6"}, {"target": "y"})]
    test_rows  = [_row(3, 3, {"a": "7", "b": "8"}, {"target": "n"})]
    by_split = {"train": train_rows, "validation": valid_rows, "test": test_rows}

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-rbench"
        assert name == "row_prediction"
        return _fake_dataset(by_split[split])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="rbench", task="row_prediction", dataset="openml_3",
        data_root=tmp_path,
    )
    assert base == tmp_path / "openml_3"
    assert (base / ".staged_ok").exists()

    # data.csv has 4 rows + header. Order is sorted by row_idx 0,1,2,3.
    data_lines = (base / "data.csv").read_text().splitlines()
    assert data_lines[0] == "a,b,target"
    assert data_lines[1] == "1,2,y"
    assert data_lines[2] == "5,6,y"
    assert data_lines[3] == "3,4,n"
    assert data_lines[4] == "7,8,n"

    # splits.npz has the three split arrays.
    splits = np.load(str(base / "splits.npz"))
    assert sorted(splits.keys()) == ["test", "train", "val"]
    assert list(splits["train"]) == [0, 2]
    assert list(splits["val"])   == [1]
    assert list(splits["test"])  == [3]

    # dataset.json is the verbatim metadata blob.
    on_disk_meta = json.loads((base / "dataset.json").read_text())
    assert on_disk_meta["label_columns"] == ["target"]
    assert on_disk_meta["dataset_name"] == "kr-vs-kp"

    # labels.json manifest.
    manifest = json.loads((base / "labels.json").read_text())
    assert manifest["data"] == "data.csv"
    assert manifest["splits"] == "splits.npz"
    assert manifest["openml_id"] == 3
    assert manifest["label_columns"] == ["target"]


def test_stage_rbench_row_prediction_rejects_unknown_openml_id(tmp_path, monkeypatch):
    """When no HF rows match the openml_id, staging refuses."""
    def fake_load(repo, name, split, revision=None):
        return _fake_dataset([])
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="no rows for openml_id"):
        stage_mod.stage_dataset(
            suite="rbench", task="row_prediction", dataset="openml_99999",
            data_root=tmp_path,
        )


def test_stage_rbench_row_prediction_rejects_bad_dataset_name(tmp_path):
    """Dataset name must start with 'openml_'."""
    with pytest.raises(ValueError, match="must start with 'openml_'"):
        stage_mod.stage_dataset(
            suite="rbench", task="row_prediction", dataset="adult",
            data_root=tmp_path,
        )


def test_stage_rbench_unknown_task_raises_not_implemented(tmp_path):
    """rbench has only ``record_linkage`` and ``row_prediction`` wired."""
    with pytest.raises(NotImplementedError, match="rbench task"):
        stage_mod.stage_dataset(
            suite="rbench", task="not_a_task", dataset="ds", data_root=tmp_path,
        )


# == dlte stager =============================================================


def test_stage_dlte_writes_manifests_and_csvs(tmp_path, monkeypatch):
    """Verify the dlte stager writes the project-rooted layout the runners
    expect: lake_manifest.jsonl, fragments_manifest.jsonl,
    parents_filtered.jsonl, ground_truth/query_tasks.jsonl, plus per-CSV
    files under queries/tables/ and lake/targets/tables/."""
    manifest_rows = [
        {"record_type": "parent",
         "record_json": json.dumps({"parent_id": "tabfact__t1.html",
                                    "csv_path": "datasets/tabfact/t1.csv",
                                    "n_rows": 5, "n_cols": 3})},
        {"record_type": "fragment",
         "record_json": json.dumps({"table_id": "dlte_v1__tabfact__t1.html__seed__t0__r0",
                                    "parent_id": "tabfact__t1.html",
                                    "split": "train", "noise_tier": 0})},
        {"record_type": "lake_table",
         "record_json": json.dumps({"table_id": "dlte_v1__tabfact__t1.html__union__t0__r0",
                                    "source": "dlte_target",
                                    "parent_id": "tabfact__t1.html",
                                    "split": "train"})},
        {"record_type": "query_task",
         "record_json": json.dumps({"query_table_id": "dlte_v1__tabfact__t1.html__seed__t0__r0",
                                    "parent_id": "tabfact__t1.html",
                                    "split": "train", "noise_tier": 0,
                                    "relevant": []})},
        {"record_type": "split_assignment",
         "record_json": json.dumps({"parent_id": "tabfact__t1.html",
                                    "split": "train"})},
    ]
    lake_rows = [
        {"table_id": "dlte_v1__tabfact__t1.html__seed__t0__r0",
         "kind": "query",
         "csv_text": "col1,col2\n1,2\n3,4\n"},
        {"table_id": "dlte_v1__tabfact__t1.html__union__t0__r0",
         "kind": "lake_table",
         "csv_text": "col1,col2\n5,6\n7,8\n"},
    ]
    table_maps_rows = [
        {"table_id": "dlte_v1__tabfact__t1.html__seed__t0__r0",
         "row_parent_idx_json": json.dumps([0, 1]),
         "col_parent_idx_json": json.dumps([0, 1])},
        {"table_id": "dlte_v1__tabfact__t1.html__union__t0__r0",
         "row_parent_idx_json": json.dumps([2, 3]),
         "col_parent_idx_json": json.dumps([0, 1])},
    ]

    def fake_load(repo, name, split, revision=None):
        assert repo == "logo-lab/trl-dlte"
        if name == "manifests":
            return _fake_dataset(manifest_rows)
        if name == "lake":
            return _fake_dataset(lake_rows)
        if name == "table_maps":
            return _fake_dataset(table_maps_rows)
        raise AssertionError(f"unexpected name: {name!r}")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="dlte", task="dlte_retrieval", dataset="dlte_v1",
        data_root=tmp_path,
    )
    assert base == tmp_path / "dlte_v1"
    assert (base / ".staged_ok").exists()
    # Layout: <base>/datasets/dlte_v1/{manifests,ground_truth,queries,lake}/.
    proj = base / "datasets" / "dlte_v1"
    assert (proj / "manifests" / "lake_manifest.jsonl").exists()
    assert (proj / "manifests" / "fragments_manifest.jsonl").exists()
    assert (proj / "manifests" / "parents_filtered.jsonl").exists()
    assert (proj / "manifests" / "split_assignments.jsonl").exists()
    assert (proj / "ground_truth" / "query_tasks.jsonl").exists()
    # Per-type record counts.
    lake_lines = (proj / "manifests" / "lake_manifest.jsonl").read_text().strip().splitlines()
    assert len(lake_lines) == 1
    parents_lines = (proj / "manifests" / "parents_filtered.jsonl").read_text().strip().splitlines()
    assert len(parents_lines) == 1
    # CSV content: queries vs lake/targets routing.
    qcsv = proj / "queries" / "tables" / "dlte_v1__tabfact__t1.html__seed__t0__r0.csv"
    tcsv = proj / "lake" / "targets" / "tables" / "dlte_v1__tabfact__t1.html__union__t0__r0.csv"
    assert qcsv.exists() and qcsv.read_text() == "col1,col2\n1,2\n3,4\n"
    assert tcsv.exists() and tcsv.read_text() == "col1,col2\n5,6\n7,8\n"
    # table_maps materialized as .npz files (the runners do np.load on these).
    import numpy as np
    tm_seed = proj / "ground_truth" / "table_maps" / "dlte_v1__tabfact__t1.html__seed__t0__r0.npz"
    tm_union = proj / "ground_truth" / "table_maps" / "dlte_v1__tabfact__t1.html__union__t0__r0.npz"
    assert tm_seed.exists() and tm_union.exists()
    d_seed = np.load(tm_seed)
    assert d_seed["row_parent_idx"].tolist() == [0, 1]
    assert d_seed["col_parent_idx"].tolist() == [0, 1]
    assert d_seed["row_parent_idx"].dtype == np.int32
    # Manifest labels.json is written at <base>/datasets/dlte_v1/labels.json
    # (the deeper path) so the dispatcher's 3-parent walk-up to derive
    # --project_root lands on <base>. The shallower <base>/labels.json is
    # NOT written.
    labels = json.loads((proj / "labels.json").read_text())
    assert labels["dataset"] == "dlte_v1"
    assert labels["table_maps_present"] is True
    assert labels["table_maps_count"] == 2
    assert not (base / "labels.json").exists()


def test_stage_dlte_rejects_unknown_dataset(tmp_path):
    """The HF release only publishes dlte_v1; other dataset names raise."""
    with pytest.raises(NotImplementedError, match="dlte stager"):
        stage_mod.stage_dataset(
            suite="dlte", task="dlte_retrieval", dataset="dlte_v0",
            data_root=tmp_path,
        )


def test_stage_dlte_unknown_task_raises_not_implemented(tmp_path):
    """dlte stager only knows the three dlte_* tasks."""
    with pytest.raises(NotImplementedError, match="dlte task"):
        stage_mod.stage_dataset(
            suite="dlte", task="dlte_foo", dataset="dlte_v1", data_root=tmp_path,
        )


def test_stage_dlte_idempotent_when_sentinel_exists(tmp_path):
    """A pre-existing .staged_ok skips work (no HF calls made)."""
    base = tmp_path / "dlte_v1"
    base.mkdir()
    (base / ".staged_ok").write_text("ok\n")
    result = stage_mod.stage_dataset(
        suite="dlte", task="dlte_retrieval", dataset="dlte_v1",
        data_root=tmp_path,
    )
    assert result == base


def test_stage_dlte_layout_matches_dispatcher_project_root_derivation(tmp_path, monkeypatch):
    """The DLTE stager + dispatcher contract: when run.py auto-stages and
    passes the produced labels.json to build_command, the dispatcher's
    derived ``--project_root`` (= ``labels_path.parent.parent.parent``) must
    equal the staged ``base`` directory.

    Without this invariant the runners' ``<project_root>/datasets/dlte_v1/
    manifests/lake_manifest.jsonl`` lookup would land on the wrong directory
    (regression caught during initial wiring).
    """
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda repo, name, split, revision=None: _fake_dataset([])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    base = stage_mod.stage_dataset(
        suite="dlte", task="dlte_retrieval", dataset="dlte_v1",
        data_root=tmp_path,
    )
    # Auto-stage path that run.py uses (suite == "dlte" branch).
    labels = base / "datasets" / "dlte_v1" / "labels.json"
    assert labels.exists(), "labels.json should be at the deep path"

    # Now invoke build_command and assert --project_root resolves back to
    # ``base`` — same invariant the runners depend on.
    from trl_bench.registry import build_command
    emb = tmp_path / "embeddings" / "table" / "bert" / "dlte_v1.pkl"
    emb.parent.mkdir(parents=True)
    emb.write_bytes(b"")
    stages = build_command(
        model="bert", task="dlte_retrieval", dataset="dlte_v1",
        setting="diagonal", probe=None, seed=42,
        results_dir=tmp_path / "results",
        embeddings_path=emb, labels_path=labels,
        configs_root=tmp_path,
    )
    cmd = stages[0]
    pr_idx = cmd.index("--project_root")
    project_root = Path(cmd[pr_idx + 1])
    assert project_root == base, (
        f"dispatcher derived --project_root {project_root!r} but stager "
        f"produced base {base!r}; runner manifests live at "
        f"<base>/datasets/dlte_v1/manifests/, not <derived>/datasets/dlte_v1/."
    )


# == ctbench: semantic_parsing stager =======================================

def test_stage_ctbench_semparse_wiki_table_questions(tmp_path, monkeypatch):
    """Hermetic: stager rehydrates the MAPO layout from a fake HF cache.

    Verifies the on-disk layout matches what ``run_training.py`` reads:
    tables.jsonl + saved_programs.json + data_split_1/{train_split_shard_90-*,
    {dev,test}_split}.jsonl + tables_all/<table_id>.csv + labels.json
    sentinel. The CSV-count guard is monkeypatched down to 3 for the fake
    snapshot; the production value (2108) is exercised by the @slow HF test.
    """
    # Build a fake snapshot directory mimicking HF cache layout.
    fake_snapshot = tmp_path / "_fake_hf_cache"
    fake_wtq = fake_snapshot / "data" / "wtq_mapo"
    fake_split = fake_wtq / "data_split_1"
    fake_split.mkdir(parents=True)
    fake_tables_all = fake_wtq / "tables_all"
    fake_tables_all.mkdir()

    (fake_wtq / "tables.jsonl").write_text(
        '{"name": "csv/200-csv/0.csv", "kg": {}, "row_ents": []}\n'
    )
    (fake_wtq / "saved_programs.json").write_text('{"nt-0": []}\n')
    (fake_split / "dev_split.jsonl").write_text(
        '{"id": "nt-0", "question": "what?"}\n'
    )
    (fake_split / "test_split.jsonl").write_text(
        '{"id": "nt-1", "question": "huh?"}\n'
    )
    (fake_split / "train_split.jsonl").write_text(
        '{"id": "nt-2", "question": "wat?"}\n'
    )
    # 90 shards — matches MAPO upstream's shard count.
    for i in range(90):
        (fake_split / f"train_split_shard_90-{i}.jsonl").write_text(
            f'{{"id": "nt-{100+i}", "question": "shard-{i}"}}\n'
        )
    # 3 fake CSVs — the count guard is monkeypatched to 3 below.
    for tid in ("t_200_0", "t_200_1", "t_201_5"):
        (fake_tables_all / f"{tid}.csv").write_text("col1,col2\n1,2\n3,4\n")

    # Monkeypatch snapshot_download to return the fake snapshot root.
    def _fake_snapshot_download(repo_id, repo_type, allow_patterns,
                                revision=None):
        assert repo_id == "logo-lab/trl-ctbench"
        assert repo_type == "dataset"
        assert "data/wtq_mapo/**" in allow_patterns
        return str(fake_snapshot)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_snapshot_download)
    monkeypatch.setattr(stage_mod, "_CTBENCH_SEMPARSE_EXPECTED_CSVS", 3)

    base = stage_mod.stage_dataset(
        suite="ctbench", task="semantic_parsing",
        dataset="wiki_table_questions",
        data_root=tmp_path,
    )
    assert base == tmp_path / "wiki_table_questions"
    assert (base / ".staged_ok").exists()
    # Top-level
    assert (base / "tables.jsonl").exists()
    assert (base / "saved_programs.json").exists()
    assert (base / "data_split_1").is_dir()
    # 90 shards
    split_files = sorted((base / "data_split_1").iterdir())
    n_shards = sum(1 for p in split_files
                   if p.name.startswith("train_split_shard_90-"))
    assert n_shards == 90, f"expected 90 shards, got {n_shards}"
    assert (base / "data_split_1" / "dev_split.jsonl").exists()
    assert (base / "data_split_1" / "test_split.jsonl").exists()
    # tables_all/<table_id>.csv — used by Stage-1 column extractors (BERT etc.)
    assert (base / "tables_all").is_dir()
    csv_files = sorted((base / "tables_all").iterdir())
    assert [p.name for p in csv_files] == [
        "t_200_0.csv", "t_200_1.csv", "t_201_5.csv",
    ]
    # Sanity-check the CSVs are bytes-preserved
    assert (base / "tables_all" / "t_200_0.csv").read_text() == "col1,col2\n1,2\n3,4\n"
    # labels.json sentinel for auto-stage detection
    labels = json.loads((base / "labels.json").read_text())
    assert labels["tables"] == "tables.jsonl"
    assert labels["saved_programs"] == "saved_programs.json"
    assert labels["data_split_1"] == "data_split_1"
    assert labels["tables_all"] == "tables_all"
    assert labels["n_train_shards"] == 90
    # n_eval_splits counts dev_split.jsonl + test_split.jsonl + train_split.jsonl
    assert labels["n_eval_splits"] == 3
    assert labels["n_tables"] == 3


def test_stage_ctbench_semparse_rejects_unknown_dataset(tmp_path, monkeypatch):
    """Stager must reject unwired datasets with NotImplementedError."""
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: str(tmp_path))
    with pytest.raises(NotImplementedError, match="semantic_parsing stager"):
        stage_mod.stage_dataset(
            suite="ctbench", task="semantic_parsing",
            dataset="not_a_real_wtq",
            data_root=tmp_path,
        )


def test_stage_ctbench_semparse_fails_loudly_when_shard_count_wrong(
    tmp_path, monkeypatch
):
    """Stager must error if the HF mirror has fewer than the expected 90
    train shards (catches a bad re-upload)."""
    fake_snapshot = tmp_path / "_fake_hf_cache"
    fake_wtq = fake_snapshot / "data" / "wtq_mapo"
    fake_split = fake_wtq / "data_split_1"
    fake_split.mkdir(parents=True)
    fake_tables_all = fake_wtq / "tables_all"
    fake_tables_all.mkdir()
    (fake_wtq / "tables.jsonl").write_text("{}\n")
    (fake_wtq / "saved_programs.json").write_text("{}\n")
    (fake_split / "dev_split.jsonl").write_text("{}\n")
    (fake_split / "test_split.jsonl").write_text("{}\n")
    # Only write 5 shards — should trip the count check.
    for i in range(5):
        (fake_split / f"train_split_shard_90-{i}.jsonl").write_text("{}\n")
    # tables_all/ must exist (otherwise the new check fires first) but the
    # guard under test is the shard-count one, which runs earlier.
    (fake_tables_all / "t_200_0.csv").write_text("col\n1\n")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: str(fake_snapshot))

    with pytest.raises(RuntimeError, match="expected 90"):
        stage_mod.stage_dataset(
            suite="ctbench", task="semantic_parsing",
            dataset="wiki_table_questions",
            data_root=tmp_path,
        )


def test_stage_ctbench_semparse_fails_loudly_when_csv_count_wrong(
    tmp_path, monkeypatch
):
    """Stager must error if the HF mirror is missing CSVs under tables_all/
    (catches a partial upload of the per-table CSV directory)."""
    fake_snapshot = tmp_path / "_fake_hf_cache"
    fake_wtq = fake_snapshot / "data" / "wtq_mapo"
    fake_split = fake_wtq / "data_split_1"
    fake_split.mkdir(parents=True)
    fake_tables_all = fake_wtq / "tables_all"
    fake_tables_all.mkdir()
    (fake_wtq / "tables.jsonl").write_text("{}\n")
    (fake_wtq / "saved_programs.json").write_text("{}\n")
    (fake_split / "dev_split.jsonl").write_text("{}\n")
    (fake_split / "test_split.jsonl").write_text("{}\n")
    (fake_split / "train_split.jsonl").write_text("{}\n")
    for i in range(90):
        (fake_split / f"train_split_shard_90-{i}.jsonl").write_text("{}\n")
    # Write only 1 CSV — should trip the count check (expects 3 here).
    (fake_tables_all / "t_200_0.csv").write_text("col\n1\n")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: str(fake_snapshot))
    monkeypatch.setattr(stage_mod, "_CTBENCH_SEMPARSE_EXPECTED_CSVS", 3)

    with pytest.raises(RuntimeError, match="per-table CSVs"):
        stage_mod.stage_dataset(
            suite="ctbench", task="semantic_parsing",
            dataset="wiki_table_questions",
            data_root=tmp_path,
        )


@pytest.mark.slow
def test_stage_ctbench_semparse_wiki_table_questions_from_hf(tmp_path):
    """Slow: stage wiki_table_questions for semantic_parsing from REAL HF and
    verify the rehydrated layout matches the MAPO runner's expectations.

    Reference counts (HF logo-lab/trl-ctbench wtq_mapo, 2026-05-21):
      tables.jsonl  ~24 MB
      saved_programs.json  ~6 MB
      data_split_1/{train_split_shard_90-0...89}.jsonl  (90 shards)
      data_split_1/{train,dev,test}_split.jsonl
      tables_all/t_NNN_NN.csv  (2108 per-table CSVs)
    """
    base = stage_mod.stage_dataset(
        suite="ctbench", task="semantic_parsing",
        dataset="wiki_table_questions",
        data_root=tmp_path, force=True,
    )
    assert (base / "tables.jsonl").exists()
    assert (base / "saved_programs.json").exists()
    assert (base / "data_split_1").is_dir()

    # 90 shards + 3 split files = 93 files
    split_files = list((base / "data_split_1").iterdir())
    assert len(split_files) == 93, (
        f"expected 93 files in data_split_1/, got {len(split_files)}"
    )
    n_shards = sum(1 for p in split_files
                   if p.name.startswith("train_split_shard_90-"))
    assert n_shards == 90

    # Sanity-check a shard is parseable JSON (one record per line)
    sample_shard = base / "data_split_1" / "train_split_shard_90-0.jsonl"
    line = sample_shard.read_text().splitlines()[0]
    rec = json.loads(line)
    for k in ("id", "question", "tokens", "answer"):
        assert k in rec, f"shard record missing key {k!r}"

    # tables.jsonl is a stream of KG-encoded tables
    line0 = (base / "tables.jsonl").read_text().splitlines()[0]
    table_rec = json.loads(line0)
    for k in ("name", "kg", "row_ents", "props"):
        assert k in table_rec, f"tables.jsonl record missing key {k!r}"

    # tables_all/<table_id>.csv — Stage-1 column extractors (BERT, GTE,
    # TabSketchFM, etc.) iterate this directory.
    tables_all = base / "tables_all"
    assert tables_all.is_dir(), f"tables_all/ missing under {base}"
    csvs = sorted(p for p in tables_all.iterdir() if p.suffix == ".csv")
    assert len(csvs) == 2108, (
        f"expected 2108 per-table CSVs under tables_all/, got {len(csvs)}"
    )
    # The CSV basenames should match tables.jsonl's ``name`` field. Use a small
    # random sample to keep the test fast.
    csv_names = {p.stem for p in csvs[:200]}
    tbl_names = set()
    for line in (base / "tables.jsonl").read_text().splitlines()[:500]:
        rec = json.loads(line)
        tbl_names.add(rec["name"])
    overlap = csv_names & tbl_names
    assert overlap, (
        f"no overlap between CSV basenames and tables.jsonl names; "
        f"csv_names sample={list(csv_names)[:3]}, "
        f"tbl_names sample={list(tbl_names)[:3]}"
    )

    # labels.json should record the CSV count
    labels = json.loads((base / "labels.json").read_text())
    assert labels["tables_all"] == "tables_all"
    assert labels["n_tables"] == 2108
