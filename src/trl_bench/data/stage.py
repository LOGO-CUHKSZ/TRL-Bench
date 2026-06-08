"""Stage HF datasets onto disk in the layout the model wrappers + probe expect.

The model wrappers (preserved as-is from the research repo) read CSV files
from disk. The probe (``run_task.py``) reads a ``labels.json`` with split
sections (``train`` / ``valid`` / ``test``) referencing those CSV filenames.
This module materializes one HF dataset entry into that layout so wrappers
and probe code do not need modification.

For ``rbench`` two task families have task-specific layouts:

  * ``record_linkage``: per-RL-dataset directory with ``tables/{tableA,tableB}.csv``,
    ``labels.json`` containing ``train`` / ``valid`` / ``test`` lists of
    ``{table1: {filename, row_idx}, table2: {filename, row_idx}, label: int}``
    entries, and ``metadata.json``. HF schema (single config ``record_linkage``):
    ``{source, family, pair_id, table_a_record_json, table_b_record_json, label}``.
    Records are reconstructed per-source-pair (the on-disk reference's ``tableA.csv``
    and ``tableB.csv`` are built by deduplicating row contents by side).
  * ``row_prediction``: per-OpenML-dataset directory with ``data.csv``,
    ``splits.npz``, ``dataset.json``, and a manifest ``labels.json`` so the
    auto-stage existence check finds it. HF schema (single config
    ``row_prediction``): ``{openml_id, dataset_name, row_idx, record_json,
    targets_json, target_specs_json, dataset_metadata_json}``.

Both rbench configs are not sub-configured by sub-dataset on HF; the stager
filters the parent config by ``source`` / ``openml_id`` to materialize one
sub-dataset at a time.

On-disk layout (per the canonical reference structure)::

    <data_root>/<dataset>/<dataset-subdir>/tables_all/<basename>.csv  # flat
    <data_root>/<dataset>/<dataset-subdir>/labels.json                # train/valid/test
    <data_root>/<dataset>/<dataset-subdir>/.staged_ok                 # sentinel

``<dataset-subdir>`` is the suite-specific inner subdirectory the paper repo
used (e.g. ``spider-join`` under ``spider_join``). Wrappers expect this
nesting; we preserve it.

Idempotent: a ``.staged_ok`` sentinel marks completion. Re-running is a no-op
unless the sentinel is removed or ``force=True`` is passed.

Per-suite stagers are dispatched by ``stage_dataset`` (ctbench / rbench / dlte).

Beyond the canonical pair-task layout, three additional ctbench task families
have task-specific layouts handled by separate stagers:

  * ``column_type_prediction`` (CTA, ``sato`` / ``sotab``): writes
    ``<base>/{train,test}.csv`` rows of ``(table_id, column_id, class)`` and
    materializes ``tables_all/<table_id>.csv``. A ``labels.json`` manifest
    of the per-split CSV paths is also written so the run.py dispatcher's
    auto-stage check finds it.
  * ``column_relation_prediction`` (CRA, ``wikict_relation``): writes
    per-table ``<base>/{train,test}/<train|test>_metadata.json`` files with
    decoded ``relation_annotations``. The HF ``sotab`` config does not carry
    relation annotations, so CRA uses ``wikict_relation`` — the ctbench config
    that ships the required annotation shape.
  * ``table_retrieval`` (``nq_tables``): writes ``train.json`` and
    ``dev.json`` (HF ``test`` split, which carries ``dev_``-prefixed
    question_ids) plus a ``table_id_to_csv.json`` mapping and per-table CSVs
    sourced from the ``nq_tables_tables`` config.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional


_SUITES = {"ctbench", "rbench", "dlte"}


def _repo_for(suite: str) -> str:
    if suite not in _SUITES:
        raise ValueError(f"unknown suite: {suite!r}. Valid: {sorted(_SUITES)}")
    return f"logo-lab/trl-{suite}"


# == ctbench: per-task stagers ==============================================
# Each ctbench task is a pair-classification / pair-regression problem with a
# uniform on-disk layout. HF parquet column names differ across tasks, but the
# on-disk shape is uniform.

# Field map: task_id -> {hf-column-name -> labels-json-key OR sentinel}.
# Sentinels recognised by the staging loop:
#   a_id / b_id  -> pair id column names (always present on the pair-info row)
#   a_csv / b_csv -> in-row CSV content columns (omit when the dataset uses a
#                    separate tables-config; see ``_CTBENCH_TABLES_CONFIG``)
#   label        -> label column name
#   label_dtype  -> "int" | "float", coerces the JSON value type. Defaults to
#                    "float" for backwards-compat with the original code, which
#                    preserved the HF parquet dtype (always float for these
#                    tasks). Classification tasks set "int" to match the
#                    reference labels.json (e.g. ``ckan_subset`` has ``"label": 0``
#                    where HF gives ``0.0``).
#   meta__<hfcol> -> per-row metadata to copy through to labels.json under
#                    a chosen JSON key (value side of the entry).
_CTBENCH_PAIR_TASK_FIELDS: dict[str, dict[str, str]] = {
    "join_classification": {
        "a_id":  "table_a_id",   "a_csv": "table_a_csv",
        "b_id":  "table_b_id",   "b_csv": "table_b_csv",
        "label": "label",
        "label_dtype": "int",
        # Extra row metadata to copy through:  hf_col_name -> labels_json_key
        "meta__join_col_a": "join_col_table1",
        "meta__join_col_b": "join_col_table2",
    },
    "join_containment": {
        "a_id":  "table_a_id",   "a_csv": "table_a_csv",
        "b_id":  "table_b_id",   "b_csv": "table_b_csv",
        "label": "label",
        "label_dtype": "float",
        "meta__join_col_a": "join_col_table1",
        "meta__join_col_b": "join_col_table2",
    },
    "union_classification": {
        # wiki_union, etc.: pair-info-only HF schema (CSV content lives in a
        # separate ``<dataset>_tables`` config — see ``_CTBENCH_TABLES_CONFIG``).
        # join_col_a/b columns are uniformly empty in this family and not
        # emitted in the reference labels.json, so we don't copy them through.
        "a_id":  "table_a_id",
        "b_id":  "table_b_id",
        "label": "label",
        "label_dtype": "int",
    },
    "union_regression": {
        # ecb_union: in-row CSV pattern, float label, no join_col metadata.
        "a_id":  "table_a_id",   "a_csv": "table_a_csv",
        "b_id":  "table_b_id",   "b_csv": "table_b_csv",
        "label": "label",
        "label_dtype": "float",
    },
    "table_subset": {
        # ckan_subset, nq_tables, opendata_*: pair-info-only HF schema. Same
        # shape as union_classification (binary label, no metadata in labels.json).
        "a_id":  "table_a_id",
        "b_id":  "table_b_id",
        "label": "label",
        "label_dtype": "int",
    },
}

# Datasets whose paper-repo on-disk subdir nests under <data_root>/<dataset>/<inner>.
# Datasets not in this map live directly at <data_root>/<dataset>/ with no inner.
_CTBENCH_INNER_SUBDIR: dict[str, str] = {
    "spider_join": "spider-join",
    "ecb_join":    "ecb-join",
    # ecb_union, wiki_union, ckan_subset, etc. -> labels.json sits at
    #   <data_root>/<dataset>/labels.json directly. Omit from this map.
}

# Datasets whose HF parquet has pair-info-only rows (no in-row CSV content).
# CSV content lives in a SEPARATE HF dataset config named by this map. The
# tables config row schema (verified on HF) is:
#   {"table_id": <basename>, "csv_text": <str>, "n_rows": int, "n_cols": int}
# Pair ids (``table_a_id`` / ``table_b_id``) sometimes carry an extra ``.bz2``
# extension that the tables config doesn't (e.g. ckan_subset pair id is
# ``foo.csv.bz2`` while the tables row id is ``foo.csv``). The staging loop
# strips trailing ``.bz2`` when looking up content. This matches the
# reference's documented ``.csv`` vs ``.csv.bz2`` extension drift.
#
# NOTE: the task spec called the content column ``table_csv``; HF actually
# uses ``csv_text``. The constant below records the verified column name.
_CTBENCH_TABLES_CONFIG: dict[str, str] = {
    "ckan_subset":      "ckan_subset_tables",
    # wiki_tables_full (46,364 tables, uploaded 2026-05-20) is the superset that
    # covers BOTH wiki_union (40,752) AND wiki_containment (44,696, of which
    # 5,612 are not in the original wiki_tables config). The legacy
    # ``wiki_tables`` config (40,752 rows) is retained on HF for back-compat,
    # but the stagers route through ``wiki_tables_full`` so containment cells
    # don't fail with KeyErrors on the 5,612 missing tables.
    "wiki_union":       "wiki_tables_full",     # filename-keyed labels; generic headers OK
    # wiki_containment routes to a dedicated config with REAL column headers.
    # wiki_tables_full's csv_text has generic col0,col1 headers, but the
    # join_containment labels reference real column names (join_col_a/b, e.g.
    # 'Lineal'); the generic version makes the runner raise "Column ... not
    # found". wiki_containment_tables (uploaded 2026-05-30) carries the same
    # 44,696 tables with their real headers. wiki_union is unaffected (its
    # labels key on filename, not column name) so it stays on wiki_tables_full.
    "wiki_containment": "wiki_containment_tables",
    "nq_tables":        "nq_tables_tables",
    "opendata_main":    "opendata_main_tables",
    "opendata_can":     "opendata_can_tables",
    "opendata_usa":     "opendata_usa_tables",
    "opendata_uk_sg":   "opendata_uk_sg_tables",
}
_CTBENCH_TABLES_CONTENT_COL = "csv_text"

# Per-process memoization of loaded tables-config dicts. Wiki tables are large
# (~hundreds of MB); re-staging wiki_union and wiki_containment in the same
# process must not reload.
_CTBENCH_TABLES_CACHE: dict[tuple[str, Optional[str]], dict[str, str]] = {}


def _load_ctbench_tables_dict(
    tables_config: str, *, revision: Optional[str] = None,
) -> dict[str, str]:
    """Load one CTbench ``*_tables`` config as a ``{table_id -> csv_text}`` dict.

    Lazy-imports ``datasets`` so unit tests can monkeypatch ``sys.modules``.
    Result is cached per (config, revision).
    """
    key = (tables_config, revision)
    cached = _CTBENCH_TABLES_CACHE.get(key)
    if cached is not None:
        return cached
    from datasets import load_dataset
    ds = load_dataset(
        _repo_for("ctbench"), name=tables_config, split="train", revision=revision,
    )
    out: dict[str, str] = {}
    for row in ds:
        out[row["table_id"]] = row[_CTBENCH_TABLES_CONTENT_COL]
    _CTBENCH_TABLES_CACHE[key] = out
    return out


# Datasets whose reference labels.json is single-line (no indent). Default is
# indent=4 to match the bulk of the reference corpus (spider_join,
# ecb_union, wiki_union, etc.). ``ckan_subset`` is the only outlier observed.
_CTBENCH_LABELS_JSON_INDENT: dict[str, Optional[int]] = {
    "ckan_subset": None,
}

# Per-dataset split ordering in the reference labels.json. Default is the
# canonical (train, valid, test) order; ckan_subset uses (train, test, valid).
_CTBENCH_SPLIT_ORDER: dict[str, tuple[str, ...]] = {
    "ckan_subset": ("train", "test", "valid"),
}
_CTBENCH_DEFAULT_SPLIT_ORDER: tuple[str, ...] = ("train", "valid", "test")
_CTBENCH_HF_SPLIT_NAMES: dict[str, str] = {
    "train": "train", "valid": "validation", "test": "test",
}


# Table-disjoint ("strict") labels for pair/union tasks are published on HF as
# raw files under ``labels_strict/<dataset>/labels_strict.json`` (like
# ``splits/`` and ``tables_json/`` — not a parquet config, so invisible to
# load_dataset). The strict split shares no table across train/test; it backs
# the paper's table-disjoint (†) results and is selected by the downstream
# runner's split_protocol='strict'. Only these pair/union datasets have strict
# labels published on HF (record_linkage's are NOT).
_CTBENCH_STRICT_LABELS_DATASETS = frozenset({
    "spider_join", "wiki_containment", "wiki_union", "ecb_union",
})


def _fetch_labels_strict(
    *, repo: str, dataset: str, dst_path: Path, revision: Optional[str] = None,
) -> None:
    """Download the table-disjoint ("strict") labels for ``dataset`` from the HF
    ``labels_strict/<dataset>/labels_strict.json`` raw-file path into
    ``dst_path`` (placed next to ``labels.json``)."""
    from huggingface_hub import hf_hub_download
    import shutil
    src = hf_hub_download(
        repo, f"labels_strict/{dataset}/labels_strict.json",
        repo_type="dataset", revision=revision,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst_path)


def _stage_ctbench_pair_task(
    *, task: str, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one (ctbench, pair-task, dataset) onto disk.

    Produces ``dst_root/<dataset>/<inner>/{tables_all/, labels.json}``.

    Two HF schema patterns are supported:

    1. **In-row CSV** (``spider_join``, ``ecb_union``, ...): the pair parquet
       row carries both pair info and the table CSV content
       (``table_a_csv``/``table_b_csv`` columns).
    2. **Multi-config** (``ckan_subset``, ``wiki_union``, ``nq_tables``, ...):
       the pair parquet row carries only pair info; CSV content lives in a
       separate HF config named by ``_CTBENCH_TABLES_CONFIG[dataset]``. The
       tables-config dict is loaded once per process and memoized.
    """
    from datasets import load_dataset

    fields = _CTBENCH_PAIR_TASK_FIELDS[task]
    inner = _CTBENCH_INNER_SUBDIR.get(dataset)   # None -> no inner subdir
    base = dst_root / dataset
    if inner is not None:
        base = base / inner
    base.mkdir(parents=True, exist_ok=True)

    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)
    labels_path = base / "labels.json"

    repo = _repo_for("ctbench")

    # Decide which CSV-source pattern this dataset uses. Datasets in the
    # tables-config map: look up content from the loaded dict. Others: read
    # content from the pair row columns named in the field map.
    tables_config = _CTBENCH_TABLES_CONFIG.get(dataset)
    tables_dict: Optional[dict[str, str]] = None
    if tables_config is not None:
        tables_dict = _load_ctbench_tables_dict(tables_config, revision=revision)

    split_order = _CTBENCH_SPLIT_ORDER.get(dataset, _CTBENCH_DEFAULT_SPLIT_ORDER)
    labels: dict[str, list[dict]] = {k: [] for k in split_order}

    meta_pairs = [
        (k.removeprefix("meta__"), v)
        for k, v in fields.items() if k.startswith("meta__")
    ]

    label_dtype = fields.get("label_dtype", "float")
    _coerce_label = int if label_dtype == "int" else float

    seen_content_hash: dict[str, str] = {}   # basename -> md5 (collision guard)
    n_total_rows = 0

    for label_split in split_order:
        hf_split = _CTBENCH_HF_SPLIT_NAMES[label_split]
        ds = load_dataset(repo, name=dataset, split=hf_split, revision=revision)
        for row in ds:
            n_total_rows += 1
            entry: dict = {}
            for side in ("a", "b"):
                full_id = row[fields[f"{side}_id"]]
                if tables_dict is not None:
                    # Multi-config lookup. Pair ids may carry an extra
                    # ``.bz2`` extension; the tables-config row ids don't.
                    lookup_id = full_id[:-4] if full_id.endswith(".bz2") else full_id
                    try:
                        csv_content = tables_dict[lookup_id]
                    except KeyError as e:
                        raise RuntimeError(
                            f"pair row references table_id {full_id!r} "
                            f"(lookup={lookup_id!r}) not present in tables "
                            f"config {tables_config!r}"
                        ) from e
                    # On-disk basename uses the tables-config id (no .bz2),
                    # since the CSV content is plain text. labels.json
                    # preserves the original pair id (which may include .bz2)
                    # to match the reference's documented extension drift.
                    basename = os.path.basename(lookup_id)
                else:
                    csv_content = row[fields[f"{side}_csv"]]
                    basename = os.path.basename(full_id)
                target = tables_all / basename
                new_hash = hashlib.md5(csv_content.encode()).hexdigest()
                if basename in seen_content_hash:
                    if seen_content_hash[basename] != new_hash:
                        raise RuntimeError(
                            f"basename collision in ctbench/{dataset}: "
                            f"{basename!r} has different content across rows"
                        )
                else:
                    target.write_text(csv_content)
                    seen_content_hash[basename] = new_hash
                entry["table1" if side == "a" else "table2"] = {"filename": full_id}
            # Coerce label to the dtype recorded in the field map. HF gives
            # floats for classification too (0.0/1.0); reference uses int.
            entry["label"] = _coerce_label(row[fields["label"]])
            for hf_col, dst_key in meta_pairs:
                if hf_col in row:
                    entry[dst_key] = row[hf_col]
            labels[label_split].append(entry)

    indent = _CTBENCH_LABELS_JSON_INDENT.get(dataset, 4)
    labels_path.write_text(json.dumps(labels, indent=indent))

    # Fetch the table-disjoint "strict" labels (consumed by the runner's
    # split_protocol='strict') from HF, next to labels.json. Only the pair/union
    # datasets in _CTBENCH_STRICT_LABELS_DATASETS have them published.
    if dataset in _CTBENCH_STRICT_LABELS_DATASETS:
        _fetch_labels_strict(
            repo=repo, dataset=dataset, dst_path=base / "labels_strict.json",
            revision=revision,
        )

    (base / ".staged_ok").write_text(
        f"rows={n_total_rows} tables={len(seen_content_hash)}\n"
    )
    return base


# == ctbench: CTA / CRA / table_retrieval stagers ===========================
# These three families don't share the pair-task layout; each has a custom
# on-disk shape consumed by the corresponding runner. Schemas match the
# logo-lab/trl-ctbench HF configs.

# CTA: HF config row schema (sato / sotab):
#   {table_id: str, column_id: int, class: str, table_csv: str, table_columns: list}
# Runner expects:
#   <base>/train.csv and <base>/test.csv with columns (table_id, column_id, class).
#   <base>/tables_all/<table_id>.csv with the per-table CSV content.
_CTBENCH_CTA_DATASETS = frozenset({"sato", "sotab"})

# CRA: HF row schema (uniform across all CRA configs):
#   {table_id, csv_filename, table_csv, relation_annotations_json: <json-encoded
#    list[{column_id, relations, relation_ids}]>, [+other config-specific fields]}
# Runner expects per-split metadata JSON: a list of dicts
#   {"table_id": <id>, "relation_annotations": [{"column_id": int,
#    "relation_ids": [int, ...]}, ...]}
# at <base>/train/train_metadata.json and <base>/test/test_metadata.json.
#
# `sotab_relation` (added 2026-05-20) is the CRA-shape upload of the working
# repo's `datasets/SOTAB/{train,test}_metadata.json` (27,867 train + 355 test
# rows) joined with per-table CSV content from `datasets/SOTAB/tables/`. The
# user-facing dataset name `sotab` resolves to this HF config so the
# reference's SOTAB-CRA cell becomes reproducible. The existing CTA-shape
# `sotab` config (column_type_prediction) is preserved unchanged for back-
# compat — the dispatcher routes by `task`, not by `dataset`.
#
# `wikict_relation` predates this layout; its schema includes extra columns
# (headers, num_columns, num_rows, type_annotations_json) that the CRA stager
# does NOT consume — only the four columns above are required.
_CTBENCH_CRA_DATASETS: dict[str, str] = {
    # user-facing dataset name -> HF config name
    "wikict_relation": "wikict_relation",
    "sotab":           "sotab_relation",
}

# table_retrieval: HF nq_tables row schema:
#   {question_id, question, table_id, answers: list[str]}
# Available splits: train, test, validation. The reference labels its
# eval split "dev"; HF "test" contains the `dev_`-prefixed question_ids, so
# HF train -> on-disk train.json, HF test -> on-disk dev.json.
#
# HF nq_tables_tables row schema (split=train only):
#   {table_id: "<basename>.csv", csv_text: str, n_rows: int, n_cols: int}
_CTBENCH_RETRIEVAL_DATASETS = frozenset({"nq_tables"})
_CTBENCH_RETRIEVAL_TABLES_CONFIG = "nq_tables_tables"


def _stage_ctbench_cta_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one CTA (column_type_prediction) dataset onto disk.

    Produces::

        <dst_root>/<dataset>/train.csv          # (table_id, column_id, class)
        <dst_root>/<dataset>/test.csv
        <dst_root>/<dataset>/tables_all/<id>.csv
        <dst_root>/<dataset>/labels.json        # manifest of split CSV paths
        <dst_root>/<dataset>/.staged_ok

    The ``labels.json`` is a manifest (not reference labels) — its purpose
    is to satisfy ``run.py::_resolve_labels_path``'s existence check so that
    auto-stage doesn't silently fall back to ``None``. The CTA runner reads
    ``train.csv`` / ``test.csv`` directly, not ``labels.json``.
    """
    from datasets import load_dataset

    if dataset not in _CTBENCH_CTA_DATASETS:
        raise NotImplementedError(
            f"CTA stager for ctbench dataset {dataset!r} is not wired. "
            f"Wired datasets: {sorted(_CTBENCH_CTA_DATASETS)}."
        )

    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")

    # HF row schema: {table_id, column_id, class, table_csv, table_columns}.
    # Each (table_id, column_id) pair appears once; the same table_csv is
    # repeated across all columns of that table.
    label_columns = ["table_id", "column_id", "class"]
    seen_table_hash: dict[str, str] = {}
    n_total_rows = 0

    for split_name, csv_filename in (("train", "train.csv"), ("test", "test.csv")):
        ds = load_dataset(repo, name=dataset, split=split_name, revision=revision)
        # Write CSV header.
        lines = [",".join(label_columns)]
        for row in ds:
            n_total_rows += 1
            tid = row["table_id"]
            cid = row["column_id"]
            cls = row["class"]
            # CSV-escape ``class`` if it contains commas/quotes.
            cls_field = _csv_escape(cls)
            tid_field = _csv_escape(str(tid))
            lines.append(f"{tid_field},{cid},{cls_field}")
            # Materialize the per-table CSV exactly once per table_id.
            # For numeric tids the filename gets a 'table_' prefix so the bert
            # column extractor's filename-stem table_id matches the CTA
            # runner's lookup (train_ct_mode4.py:84 does f'table_{tid}' when
            # pandas reads the labels.csv table_id column as int). String tids
            # (e.g. sotab's URL-shaped IDs) keep their natural names.
            tid_str = str(tid)
            tid_filename = (
                f"table_{tid_str}"
                if tid_str.lstrip("-").isdigit()
                else tid_str
            )
            csv_content = row["table_csv"]
            new_hash = hashlib.md5(csv_content.encode()).hexdigest()
            prior = seen_table_hash.get(tid_str)
            if prior is None:
                (tables_all / f"{tid_filename}.csv").write_text(csv_content)
                seen_table_hash[tid_str] = new_hash
            elif prior != new_hash:
                raise RuntimeError(
                    f"table_id collision in ctbench/{dataset}: {tid_str!r} "
                    f"has different content across rows"
                )
        (base / csv_filename).write_text("\n".join(lines) + "\n")

    # Manifest labels.json so auto-stage's existence check sees a file.
    labels_manifest = {
        "train": "train.csv",
        "test":  "test.csv",
        "tables_all": "tables_all",
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"rows={n_total_rows} tables={len(seen_table_hash)}\n"
    )
    return base


def _csv_escape(value: str) -> str:
    """RFC 4180 minimal CSV escape: wrap in quotes (and double inner quotes)
    only when the value contains a comma, quote, or newline. Pandas reads
    both quoted and unquoted forms identically for these label columns.
    """
    if any(c in value for c in (",", '"', "\n", "\r")):
        return '"' + value.replace('"', '""') + '"'
    return value


def _stage_ctbench_cra_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
    on_disk_dataset: Optional[str] = None,
) -> Path:
    """Materialize one CRA (column_relation_prediction) dataset onto disk.

    Produces::

        <dst_root>/<on_disk_dataset>/train/train_metadata.json
        <dst_root>/<on_disk_dataset>/test/test_metadata.json
        <dst_root>/<on_disk_dataset>/tables_all/<csv_filename>
        <dst_root>/<on_disk_dataset>/labels.json
        <dst_root>/<on_disk_dataset>/.staged_ok

    ``on_disk_dataset`` defaults to ``dataset`` but is overridden for
    datasets that also have a CTA-shape config under the same name (e.g.
    ``sotab`` -> ``sotab_cra``) so CTA and CRA layouts don't collide on disk.

    Each ``*_metadata.json`` is a list of per-table dicts:
        {"table_id": str, "relation_annotations": [
            {"column_id": int, "relations": list[str], "relation_ids": list[int]},
            ...
        ]}

    The HF row's ``relation_annotations_json`` field is a JSON-encoded list of
    these per-column dicts; we decode and pass it through verbatim.
    """
    from datasets import load_dataset

    if dataset not in _CTBENCH_CRA_DATASETS:
        raise NotImplementedError(
            f"CRA stager for ctbench dataset {dataset!r} is not wired. "
            f"Wired datasets: {sorted(_CTBENCH_CRA_DATASETS)}. The paper's "
            f"`SOTAB` CRA cell is reproducible from HF via the "
            f"`sotab_relation` config (added 2026-05-20); pass dataset='sotab' "
            f"with task='column_relation_prediction'."
        )

    hf_config = _CTBENCH_CRA_DATASETS[dataset]
    on_disk_dataset = on_disk_dataset or dataset
    base = dst_root / on_disk_dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")

    seen_table_hash: dict[str, str] = {}
    n_total_rows = 0

    for split_name, inner_dirname, json_filename in (
        ("train", "train", "train_metadata.json"),
        ("test",  "test",  "test_metadata.json"),
    ):
        split_dir = base / inner_dirname
        split_dir.mkdir(exist_ok=True)
        ds = load_dataset(repo, name=hf_config, split=split_name, revision=revision)
        metadata: list[dict] = []
        for row in ds:
            n_total_rows += 1
            try:
                rel_annots = json.loads(row["relation_annotations_json"])
            except KeyError as e:
                raise RuntimeError(
                    f"HF config {hf_config!r} row lacks "
                    f"`relation_annotations_json`; cannot stage CRA"
                ) from e
            # Materialize CSV content. ``csv_filename`` is the on-disk name;
            # ``table_csv`` is the content. Idempotent on (filename -> hash).
            csv_filename = row["csv_filename"]
            csv_content = row["table_csv"]
            # Skip empty/whitespace-only CSVs: the bert column extractor's
            # ``pd.read_csv(csv_path, dtype=str, engine='python')`` raises
            # ``pandas.errors.EmptyDataError`` on zero-byte / header-less
            # inputs, crashing Stage-1. Empirically ~201/53768 wikict_relation
            # rows have empty ``table_csv``. Also drop the metadata entry so
            # downstream consumers don't reference a missing on-disk CSV.
            if not csv_content or not csv_content.strip():
                continue
            metadata.append({
                "table_id": row["table_id"],
                "relation_annotations": rel_annots,
            })
            new_hash = hashlib.md5(csv_content.encode()).hexdigest()
            prior = seen_table_hash.get(csv_filename)
            if prior is None:
                (tables_all / csv_filename).write_text(csv_content)
                seen_table_hash[csv_filename] = new_hash
            elif prior != new_hash:
                raise RuntimeError(
                    f"csv_filename collision in ctbench/{dataset}: "
                    f"{csv_filename!r} has different content across rows"
                )
        (split_dir / json_filename).write_text(json.dumps(metadata, indent=2))

    # Manifest labels.json — see CTA stager docstring for rationale.
    labels_manifest = {
        "train": "train/train_metadata.json",
        "test":  "test/test_metadata.json",
        "tables_all": "tables_all",
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"rows={n_total_rows} tables={len(seen_table_hash)}\n"
    )
    return base


# The NQ-Tables retrieval corpus ``tables.json`` is published on HF as a
# raw-file artifact (top-level ``tables_json/nq_tables/``, like ``splits/`` and
# ``labels_strict/`` — NOT a parquet config, so it doesn't surface via
# load_dataset). It is required because the ``nq_tables_tables`` parquet
# ``csv_text`` is lossy: original EMPTY column headers were replaced upstream
# with slugified placeholders, which would change the retrieval encoder's table
# embeddings. tables.json preserves the ``''`` headers.
_CTBENCH_RETRIEVAL_TABLES_JSON_HF = "tables_json/nq_tables/tables.json"


def _fetch_nq_tables_json(
    *, repo: str, dst_path: Path, revision: Optional[str] = None,
) -> None:
    """Download the NQ-Tables corpus ``tables.json`` from the HF
    ``tables_json/nq_tables/tables.json`` raw-file path into ``dst_path``.

    Consumed DIRECTLY by the retrieval encoder (``generate_tabert_embeddings``).
    Large (~450 MB) heterogeneous JSON, hence a raw-file artifact.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    src = hf_hub_download(
        repo, _CTBENCH_RETRIEVAL_TABLES_JSON_HF, repo_type="dataset",
        revision=revision,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst_path)


def _stage_ctbench_retrieval_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one table_retrieval dataset onto disk.

    Produces::

        <dst_root>/<dataset>/train.json              # train questions
        <dst_root>/<dataset>/dev.json                # dev/test questions
        <dst_root>/<dataset>/table_id_to_csv.json    # {table_id -> csv basename}
        <dst_root>/<dataset>/tables_all/<basename>.csv
        <dst_root>/<dataset>/labels.json             # manifest
        <dst_root>/<dataset>/.staged_ok

    The HF ``nq_tables`` config provides queries (3 splits: train, test,
    validation; ``test`` contains the dev_*-prefixed question_ids that the
    reference labels "dev"). The HF ``nq_tables_tables`` config provides
    the per-table CSVs (1 split: train) — these are the corpus of tables to
    retrieve from.

    Output JSON shape (consumed by ``trl_bench.tasks.table_retrieval``):
        train.json / dev.json: list of dicts
            {"question_id": str, "question": str, "table_id": str,
             "answers": list[str]}
        table_id_to_csv.json: {"<table_id>": "<csv_basename>.csv"}
    """
    from datasets import load_dataset

    if dataset not in _CTBENCH_RETRIEVAL_DATASETS:
        raise NotImplementedError(
            f"table_retrieval stager for ctbench dataset {dataset!r} is not "
            f"wired. Wired datasets: {sorted(_CTBENCH_RETRIEVAL_DATASETS)}."
        )

    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")

    n_total_rows = 0

    # Queries side: train -> train.json, test -> dev.json. ``validation`` is
    # currently ignored — the paper's dev/test grid uses only train+dev.
    # We also collect every canonical query-side table_id seen so we can build
    # the reference's ``table_id_to_csv.json`` mapping below.
    canonical_table_ids: set[str] = set()
    for hf_split, out_filename in (("train", "train.json"), ("test", "dev.json")):
        ds = load_dataset(repo, name=dataset, split=hf_split, revision=revision)
        questions: list[dict] = []
        for row in ds:
            n_total_rows += 1
            tid = row["table_id"]
            canonical_table_ids.add(tid)
            questions.append({
                "question_id": row["question_id"],
                "question":    row["question"],
                "table_id":    tid,
                "answers":     list(row["answers"]),
            })
        (base / out_filename).write_text(json.dumps(questions, indent=2))

    # Tables side: load nq_tables_tables (train-only split). The HF config
    # carries ``table_id`` as the CSV filename (underscores + ``.csv`` suffix);
    # the queries side uses canonical names with spaces. We materialize each
    # CSV at ``tables_all/<csv_filename>`` and record both forms in the
    # mapping so the runtime ``build_csv_to_table_id_mapping`` helper can
    # invert canonical_tid <-> csv_basename via its ``[:-4]`` strip.
    tables_ds = load_dataset(
        repo, name=_CTBENCH_RETRIEVAL_TABLES_CONFIG, split="train",
        revision=revision,
    )
    seen_csv_hash: dict[str, str] = {}
    # Build {csv_basename(no .csv) -> csv_filename(.csv)} for quick lookup.
    csv_filenames_index: dict[str, str] = {}
    for row in tables_ds:
        csv_filename = row["table_id"]    # e.g. "Foo_Bar_HASH.csv"
        csv_content = row["csv_text"]
        new_hash = hashlib.md5(csv_content.encode()).hexdigest()
        prior = seen_csv_hash.get(csv_filename)
        if prior is None:
            (tables_all / csv_filename).write_text(csv_content)
            seen_csv_hash[csv_filename] = new_hash
            basename = (
                csv_filename[:-4] if csv_filename.endswith(".csv") else csv_filename
            )
            csv_filenames_index[basename] = csv_filename
        elif prior != new_hash:
            raise RuntimeError(
                f"csv_filename collision in ctbench/{dataset}: "
                f"{csv_filename!r} has different content across rows"
            )

    # Build the canonical {canonical_table_id -> csv_filename} mapping. The
    # transform between canonical (spaces, no ext) and CSV (underscores, .csv)
    # is `canonical.replace(" ", "_") + ".csv"`; this matches the HF rows. If
    # the underscored form isn't in the tables index we record the canonical
    # form back to itself with .csv appended, matching the reference's
    # behavior of including every query-referenced table_id even when its CSV
    # is absent from the corpus (the retrieval runner tolerates this).
    table_id_to_csv: dict[str, str] = {}
    for canonical in sorted(canonical_table_ids):
        underscored = canonical.replace(" ", "_")
        csv_filename = csv_filenames_index.get(underscored, f"{underscored}.csv")
        table_id_to_csv[canonical] = csv_filename
    # Also include corpus tables not referenced by any query, so the corpus
    # side has full coverage when iterated.
    for basename, csv_filename in csv_filenames_index.items():
        if csv_filename not in table_id_to_csv.values():
            # Canonical key for orphan corpus entries: use the underscored
            # basename as-is (no canonical equivalent known).
            table_id_to_csv.setdefault(basename, csv_filename)

    (base / "table_id_to_csv.json").write_text(
        json.dumps(table_id_to_csv, indent=2)
    )

    # Fetch the NQ corpus tables.json (raw artifact; preserves '' headers the
    # nq_tables_tables parquet drops). Consumed directly by the retrieval encoder.
    _fetch_nq_tables_json(repo=repo, dst_path=base / "tables.json", revision=revision)

    # Manifest labels.json.
    labels_manifest = {
        "train": "train.json",
        "dev":   "dev.json",
        "table_id_mapping": "table_id_to_csv.json",
        "tables_json": "tables.json",
        "tables_all": "tables_all",
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"rows={n_total_rows} tables={len(seen_csv_hash)}\n"
    )
    return base


# == ctbench: column_clustering stager ======================================
# Column-clustering eval reads ``<ds>/all.csv`` (table_id, column_id, class)
# and per-table CSVs at ``<ds>/tables_all/<table_id>.csv``. The HF ``sato``
# config already carries these in its CTA shape (one row per (table, column)
# with the same table_csv content); we just emit the canonical ``all.csv``
# file alongside the CTA staging output instead of two separate CSV splits.
#
# The runner's ``--dataset <ds>`` argument is a DIRECTORY path; the dispatcher
# uses ``labels_path.parent``. So ``labels.json`` and ``all.csv`` both live
# at ``<dst_root>/<dataset>/``.
_CTBENCH_CLUSTERING_DATASETS = frozenset({"sato", "sotab"})


def _stage_ctbench_clustering_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one column_clustering dataset (``sato`` or ``sotab``) to disk.

    Produces (for ``dataset='sato'``)::

        <dst_root>/sato/all.csv               # (table_id, column_id, class)
        <dst_root>/sato/tables_all/<id>.csv
        <dst_root>/sato/labels.json           # manifest (merged with CTA's)
        <dst_root>/sato/.staged_ok_clustering # task-specific sentinel

    The HF ``sato`` / ``sotab`` configs are CTA-shaped (one row per
    (table, column) with the same ``table_csv``), each with ``train`` + ``test``
    splits. The runner's clustering metric is on the union of both — the
    on-disk ``all.csv`` concatenates them in (train, test) order, matching the
    reference layout (``datasets/sato/all.csv`` is identical to
    train.csv + test.csv minus duplicated headers, verified row count =
    120,609 for sato).

    ``sato``/``sotab`` are ALSO CTA datasets and share this directory + the
    per-column embeddings; clustering only adds ``all.csv``. ``sotab`` has
    string table_ids, so its ``tables_all/`` filenames match
    the CTA stager's (no numeric ``table_`` prefix) — the two stagings reuse
    the same per-table CSVs and embeddings. The task-specific
    ``.staged_ok_clustering`` sentinel ensures a prior CTA ``.staged_ok`` does
    not short-circuit this stager (and vice versa).
    """
    from datasets import load_dataset

    if dataset not in _CTBENCH_CLUSTERING_DATASETS:
        raise NotImplementedError(
            f"column_clustering stager for ctbench dataset {dataset!r} is not "
            f"wired. Wired datasets: {sorted(_CTBENCH_CLUSTERING_DATASETS)}."
        )

    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")
    seen_table_hash: dict[str, str] = {}
    lines = ["table_id,column_id,class"]
    n_total_rows = 0

    for split_name in ("train", "test"):
        ds = load_dataset(repo, name=dataset, split=split_name, revision=revision)
        for row in ds:
            n_total_rows += 1
            tid = row["table_id"]
            cid = row["column_id"]
            cls = row["class"]
            tid_str = str(tid)
            lines.append(
                f"{_csv_escape(tid_str)},{cid},{_csv_escape(cls)}"
            )
            csv_content = row["table_csv"]
            new_hash = hashlib.md5(csv_content.encode()).hexdigest()
            prior = seen_table_hash.get(tid_str)
            if prior is None:
                (tables_all / f"{tid_str}.csv").write_text(csv_content)
                seen_table_hash[tid_str] = new_hash
            elif prior != new_hash:
                raise RuntimeError(
                    f"table_id collision in ctbench/{dataset} (clustering): "
                    f"{tid_str!r} has different content across rows"
                )

    (base / "all.csv").write_text("\n".join(lines) + "\n")
    # Merge into any existing labels.json rather than clobber it: sato/sotab
    # share this directory with the CTA stager, whose manifest lists
    # train/test. Preserve those keys and add the clustering ones.
    labels_path = base / "labels.json"
    manifest: dict[str, str] = {}
    if labels_path.exists():
        try:
            manifest = json.loads(labels_path.read_text())
        except (ValueError, OSError):
            manifest = {}
    manifest["all"] = "all.csv"
    manifest["tables_all"] = "tables_all"
    labels_path.write_text(json.dumps(manifest, indent=2))
    # Clustering writes a task-specific sentinel so its staging stays
    # independent of CTA's ".staged_ok" in the shared directory (see the
    # sentinel_name selection in stage_dataset).
    (base / ".staged_ok_clustering").write_text(
        f"rows={n_total_rows} tables={len(seen_table_hash)}\n"
    )
    return base


# == ctbench: schema_matching stager ========================================
# Valentine: HF row schema is one row per (pair, gt_correspondence). Each row
# carries the pair metadata (pair_id, source, noise_type, noise_param,
# table_a_id, table_a_csv, table_a_columns, table_b_id, table_b_csv,
# table_b_columns) PLUS one GT column pair (column_a, column_b). To rebuild
# the working-repo layout:
#
#   <ds>/pairs.json         list of unique pair metadata
#   <ds>/ground_truth.csv   one row per GT column correspondence
#   <ds>/tables/<id>.csv    per-pair source / target CSVs
#   <ds>/labels.json        manifest
_CTBENCH_SCHEMA_MATCHING_DATASETS = frozenset({"valentine"})


def _stage_ctbench_schema_matching_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one schema_matching (valentine) dataset onto disk.

    Produces::

        <dst_root>/valentine/pairs.json            # 550 unique pair entries
        <dst_root>/valentine/ground_truth.csv      # one row per GT col pair
        <dst_root>/valentine/tables/<id>.csv       # 1098 per-pair CSVs
        <dst_root>/valentine/labels.json           # manifest
        <dst_root>/valentine/.staged_ok

    HF row count = 8,681 (= GT count). Number of unique (table_a_id, table_b_id)
    pairs = 550. Number of unique CSVs = 1,098 (= 2 sides * 549, since a few
    table_ids are shared across pair rows; the staging code deduplicates by
    table_id and stops re-writing identical-content rows).
    """
    from datasets import load_dataset

    if dataset not in _CTBENCH_SCHEMA_MATCHING_DATASETS:
        raise NotImplementedError(
            f"schema_matching stager for ctbench dataset {dataset!r} is not "
            f"wired. Wired datasets: {sorted(_CTBENCH_SCHEMA_MATCHING_DATASETS)}."
        )

    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_dir = base / "tables"
    tables_dir.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")
    ds = load_dataset(repo, name=dataset, split="train", revision=revision)

    seen_pair_ids: set[str] = set()
    pairs: list[dict] = []
    gt_lines: list[str] = [
        "pair_id,source,noise_type,noise_param,table_a,table_b,column_a,column_b"
    ]
    seen_csv_hash: dict[str, str] = {}

    for row in ds:
        pid = row["pair_id"]
        a_id = row["table_a_id"]
        b_id = row["table_b_id"]
        # First time we see a pair_id, write its metadata + materialize CSVs.
        if pid not in seen_pair_ids:
            seen_pair_ids.add(pid)
            pairs.append({
                "pair_id":     pid,
                "source":      row["source"],
                "noise_type":  row["noise_type"],
                "noise_param": row["noise_param"],
                "table_a":     a_id,
                "table_b":     b_id,
            })
            for tid, csv_content in (
                (a_id, row["table_a_csv"]),
                (b_id, row["table_b_csv"]),
            ):
                new_hash = hashlib.md5(csv_content.encode()).hexdigest()
                prior = seen_csv_hash.get(tid)
                if prior is None:
                    (tables_dir / tid).write_text(csv_content)
                    seen_csv_hash[tid] = new_hash
                elif prior != new_hash:
                    raise RuntimeError(
                        f"table_id collision in ctbench/valentine: {tid!r} "
                        f"has different content across rows"
                    )
        # Every row is one GT correspondence.
        gt_lines.append(
            ",".join(
                _csv_escape(str(row[k]))
                for k in (
                    "pair_id", "source", "noise_type", "noise_param",
                )
            )
            + f",{_csv_escape(a_id)},{_csv_escape(b_id)},"
            f"{_csv_escape(row['column_a'])},{_csv_escape(row['column_b'])}"
        )

    (base / "pairs.json").write_text(json.dumps(pairs, indent=2))
    (base / "ground_truth.csv").write_text("\n".join(gt_lines) + "\n")
    labels_manifest = {
        "pairs":        "pairs.json",
        "ground_truth": "ground_truth.csv",
        "tables":       "tables",
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"gt_rows={len(gt_lines) - 1} pairs={len(pairs)} "
        f"tables={len(seen_csv_hash)}\n"
    )
    return base


# == ctbench: union_search stager ===========================================
# Union-search datasets in HF: santos, tus, tus_hard, ugen_v1, ugen_v2 — all
# share the schema {table_id, csv_text, n_rows, n_cols, unionable_with}, with
# split={queries, datalake}. The runner expects
#   <ds>/groundtruth.pickle             dict[query_table_id -> [unionable_with]]
#   <ds>/tables_all/<id>.csv            queries + datalake CSV content
#   <ds>/labels.json                    manifest
#
# HF ``unionable_with`` carries the union-search ground truth.
_CTBENCH_UNION_SEARCH_DATASETS = frozenset({
    "santos", "tus", "tus_hard", "ugen_v1", "ugen_v2",
})


def _stage_ctbench_union_search_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one union_search dataset onto disk.

    Produces::

        <dst_root>/<dataset>/groundtruth.pickle    {query_id -> unionable_with}
        <dst_root>/<dataset>/tables_all/<id>.csv   per-table CSVs (queries+datalake)
        <dst_root>/<dataset>/labels.json           manifest
        <dst_root>/<dataset>/.staged_ok
    """
    import pickle
    from datasets import load_dataset

    if dataset not in _CTBENCH_UNION_SEARCH_DATASETS:
        raise NotImplementedError(
            f"union_search stager for ctbench dataset {dataset!r} is not "
            f"wired. Wired datasets: {sorted(_CTBENCH_UNION_SEARCH_DATASETS)}."
        )

    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")
    seen_csv_hash: dict[str, str] = {}
    groundtruth: dict[str, list[str]] = {}
    n_total = 0

    # Queries split provides GT (unionable_with). Datalake split is the search
    # corpus — load both and materialize every CSV under tables_all/.
    for split_name in ("queries", "datalake"):
        ds = load_dataset(repo, name=dataset, split=split_name, revision=revision)
        for row in ds:
            n_total += 1
            tid = row["table_id"]
            csv_content = row["csv_text"]
            new_hash = hashlib.md5(csv_content.encode()).hexdigest()
            prior = seen_csv_hash.get(tid)
            if prior is None:
                (tables_all / tid).write_text(csv_content)
                seen_csv_hash[tid] = new_hash
            elif prior != new_hash:
                raise RuntimeError(
                    f"table_id collision in ctbench/{dataset}: {tid!r} "
                    f"has different content across queries/datalake splits"
                )
            if split_name == "queries":
                groundtruth[tid] = list(row["unionable_with"])

    with open(base / "groundtruth.pickle", "wb") as f:
        pickle.dump(groundtruth, f)

    labels_manifest = {
        "groundtruth": "groundtruth.pickle",
        "tables_all":  "tables_all",
        "n_queries":   len(groundtruth),
        "n_tables":    len(seen_csv_hash),
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"queries={len(groundtruth)} tables={len(seen_csv_hash)} "
        f"hf_rows={n_total}\n"
    )
    return base


# == ctbench: join_search stager ============================================
# Join-search datasets in HF: opendata_main / opendata_can / opendata_usa /
# opendata_uk_sg. Each has two configs on HF:
#   <ds>          schema: query_table_id, candidate_table_id, task, query_column,
#                          candidate_column
#                 split=train, rows contain BOTH 'join' and 'union' tasks; we
#                 filter to task='join' for join_search.
#   <ds>_tables   schema: table_id, csv_text, n_rows, n_cols  (split=train)
#
# Dispatcher expects:
#   <ds>/queries.csv         (query_table, query_column)
#   <ds>/ground_truth.csv    (query_table, candidate_table, query_column, candidate_column)
#   <ds>/tables_all/<id>     per-table CSVs
#   <ds>/labels.json         manifest
#
# Note: The reference `bert_opendata.json` etc cells were produced against the
# opendata_main / opendata_CAN / opendata_USA / opendata_UK_SG splits. The
# user-facing dataset names are the same (`opendata_main`, etc.); the HF
# tables-config name is suffixed with ``_tables``.
_CTBENCH_JOIN_SEARCH_DATASETS: dict[str, str] = {
    "opendata_main":  "opendata_main_tables",
    "opendata_can":   "opendata_can_tables",
    "opendata_usa":   "opendata_usa_tables",
    "opendata_uk_sg": "opendata_uk_sg_tables",
}


# The canonical query-role-disjoint split for join_search_learned is published
# on HF as raw files under ``splits/join_search/<variant>/`` (NOT a parquet
# config — it's a derived artifact, so it doesn't surface via load_dataset).
# Variant names predate the dataset rename (opendata_main -> 'opendata', etc.).
# The split is also reproducible from the staged GT + curated query list via
# ``trl_bench.tasks.join_search.splits.generate_canonical_split``.
_CTBENCH_JOIN_SEARCH_SPLIT_VARIANT: dict[str, str] = {
    "opendata_main":  "opendata",
    "opendata_can":   "opendata_CAN",
    "opendata_usa":   "opendata_USA",
    "opendata_uk_sg": "opendata_UK_SG",
}
_CTBENCH_JOIN_SEARCH_SPLIT_FILES = (
    "train_queries.csv", "test_queries.csv", "test_gt.csv", "split_info.json",
)


def _fetch_join_search_split(
    *, repo: str, variant: str, dst_dir: Path, revision: Optional[str] = None,
) -> None:
    """Download the canonical join-search split from the HF
    ``splits/join_search/<variant>/`` subtree into ``dst_dir`` (the four files
    in ``_CTBENCH_JOIN_SEARCH_SPLIT_FILES``).

    These pre-computed files pin the paper's 20/80 query-role-disjoint split
    consumed by the learned-projection probe (join_search_learned).
    """
    from huggingface_hub import snapshot_download
    snap = snapshot_download(
        repo, repo_type="dataset", revision=revision,
        allow_patterns=f"splits/join_search/{variant}/*",
    )
    src = Path(snap) / "splits" / "join_search" / variant
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fn in _CTBENCH_JOIN_SEARCH_SPLIT_FILES:
        (dst_dir / fn).write_bytes((src / fn).read_bytes())


def _stage_ctbench_join_search_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one join_search dataset onto disk.

    Produces::

        <dst_root>/<dataset>/queries.csv          query_table,query_column
        <dst_root>/<dataset>/ground_truth.csv     query_table,candidate_table,query_column,candidate_column
        <dst_root>/<dataset>/tables_all/<id>.csv
        <dst_root>/<dataset>/labels.json
        <dst_root>/<dataset>/.staged_ok
    """
    from datasets import load_dataset

    if dataset not in _CTBENCH_JOIN_SEARCH_DATASETS:
        raise NotImplementedError(
            f"join_search stager for ctbench dataset {dataset!r} is not wired. "
            f"Wired datasets: {sorted(_CTBENCH_JOIN_SEARCH_DATASETS)}."
        )

    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_all = base / "tables_all"
    tables_all.mkdir(exist_ok=True)

    repo = _repo_for("ctbench")

    # 1) Pair config carries (query_table_id, candidate_table_id, task,
    #    query_column, candidate_column). Filter to task=='join'.
    ds_pair = load_dataset(repo, name=dataset, split="train", revision=revision)
    seen_queries: set[tuple[str, str]] = set()
    query_lines: list[str] = ["query_table,query_column"]
    gt_lines: list[str] = [
        "query_table,candidate_table,query_column,candidate_column"
    ]
    n_join = 0
    for row in ds_pair:
        if row.get("task") != "join":
            continue
        n_join += 1
        qt = row["query_table_id"]
        qc = row["query_column"]
        ct = row["candidate_table_id"]
        cc = row["candidate_column"]
        gt_lines.append(
            f"{_csv_escape(qt)},{_csv_escape(ct)},"
            f"{_csv_escape(qc)},{_csv_escape(cc)}"
        )
        key = (qt, qc)
        if key not in seen_queries:
            seen_queries.add(key)
            query_lines.append(f"{_csv_escape(qt)},{_csv_escape(qc)}")

    # 2) Tables config — materialize every CSV referenced.
    tables_cfg = _CTBENCH_JOIN_SEARCH_DATASETS[dataset]
    ds_tables = load_dataset(repo, name=tables_cfg, split="train", revision=revision)
    seen_csv_hash: dict[str, str] = {}
    for row in ds_tables:
        tid = row["table_id"]
        csv_content = row["csv_text"]
        new_hash = hashlib.md5(csv_content.encode()).hexdigest()
        prior = seen_csv_hash.get(tid)
        if prior is None:
            (tables_all / tid).write_text(csv_content)
            seen_csv_hash[tid] = new_hash
        elif prior != new_hash:
            raise RuntimeError(
                f"table_id collision in ctbench/{dataset} (tables): {tid!r} "
                f"has different content across rows"
            )

    (base / "queries.csv").write_text("\n".join(query_lines) + "\n")
    (base / "ground_truth.csv").write_text("\n".join(gt_lines) + "\n")
    labels_manifest = {
        "queries":      "queries.csv",
        "ground_truth": "ground_truth.csv",
        "tables_all":   "tables_all",
        "n_queries":    len(query_lines) - 1,
        "n_gt":         n_join,
        "n_tables":     len(seen_csv_hash),
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))

    # Fetch the canonical query-disjoint split (consumed by join_search_learned)
    # from the HF splits/join_search/<variant>/ subtree, into
    # <base>/splits/join_search/ (matches task_datasets.yaml split_dir).
    variant = _CTBENCH_JOIN_SEARCH_SPLIT_VARIANT.get(dataset)
    if variant is not None:
        _fetch_join_search_split(
            repo=repo, variant=variant,
            dst_dir=base / "splits" / "join_search", revision=revision,
        )

    (base / ".staged_ok").write_text(
        f"gt_rows={n_join} queries={len(query_lines) - 1} "
        f"tables={len(seen_csv_hash)}\n"
    )
    return base


# == ctbench: semantic_parsing stager ========================================
# WikiTableQuestions semantic_parsing requires preprocessed artifacts that the
# raw ``wtq`` HF config does not carry:
#
#   * tables.jsonl              (~24 MB) KG-encoded table representations
#     produced by the MAPO preprocessing pipeline (token/prop features).
#   * data_split_1/train_split_shard_90-*.jsonl  (90 shards, ~13 MB total)
#     pre-tokenized + entity-tagged training examples.
#   * data_split_1/{train,dev,test}_split.jsonl   pre-shard splits + eval.
#   * saved_programs.json       (~6 MB) pre-searched MAPO program cache.
#
# These are mirrored as raw JSONL/JSON files under ``data/wtq_mapo/`` on
# ``logo-lab/trl-ctbench`` (NOT registered as a parquet config — the schemas
# are deeply nested heterogeneous JSON; raw-file mirror preserves byte
# fidelity for the MAPO runner). The stager uses ``snapshot_download`` with
# ``allow_patterns="data/wtq_mapo/**"`` and rehydrates the layout the runner
# expects: ``<dataset_dir>/{tables.jsonl, saved_programs.json,
# data_split_1/...}`` plus a small ``labels.json`` sentinel so the
# auto-stage existence check finds a file.
#
# User-facing dataset name is ``wiki_table_questions`` (matches the runner's
# ``--task wiki_table_questions`` and the existing registry convention); the
# stager maps it internally to the HF prefix ``data/wtq_mapo/``.

_CTBENCH_SEMPARSE_DATASETS: tuple[str, ...] = ("wiki_table_questions",)
_CTBENCH_SEMPARSE_HF_PREFIX: str = "data/wtq_mapo"
_CTBENCH_SEMPARSE_EXPECTED_SHARDS: int = 90
# Per-table CSV count under ``data/wtq_mapo/tables_all/`` on the HF mirror —
# matches the WikiTableQuestions release (t_200_*.csv … t_204_*.csv).
# Stage-1 column extraction (BERT etc.) iterates this directory.
_CTBENCH_SEMPARSE_EXPECTED_CSVS: int = 2108


def _stage_ctbench_semparse_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one semantic_parsing dataset onto disk from HF.

    Produces (mirroring the MAPO runner's expected layout, plus a
    ``tables_all/`` directory that the column-extractor wrappers consume)::

        <dst_root>/<dataset>/tables.jsonl
        <dst_root>/<dataset>/saved_programs.json
        <dst_root>/<dataset>/data_split_1/train_split_shard_90-*.jsonl  (90)
        <dst_root>/<dataset>/data_split_1/{train,dev,test}_split.jsonl
        <dst_root>/<dataset>/tables_all/<table_id>.csv      (2108)
        <dst_root>/<dataset>/labels.json    (sentinel; the dispatcher derives
                                             <dataset_dir> as labels.json's
                                             parent)
        <dst_root>/<dataset>/.staged_ok

    The stager calls ``huggingface_hub.snapshot_download`` with
    ``allow_patterns="data/wtq_mapo/**"`` to fetch the wtq_mapo subtree,
    then copies the files into the expected on-disk layout. ``tables.jsonl``,
    ``saved_programs.json`` and the ``data_split_1/`` shards mirror the MAPO
    upstream preprocessing exactly; ``tables_all/<table_id>.csv`` is
    the per-table CSV directory from the WikiTableQuestions release (the
    column extractors — BERT, GTE, TabSketchFM, etc. — iterate this
    directory to materialize column embeddings).
    """
    from huggingface_hub import snapshot_download

    if dataset not in _CTBENCH_SEMPARSE_DATASETS:
        raise NotImplementedError(
            f"semantic_parsing stager for ctbench dataset {dataset!r} is not "
            f"wired. Wired datasets: {sorted(_CTBENCH_SEMPARSE_DATASETS)}."
        )

    repo = _repo_for("ctbench")
    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    split_dir = base / "data_split_1"
    split_dir.mkdir(exist_ok=True)

    # 1) Pull the wtq_mapo subtree from HF in one go (snapshot_download caches
    #    bytes under ~/.cache/huggingface; re-runs are fast).
    snapshot_root = snapshot_download(
        repo_id=repo, repo_type="dataset",
        allow_patterns=[f"{_CTBENCH_SEMPARSE_HF_PREFIX}/**"],
        revision=revision,
    )
    src_root = Path(snapshot_root) / _CTBENCH_SEMPARSE_HF_PREFIX
    if not src_root.is_dir():
        raise RuntimeError(
            f"semantic_parsing stager: expected HF subtree at "
            f"{src_root} after snapshot_download but it was not present"
        )

    # 2) Copy/symlink the files into the expected layout. Use shutil.copy2 so
    #    callers can mutate the staged dir without affecting the HF cache.
    src_tables    = src_root / "tables.jsonl"
    src_programs  = src_root / "saved_programs.json"
    src_split_dir = src_root / "data_split_1"
    src_tables_all = src_root / "tables_all"
    if not src_tables.exists():
        raise RuntimeError(
            f"semantic_parsing stager: tables.jsonl missing at {src_tables}"
        )
    if not src_programs.exists():
        raise RuntimeError(
            f"semantic_parsing stager: saved_programs.json missing at "
            f"{src_programs}"
        )
    if not src_split_dir.is_dir():
        raise RuntimeError(
            f"semantic_parsing stager: data_split_1/ missing at {src_split_dir}"
        )
    if not src_tables_all.is_dir():
        raise RuntimeError(
            f"semantic_parsing stager: tables_all/ missing at {src_tables_all}; "
            f"the HF mirror at {_CTBENCH_SEMPARSE_HF_PREFIX} must publish a "
            f"tables_all/ directory of per-table CSVs for the column extractor"
        )

    shutil.copy2(src_tables,   base / "tables.jsonl")
    shutil.copy2(src_programs, base / "saved_programs.json")

    n_shards = 0
    n_splits = 0
    for fpath in sorted(src_split_dir.iterdir()):
        if not fpath.is_file():
            continue
        shutil.copy2(fpath, split_dir / fpath.name)
        if fpath.name.startswith("train_split_shard_90-"):
            n_shards += 1
        elif fpath.name in ("dev_split.jsonl", "test_split.jsonl",
                            "train_split.jsonl"):
            n_splits += 1

    if n_shards != _CTBENCH_SEMPARSE_EXPECTED_SHARDS:
        raise RuntimeError(
            f"semantic_parsing stager: expected "
            f"{_CTBENCH_SEMPARSE_EXPECTED_SHARDS} train_split_shard_90-*.jsonl "
            f"files, got {n_shards} under {split_dir}"
        )

    # 2b) Copy the per-table CSV directory. The column extractors (BERT, GTE,
    #     TabSketchFM, etc.) iterate ``tables_all/*.csv`` to produce column
    #     embeddings; without this directory Stage-1 fails for semantic_parsing.
    tables_all_dir = base / "tables_all"
    tables_all_dir.mkdir(exist_ok=True)
    n_csvs = 0
    for fpath in sorted(src_tables_all.iterdir()):
        if not fpath.is_file() or fpath.suffix != ".csv":
            continue
        shutil.copy2(fpath, tables_all_dir / fpath.name)
        n_csvs += 1

    if n_csvs != _CTBENCH_SEMPARSE_EXPECTED_CSVS:
        raise RuntimeError(
            f"semantic_parsing stager: expected "
            f"{_CTBENCH_SEMPARSE_EXPECTED_CSVS} per-table CSVs under "
            f"tables_all/, got {n_csvs} under {tables_all_dir}"
        )

    # 3) Sentinel ``labels.json`` so the dispatcher's --labels-path resolves
    #    to <base>/labels.json and its parent is the dataset_dir.
    labels_manifest = {
        "tables":         "tables.jsonl",
        "saved_programs": "saved_programs.json",
        "data_split_1":   "data_split_1",
        "tables_all":     "tables_all",
        "n_train_shards": n_shards,
        "n_eval_splits":  n_splits,
        "n_tables":       n_csvs,
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"train_shards={n_shards} eval_splits={n_splits} n_tables={n_csvs}\n"
    )
    return base


# Dispatch table: ctbench task -> per-task stager function. Pair-task families
# are handled by the older ``_stage_ctbench_pair_task`` and are mapped here too
# for a uniform dispatch surface.
_CTBENCH_TASK_DISPATCH: dict[str, str] = {
    "column_type_prediction":     "cta",
    "column_relation_prediction": "cra",
    "table_retrieval":            "retrieval",
    "column_clustering":          "clustering",
    "schema_matching":            "schema_matching",
    "union_search":               "union_search",
    "join_search":                "join_search",
    "semantic_parsing":           "semparse",
}


# == rbench: record_linkage + row_prediction stagers ========================
# rbench publishes two HF configs, NOT one per sub-dataset:
#   "record_linkage"  -> rows tagged by (source, family, pair_id)
#   "row_prediction"  -> rows tagged by (openml_id, dataset_name)
# The user-facing ``dataset`` argument names a sub-dataset; the stager filters
# the parent config by that key. Note that ``trl_bench.data.rbench.load()``
# currently uses a broken colon-subconfig name (``row_prediction:<id>``) — HF
# rejects it. Stagers here bypass that loader and load the parent config
# directly, then filter row-by-row.
#
# Schemas verified on logo-lab/trl-rbench on 2026-05-19:
#   record_linkage row keys:
#     source, family, pair_id, table_a_record_json, table_b_record_json, label
#   row_prediction row keys:
#     openml_id, dataset_name, row_idx, record_json, targets_json,
#     target_specs_json, dataset_metadata_json

_RBENCH_RECORD_LINKAGE_HF_SPLIT_NAMES: dict[str, str] = {
    "train": "train", "valid": "validation", "test": "test",
}
_RBENCH_RECORD_LINKAGE_SPLIT_ORDER: tuple[str, ...] = ("train", "valid", "test")


def _stage_rbench_pair_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one record_linkage RL-dataset onto disk.

    Produces::

        <dst_root>/<dataset>/tables/{tableA.csv, tableB.csv}
        <dst_root>/<dataset>/labels.json     # train/valid/test pair lists
        <dst_root>/<dataset>/metadata.json   # dataset_source / sub_dataset / counts
        <dst_root>/<dataset>/.staged_ok

    ``labels.json`` shape (matches reference
    ``datasets/record_linkage/<dataset>/labels.json``)::

        {"dataset_source": <family>, "sub_dataset": <pretty>,
         "train": [{"table1": {"filename": "tableA.csv", "row_idx": <i>},
                    "table2": {"filename": "tableB.csv", "row_idx": <j>},
                    "label":  <0|1>}, ...],
         "valid": [...], "test": [...]}

    Row contents come from ``table_a_record_json`` / ``table_b_record_json``
    decoded JSON. Each side is deduplicated by row content; the deduplicated
    set is written as a CSV (tableA.csv, tableB.csv). Pair entries reference
    the side and the row index in that CSV.
    """
    from datasets import load_dataset

    repo = _repo_for("rbench")
    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)
    tables_dir = base / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Accumulate per-side deduplicated records and their assigned row_idx.
    # We hash the JSON string to detect duplicates within a side.
    side_rows: dict[str, list[dict]] = {"a": [], "b": []}
    side_row_index: dict[str, dict[str, int]] = {"a": {}, "b": {}}
    side_columns: dict[str, list[str]] = {"a": [], "b": []}

    labels_split: dict[str, list[dict]] = {
        k: [] for k in _RBENCH_RECORD_LINKAGE_SPLIT_ORDER
    }
    family_seen: Optional[str] = None
    n_total_rows = 0

    for label_split in _RBENCH_RECORD_LINKAGE_SPLIT_ORDER:
        hf_split = _RBENCH_RECORD_LINKAGE_HF_SPLIT_NAMES[label_split]
        ds = load_dataset(
            repo, name="record_linkage", split=hf_split, revision=revision,
        )
        for row in ds:
            if row["source"] != dataset:
                continue
            n_total_rows += 1
            if family_seen is None:
                family_seen = row["family"]
            for side, hf_key, csv_filename in (
                ("a", "table_a_record_json", "tableA.csv"),
                ("b", "table_b_record_json", "tableB.csv"),
            ):
                rec_json = row[hf_key]
                rec = json.loads(rec_json) if isinstance(rec_json, str) else rec_json
                key = json.dumps(rec, sort_keys=True)
                idx = side_row_index[side].get(key)
                if idx is None:
                    idx = len(side_rows[side])
                    side_rows[side].append(rec)
                    side_row_index[side][key] = idx
                    # Track column order from the first record of this side; later
                    # records may introduce new keys, which get appended.
                    for k in rec.keys():
                        if k not in side_columns[side]:
                            side_columns[side].append(k)
            entry = {
                "table1": {"filename": "tableA.csv",
                           "row_idx": side_row_index["a"][
                               json.dumps(json.loads(row["table_a_record_json"])
                                          if isinstance(row["table_a_record_json"], str)
                                          else row["table_a_record_json"],
                                          sort_keys=True)
                           ]},
                "table2": {"filename": "tableB.csv",
                           "row_idx": side_row_index["b"][
                               json.dumps(json.loads(row["table_b_record_json"])
                                          if isinstance(row["table_b_record_json"], str)
                                          else row["table_b_record_json"],
                                          sort_keys=True)
                           ]},
                "label": int(row["label"]),
            }
            labels_split[label_split].append(entry)

    if n_total_rows == 0:
        raise RuntimeError(
            f"rbench/record_linkage: no rows for source={dataset!r} found "
            f"in HF config 'record_linkage'. Verify the dataset name."
        )

    # Write per-side CSVs.
    for side, csv_filename in (("a", "tableA.csv"), ("b", "tableB.csv")):
        cols = side_columns[side]
        lines = [",".join(_csv_escape(c) for c in cols)]
        for rec in side_rows[side]:
            lines.append(
                ",".join(_csv_escape(str(rec.get(c, ""))) for c in cols)
            )
        (tables_dir / csv_filename).write_text("\n".join(lines) + "\n")

    labels_doc = {
        "dataset_source": family_seen or "",
        "sub_dataset": dataset,
        **labels_split,
    }
    (base / "labels.json").write_text(json.dumps(labels_doc, indent=2))

    metadata = {
        "dataset_source": family_seen or "",
        "sub_dataset": dataset,
        "tableA_rows": len(side_rows["a"]),
        "tableA_cols": len(side_columns["a"]),
        "tableB_rows": len(side_rows["b"]),
        "tableB_cols": len(side_columns["b"]),
        "columns_A": list(side_columns["a"]),
        "columns_B": list(side_columns["b"]),
        "train_stats": {"n_pairs": len(labels_split["train"])},
        "valid_stats": {"n_pairs": len(labels_split["valid"])},
        "test_stats":  {"n_pairs": len(labels_split["test"])},
    }
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (base / ".staged_ok").write_text(
        f"rows={n_total_rows} A_rows={len(side_rows['a'])} "
        f"B_rows={len(side_rows['b'])}\n"
    )
    return base


# row_prediction: HF parent config "row_prediction" with one row per data row.
# Row keys verified on 2026-05-19:
#   openml_id (int), dataset_name (str), row_idx (int),
#   record_json (json-encoded {feature -> value}),
#   targets_json (json-encoded {label_col -> value}),
#   target_specs_json (json-encoded [{name, role, task_type}, ...]),
#   dataset_metadata_json (json-encoded {schema_version, data, label_columns,
#     splits, labels, ..., benchmark_version})
#
# On-disk layout (reference ``datasets/row_data/openml_<id>/``):
#   data.csv          # full table with feature columns + label columns
#   dataset.json      # decoded dataset_metadata_json (passed through verbatim)
#   splits.npz        # row-index numpy arrays for train/val/test
#   labels.json       # manifest pointing to the above (release-only)
#   .staged_ok
#
# HF split name "validation" maps to on-disk splits.npz key "val" (per reference).
_RBENCH_ROW_PREDICTION_HF_SPLIT_NAMES: dict[str, str] = {
    "train": "train", "val": "validation", "test": "test",
}


def _stage_rbench_row_prediction_task(
    *, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize one row_prediction OpenML dataset onto disk.

    ``dataset`` is the on-disk subdirectory name (e.g. ``openml_3``); the
    numeric OpenML id used to filter HF rows is parsed from the
    ``openml_<id>`` prefix.
    """
    import numpy as np
    from datasets import load_dataset

    if not dataset.startswith("openml_"):
        raise ValueError(
            f"row_prediction dataset name must start with 'openml_'; "
            f"got {dataset!r}"
        )
    try:
        openml_id = int(dataset.removeprefix("openml_"))
    except ValueError as e:
        raise ValueError(
            f"row_prediction dataset {dataset!r}: cannot parse openml_id"
        ) from e

    repo = _repo_for("rbench")
    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)

    # Stream HF rows for this openml_id across the three splits. We collect
    # (split_name, row_idx, record_dict, targets_dict) and the per-dataset
    # metadata blob (same across all rows of one dataset).
    feature_cols: list[str] = []
    label_cols: list[str] = []
    rows_by_idx: dict[int, dict] = {}
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    dataset_metadata: Optional[dict] = None
    n_total = 0

    for on_disk_split, hf_split in _RBENCH_ROW_PREDICTION_HF_SPLIT_NAMES.items():
        ds = load_dataset(
            repo, name="row_prediction", split=hf_split, revision=revision,
        )
        for row in ds:
            if int(row["openml_id"]) != openml_id:
                continue
            n_total += 1
            ridx = int(row["row_idx"])
            rec = json.loads(row["record_json"])
            tgts = json.loads(row["targets_json"])
            if dataset_metadata is None:
                dataset_metadata = json.loads(row["dataset_metadata_json"])
                label_cols = list(dataset_metadata.get("label_columns") or [])
                for k in rec.keys():
                    if k not in feature_cols and k not in label_cols:
                        feature_cols.append(k)
            else:
                # Track any feature/label keys not seen in the first row.
                for k in rec.keys():
                    if k not in feature_cols and k not in label_cols:
                        feature_cols.append(k)
            merged = {**rec, **tgts}
            rows_by_idx[ridx] = merged
            split_indices[on_disk_split].append(ridx)

    if n_total == 0 or dataset_metadata is None:
        raise RuntimeError(
            f"rbench/row_prediction: no rows for openml_id={openml_id} "
            f"({dataset!r}) found in HF config 'row_prediction'."
        )

    # Reconstruct the full data.csv in stable row_idx order. The reference
    # data.csv uses (features..., labels...) ordering with label_columns last.
    all_cols = feature_cols + label_cols
    lines = [",".join(_csv_escape(c) for c in all_cols)]
    for ridx in sorted(rows_by_idx.keys()):
        rec = rows_by_idx[ridx]
        lines.append(
            ",".join(_csv_escape(str(rec.get(c, ""))) for c in all_cols)
        )
    (base / "data.csv").write_text("\n".join(lines) + "\n")

    # Write splits.npz (train/val/test row index arrays, dtype int64).
    npz_kwargs = {
        sp: np.asarray(sorted(idxs), dtype=np.int64)
        for sp, idxs in split_indices.items()
    }
    np.savez(str(base / "splits.npz"), **npz_kwargs)

    # Pass dataset_metadata through verbatim as dataset.json.
    (base / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2))

    # Manifest labels.json so auto-stage's existence check finds a file.
    labels_manifest = {
        "data":         "data.csv",
        "splits":       "splits.npz",
        "dataset_json": "dataset.json",
        "openml_id":    openml_id,
        "label_columns": label_cols,
    }
    (base / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    (base / ".staged_ok").write_text(
        f"rows={n_total} features={len(feature_cols)} "
        f"labels={len(label_cols)}\n"
    )
    return base


# Dispatch table: rbench task -> per-task stager family. Parallel to
# ``_CTBENCH_TASK_DISPATCH``.
_RBENCH_TASK_DISPATCH: dict[str, str] = {
    "record_linkage": "pair_task",
    "row_prediction": "row_prediction",
}


# == dlte: Data-Lake Table Enrichment staging ===============================
# DLTE is a 3-stage pipeline (retrieval -> alignment -> merge). Each stage's
# runner reads from a project-rooted layout::
#
#   <project_root>/datasets/dlte_v1/
#     manifests/lake_manifest.jsonl       (1 line per lake table)
#     manifests/fragments_manifest.jsonl  (1 line per query/target fragment)
#     manifests/parents_filtered.jsonl    (1 line per parent table)
#     ground_truth/query_tasks.jsonl      (1 line per query task with relevant set)
#     ground_truth/table_maps/<table_id>.npz   (per-table column mapping; *NOT*
#                                                in HF — produced by step2)
#     queries/tables/<table_id>.csv       (per-query-fragment CSV)
#     lake/targets/tables/<table_id>.csv  (per-target-fragment CSV; the lake
#                                          tables are stored under lake/)
#     [+ many other artifacts]
#
# HF coverage (logo-lab/trl-dlte, configs ``manifests`` + ``lake`` + ``table_maps``):
#   * `manifests` config: all five record_types — parent, fragment, lake_table,
#     query_task, split_assignment — are present (plus fragmentation_config,
#     ckan_distractor, gt_validation). We can materialize ALL of the JSONL
#     manifest files from this config.
#   * `lake` config: row schema {table_id, kind, parent_id, parent_source,
#     noise_tier, fragment_type, split, csv_text, n_rows, n_cols}. We can
#     materialize all lake/query/target CSVs from this config.
#   * `table_maps` config: per-fragment column-mapping records. Row schema
#     {table_id, row_parent_idx_json, col_parent_idx_json}. JSON-encoded
#     `list[int]` arrays. The stager reconstructs `.npz` files with the
#     original {row_parent_idx, col_parent_idx} int32 arrays under
#     `<project_root>/datasets/dlte_v1/ground_truth/table_maps/<table_id>.npz`
#     so Stages 2 (alignment) and 3 (merge) run end-to-end. Stage 1
#     (retrieval) does not consume table_maps.

_DLTE_VALID_TASKS = frozenset({"dlte_retrieval", "dlte_alignment", "dlte_merge"})
_DLTE_DATASET_NAME = "dlte_v1"


def _stage_dlte_task(
    *, task: str, dataset: str, dst_root: Path, revision: Optional[str] = None,
) -> Path:
    """Materialize the DLTE dataset (manifests + ground-truth + CSVs) onto disk.

    Produces ``<dst_root>/<dataset>/datasets/dlte_v1/`` with the project-rooted
    layout the DLTE runners expect. ``dataset`` is conventionally ``dlte_v1``;
    other values raise.

    The on-disk layout is rooted at ``<dst_root>/<dataset>/`` so the DLTE
    runners' ``--project_root <dst_root>/<dataset>`` resolves
    ``datasets/dlte_v1/`` correctly.

    Returns the manifest-bearing base directory; a manifest ``labels.json`` is
    written there so the auto-stage existence check finds a file. ``labels.json``
    points at the project_root the runners should be invoked with.
    """
    if task not in _DLTE_VALID_TASKS:
        raise ValueError(
            f"unknown dlte task {task!r}. Valid: {sorted(_DLTE_VALID_TASKS)}."
        )
    if dataset != _DLTE_DATASET_NAME:
        raise NotImplementedError(
            f"dlte stager: unsupported dataset {dataset!r}. The HF release only "
            f"publishes {_DLTE_DATASET_NAME!r}."
        )

    from datasets import load_dataset

    repo = _repo_for("dlte")
    base = dst_root / dataset
    base.mkdir(parents=True, exist_ok=True)

    # Project-root layout the DLTE runners read.
    project_dir = base / "datasets" / _DLTE_DATASET_NAME
    manifests_dir = project_dir / "manifests"
    gt_dir = project_dir / "ground_truth"
    queries_csv_dir = project_dir / "queries" / "tables"
    targets_csv_dir = project_dir / "lake" / "targets" / "tables"
    for d in (manifests_dir, gt_dir, queries_csv_dir, targets_csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Materialize the manifests + ground-truth JSONLs from the ``manifests``
    #    HF config. Row schema: {record_type, record_json}.
    manifest_ds = load_dataset(
        repo, name="manifests", split="train", revision=revision,
    )

    record_buckets: dict[str, list[dict]] = {
        "parent": [], "fragment": [], "lake_table": [],
        "query_task": [], "split_assignment": [],
        "ckan_distractor": [], "fragmentation_config": [], "gt_validation": [],
    }
    n_total = 0
    for row in manifest_ds:
        rtype = row["record_type"]
        try:
            payload = json.loads(row["record_json"])
        except Exception:
            continue
        bucket = record_buckets.setdefault(rtype, [])
        bucket.append(payload)
        n_total += 1

    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    _write_jsonl(manifests_dir / "lake_manifest.jsonl",      record_buckets.get("lake_table", []))
    _write_jsonl(manifests_dir / "fragments_manifest.jsonl", record_buckets.get("fragment", []))
    _write_jsonl(manifests_dir / "parents_filtered.jsonl",   record_buckets.get("parent", []))
    _write_jsonl(gt_dir        / "query_tasks.jsonl",        record_buckets.get("query_task", []))
    _write_jsonl(manifests_dir / "split_assignments.jsonl",  record_buckets.get("split_assignment", []))

    # 2) Materialize CSV content from the ``lake`` HF config. Each row carries
    #    {table_id, kind, csv_text, ...}. Place each CSV under the directory
    #    implied by its ``kind`` (query fragments -> queries/tables/,
    #    everything else -> lake/targets/tables/).
    lake_ds = load_dataset(
        repo, name="lake", split="train", revision=revision,
    )
    n_csv = 0
    for row in lake_ds:
        tid = row["table_id"]
        csv_text = row.get("csv_text")
        if not csv_text:
            continue
        kind = (row.get("kind") or "").lower()
        # The DLTE pipeline's fragment role/type taxonomy distinguishes:
        #   * query fragments (seed rows)          -> queries/tables/<table_id>.csv
        #   * lake corpus (target unions/joins +
        #     ckan-distractor tables)              -> lake/targets/tables/<table_id>.csv
        #   * parent tables (full source tables
        #     used by Stage-3 evaluation as the
        #     ground-truth row pool)              -> datasets/<source>/tables/<stem>.csv
        #     where <source> is tabfact/wtq and <stem> is the table_id with the
        #     ``<source>__`` prefix stripped, matching the path stored in
        #     ``parents_filtered.jsonl``'s ``csv_path``.
        # HF upload uses ``query_seed`` for query fragments; we also accept the
        # older sentinel names for backwards compatibility.
        if kind in ("query", "query_fragment", "seed", "query_seed"):
            target = queries_csv_dir / f"{tid}.csv"
        elif kind in ("parent_tabfact", "parent_wtq"):
            parent_source = kind.removeprefix("parent_")  # "tabfact" | "wtq"
            # Strip the ``<source>__`` prefix from table_id to recover the
            # csv_stem the parents manifest uses.
            csv_stem = tid.removeprefix(f"{parent_source}__")
            parent_tables_dir = (
                project_dir.parent.parent / "datasets" / parent_source / "tables"
            )
            parent_tables_dir.mkdir(parents=True, exist_ok=True)
            target = parent_tables_dir / f"{csv_stem}.csv"
        else:
            target = targets_csv_dir / f"{tid}.csv"
        target.write_text(csv_text)
        n_csv += 1

    # 3) Materialize ``table_maps/`` from the ``table_maps`` HF config. Row
    #    schema: {table_id, row_parent_idx_json, col_parent_idx_json}. JSON-
    #    encoded ``list[int]`` arrays are reconstructed as int32 numpy arrays
    #    and saved as `.npz` files under ``ground_truth/table_maps/`` so the
    #    Stage-2 (alignment) and Stage-3 (merge) runners' direct
    #    ``np.load(...)`` calls work. Stage 1 (retrieval) doesn't consume
    #    these files.
    import numpy as np
    table_maps_dir = gt_dir / "table_maps"
    table_maps_dir.mkdir(parents=True, exist_ok=True)
    try:
        table_maps_ds = load_dataset(
            repo, name="table_maps", split="train", revision=revision,
        )
    except Exception as e:
        # The table_maps config was added 2026-05-20; old revisions of the
        # repo may not have it. Older clients should bump the dataset
        # version; we surface this as a soft fail (continue without
        # writing the per-table maps) so retrieval-only callers aren't
        # blocked by the absence of this config.
        print(
            f"warning: dlte stager: 'table_maps' HF config not available "
            f"({e!r}); Stage-2 (alignment) and Stage-3 (merge) will fail.",
        )
        n_maps = 0
    else:
        n_maps = 0
        for row in table_maps_ds:
            tid = row["table_id"]
            try:
                row_idx = np.asarray(
                    json.loads(row["row_parent_idx_json"]), dtype=np.int32,
                )
                col_idx = np.asarray(
                    json.loads(row["col_parent_idx_json"]), dtype=np.int32,
                )
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(
                    f"dlte/table_maps row {tid!r} has malformed "
                    f"row/col_parent_idx_json: {e}"
                ) from e
            np.savez(
                str(table_maps_dir / f"{tid}.npz"),
                row_parent_idx=row_idx,
                col_parent_idx=col_idx,
            )
            n_maps += 1

    # 4) Materialize ``labels.json`` (manifest pointer) so the run.py
    #    auto-stage check finds a file. The DLTE runners do NOT read this
    #    file; they read the JSONL manifests + per-CSV files written above.
    #
    # IMPORTANT layout invariant: labels.json sits at
    #   ``<base>/datasets/dlte_v1/labels.json``
    # NOT at ``<base>/labels.json``. This matches the convention used by the
    # other staging functions (CTA / CRA / retrieval): the dispatcher takes
    # ``labels_path`` and derives the directory the runner reads from by
    # walking up parents. For DLTE the dispatcher walks up 3 parents to
    # reach ``--project_root`` (= ``<base>``), and the runner then looks for
    # manifests at ``<project_root>/datasets/dlte_v1/manifests/``. Writing
    # the sentinel at ``project_dir/labels.json`` (rather than ``base/``)
    # preserves that 3-parent invariant.
    labels_manifest = {
        "project_root": str(base),
        "dataset":      _DLTE_DATASET_NAME,
        "manifests":    "manifests",
        "ground_truth": "ground_truth",
        "queries":      "queries/tables",
        "targets":      "lake/targets/tables",
        # table_maps/ are now materialized from the ``table_maps`` HF config
        # (added 2026-05-20). All three DLTE stages run end-to-end.
        "table_maps_present": n_maps > 0,
        "table_maps_count": n_maps,
    }
    (project_dir / "labels.json").write_text(json.dumps(labels_manifest, indent=2))
    # Sentinel at <base>/.staged_ok so idempotency check finds it via
    # stage_dataset's `(data_root / dataset) / .staged_ok` lookup.
    (base / ".staged_ok").write_text(
        f"manifests_rows={n_total} csv_rows={n_csv} "
        f"table_maps_count={n_maps}\n"
    )
    # Return ``base`` (= ``data_root / dataset``) so the auto-stage idempotency
    # check (run.py: ``labels = base / "labels.json"``) works AND so the
    # dispatcher's 3-parent walk from the labels file lands on ``base``. The
    # caller (run.py::_resolve_labels_path) must then use the labels.json
    # that lives at ``base / "datasets" / "dlte_v1" / "labels.json"``, NOT
    # ``base / "labels.json"``.
    return base


# Dispatch table: dlte task -> per-task stager family. All three dlte tasks
# share the same on-disk layout (the runners read from a project-rooted tree),
# so they all dispatch through the same stager.
_DLTE_TASK_DISPATCH: dict[str, str] = {
    "dlte_retrieval": "dlte",
    "dlte_alignment": "dlte",
    "dlte_merge":     "dlte",
}


# == Public API ============================================================

def stage_dataset(
    *, suite: str, task: str, dataset: str,
    data_root: str | Path,
    revision: Optional[str] = None, force: bool = False,
) -> Path:
    """Materialize one (suite, task, dataset) onto disk under ``data_root``.

    Returns the dataset root path (``<data_root>/<dataset>/<inner-subdir>``).

    Idempotent: skips work if ``.staged_ok`` already exists unless
    ``force=True``.
    """
    if suite not in _SUITES:
        raise ValueError(f"unknown suite: {suite!r}. Valid: {sorted(_SUITES)}")

    data_root = Path(data_root)

    if suite == "ctbench":
        # CTA / CRA / retrieval all live at <data_root>/<dataset>/ directly
        # (no inner subdir); pair tasks may add an inner subdir.
        #
        # Datasets that have BOTH a CTA and a CRA HF config (currently only
        # ``sotab`` — its CTA config is column-type-prediction shape and its
        # ``sotab_relation`` config is column-relation-prediction shape) need
        # task-disambiguated staging directories so the two layouts don't
        # collide. We tack a ``_cra`` suffix onto the on-disk dataset name
        # for the CRA family in that case; user-facing dataset name passed
        # to the runner is unchanged. The dispatcher in run.py reads
        # ``labels_path`` (returned here) directly so callers don't have to
        # know the on-disk naming.
        family = _CTBENCH_TASK_DISPATCH.get(task)
        if family is not None:
            on_disk_dataset = dataset
            if family == "cra" and dataset in _CTBENCH_CTA_DATASETS:
                on_disk_dataset = f"{dataset}_cra"
            base = data_root / on_disk_dataset
            # column_clustering shares data/<ds>/ + the per-column embeddings
            # with CTA for sato/sotab (it only ADDS all.csv). A task-specific
            # sentinel keeps the two stagings independent so neither
            # short-circuits the other on a shared ".staged_ok".
            sentinel_name = (
                ".staged_ok_clustering" if family == "clustering" else ".staged_ok"
            )
            sentinel = base / sentinel_name
            if sentinel.exists() and not force:
                return base
            if family == "cta":
                return _stage_ctbench_cta_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            if family == "cra":
                return _stage_ctbench_cra_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                    on_disk_dataset=on_disk_dataset,
                )
            if family == "retrieval":
                return _stage_ctbench_retrieval_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            if family == "clustering":
                return _stage_ctbench_clustering_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            if family == "schema_matching":
                return _stage_ctbench_schema_matching_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            if family == "union_search":
                return _stage_ctbench_union_search_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            if family == "join_search":
                return _stage_ctbench_join_search_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            if family == "semparse":
                return _stage_ctbench_semparse_task(
                    dataset=dataset, dst_root=data_root, revision=revision,
                )
            raise AssertionError(f"unreachable: family {family!r}")

        inner = _CTBENCH_INNER_SUBDIR.get(dataset)
        base = data_root / dataset
        if inner is not None:
            base = base / inner
        sentinel = base / ".staged_ok"
        if sentinel.exists() and not force:
            return base
        if task in _CTBENCH_PAIR_TASK_FIELDS:
            return _stage_ctbench_pair_task(
                task=task, dataset=dataset,
                dst_root=data_root, revision=revision,
            )
        raise NotImplementedError(
            f"ctbench task {task!r} stager is not supported in this release; "
            f"see docs/USAGE.md"
        )

    if suite == "rbench":
        family = _RBENCH_TASK_DISPATCH.get(task)
        if family is None:
            raise NotImplementedError(
                f"rbench task {task!r} stager is not wired. Wired tasks: "
                f"{sorted(_RBENCH_TASK_DISPATCH)}."
            )
        base = data_root / dataset
        sentinel = base / ".staged_ok"
        if sentinel.exists() and not force:
            return base
        if family == "pair_task":
            return _stage_rbench_pair_task(
                dataset=dataset, dst_root=data_root, revision=revision,
            )
        if family == "row_prediction":
            return _stage_rbench_row_prediction_task(
                dataset=dataset, dst_root=data_root, revision=revision,
            )
        raise AssertionError(f"unreachable: rbench family {family!r}")

    if suite == "dlte":
        family = _DLTE_TASK_DISPATCH.get(task)
        if family is None:
            raise NotImplementedError(
                f"dlte task {task!r} stager is not wired. Wired tasks: "
                f"{sorted(_DLTE_TASK_DISPATCH)}."
            )
        base = data_root / dataset
        sentinel = base / ".staged_ok"
        if sentinel.exists() and not force:
            return base
        if family == "dlte":
            return _stage_dlte_task(
                task=task, dataset=dataset, dst_root=data_root, revision=revision,
            )
        raise AssertionError(f"unreachable: dlte family {family!r}")

    raise NotImplementedError(
        f"suite {suite!r} stager is not supported in this release; "
        f"see docs/USAGE.md"
    )


__all__ = ["stage_dataset"]
