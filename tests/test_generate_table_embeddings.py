"""Tests for the Stage-2 table-embedding aggregator.

The fast tests synthesize a small in-memory column pickle and verify that
extract_table_embeddings emits the expected per-table dict layout.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pytest

from trl_bench.scripts.generate_table_embeddings import (
    discover_column_embeddings,
    extract_table_embeddings,
    process_model_dataset,
)


# == fast / hermetic tests ===================================================

def _write_column_pickle(path: Path, tables: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_extract_recomputes_column_mean_when_absent(tmp_path):
    col_pkl = tmp_path / "col" / "bert" / "tiny.pkl"
    _write_column_pickle(col_pkl, [{
        "table_id": "t1",
        "table_embedding": {        # only cls available, no column_mean.
            "cls_embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        "column_embeddings": {
            "col_a": np.array([2.0, 4.0, 6.0], dtype=np.float32),
            "col_b": np.array([4.0, 8.0, 12.0], dtype=np.float32),
        },
        "model_name": "bert",
        "embedding_dim": 3,
    }])
    out = extract_table_embeddings(col_pkl)
    assert len(out) == 1
    item = out[0]
    assert item["table_id"]   == "t1"
    assert item["model_name"] == "bert"
    assert item["embedding_dim"] == 3

    emb = item["table_embedding"]
    assert isinstance(emb, dict)
    np.testing.assert_array_equal(emb["cls_embedding"],
                                  np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(emb["column_mean"],
                                  np.array([3.0, 6.0, 9.0], dtype=np.float32))
    assert emb["table_embedding"] is None
    assert emb["token_mean"]      is None


def test_extract_preserves_precomputed_aggregations(tmp_path):
    cls = np.array([0.1, 0.2], dtype=np.float32)
    tbl = np.array([0.5, 0.6], dtype=np.float32)
    cm  = np.array([0.9, 1.0], dtype=np.float32)
    tm  = np.array([1.3, 1.4], dtype=np.float32)
    col_pkl = tmp_path / "col" / "m" / "ds.pkl"
    _write_column_pickle(col_pkl, [{
        "table_id": "t1",
        "table_embedding": {
            "cls_embedding": cls, "table_embedding": tbl,
            "column_mean": cm,    "token_mean": tm,
        },
        "model_name": "m", "embedding_dim": 2,
    }])
    emb = extract_table_embeddings(col_pkl)[0]["table_embedding"]
    np.testing.assert_array_equal(emb["cls_embedding"],   cls)
    np.testing.assert_array_equal(emb["table_embedding"], tbl)
    np.testing.assert_array_equal(emb["column_mean"],     cm)
    np.testing.assert_array_equal(emb["token_mean"],      tm)


def test_discover_skips_checkpoint_and_shard_files(tmp_path):
    col_root = tmp_path / "col"
    for rel in ("bert/foo.pkl", "bert/bar.pkl", "bert/foo.checkpoint.pkl",
                "bert/foo_shard1of3.pkl", "gte/foo.pkl"):
        p = col_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    discovered = discover_column_embeddings(col_root)
    assert sorted(discovered) == ["bert", "gte"]
    assert sorted(discovered["bert"]) == ["bar", "foo"]
    assert sorted(discovered["gte"])  == ["foo"]


def test_process_model_dataset_writes_output(tmp_path):
    cls = np.array([1.0, 2.0], dtype=np.float32)
    col_pkl = tmp_path / "col" / "bert" / "ds.pkl"
    _write_column_pickle(col_pkl, [{
        "table_id": "t1",
        "table_embedding": {"cls_embedding": cls},
        "column_embeddings": {},
        "model_name": "bert", "embedding_dim": 2,
    }])
    out_root = tmp_path / "table"
    ok = process_model_dataset(
        "bert", "ds", tmp_path / "col", out_root,
    )
    assert ok is True
    out_pkl = out_root / "bert" / "ds.pkl"
    assert out_pkl.exists()
    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    assert len(data) == 1
    np.testing.assert_array_equal(data[0]["table_embedding"]["cls_embedding"], cls)


def test_process_model_dataset_skips_when_output_exists(tmp_path, capsys):
    col = tmp_path / "col" / "bert" / "ds.pkl"
    _write_column_pickle(col, [{"table_id": "t", "table_embedding": {},
                                "model_name": "bert", "embedding_dim": 0}])
    out = tmp_path / "table" / "bert" / "ds.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"existing")
    ok = process_model_dataset("bert", "ds", tmp_path / "col", tmp_path / "table")
    assert ok is True
    assert out.read_bytes() == b"existing"   # not overwritten
