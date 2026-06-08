"""Regression tests for the TabSketchFM embedding-repair degenerate-column gate.

Root cause this guards against
==============================
TabSketchFM's ``preprocess_cols`` (``data_prep.py``) INTENTIONALLY drops a
*numeric* (INTEGER/FLOAT) column -- producing no token, hence no column
embedding -- when the column is degenerate:

    if all_na or len(df[col]) <= 1 or c['unique'] == 1:
        continue

So an all-constant numeric column (e.g. every cell ``5``) or a single-data-row
numeric column yields NO embedding from the model. The two-pass embedding repair
re-embeds the "missing" columns; TabSketchFM drops them again. The repair must
treat these intentionally-dropped degenerate columns as a **zero-vector
fallback** (exactly like a truly-empty column), NOT as a failure.

CRITICAL: the repair must still distinguish

  * degenerate-dropped (all-NaN / single-unique / len<=1 numeric) -> zero-vector,
    NOT a failure, from
  * a GENUINE missing column (a column that *should* have embedded: numeric with
    >=2 distinct values, or a string column with real content) -> still a
    failure that propagates and raises.

These tests stub the model so it embeds only the columns TabSketchFM would keep,
reproducing the real drop behaviour without loading the checkpoint.

Confirmed on real ckan_subset data (table ``yht4-twf4.csv.0.neg.csv`` col 102
``CAU_HOMOL``: 334 rows, single unique value ``9``): the model returns NO
embedding for it; pre-fix the repair mis-handled it (assigned the table's
metadata-segment vector, or in multi-column chunks raised "non-empty failure"),
post-fix it is a clean zero-vector. ``run.py`` invokes the repair WITHOUT
``--allow_failures``, so any such misclassified-as-failure column aborts the
whole one-command path before downstream runs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trl_bench.utils.embedding_repair.adapters.tabsketchfm import TabSketchFMAdapter
from trl_bench.utils.embedding_repair.core import repair_embeddings


_DIM = 8


class _FakeConfig:
    hidden_size = _DIM


class _FakeInner:
    config = _FakeConfig()


class _FakeModel:
    # Mirrors the real embedder's nested attribute path
    # (embedder.model.model.config.hidden_size) so the adapter's dim-inference
    # fallback resolves to _DIM when a lone degenerate column leaves mapping empty.
    model = _FakeInner()


class _FakeTSFMEmbedder:
    """Mimics TabSketchFMEmbedder.encode_csv: drops degenerate NUMERIC columns
    exactly as ``preprocess_cols`` does, and never emits an embedding for them.

    The returned ``column_embeddings`` dict is keyed by the *local* position of
    the kept columns within the subset CSV that the adapter wrote (the adapter
    re-keys those back to original indices). To match the real embedder, which
    keys by the surviving column's position in the table it tokenised, we emit a
    dense 0..k-1 keying over the *kept* columns only.
    """

    model = _FakeModel()

    def encode_csv(self, csv_path: str) -> dict:
        df = pd.read_csv(csv_path, dtype=str)
        kept_embs: dict[int, np.ndarray] = {}
        local = 0
        for col in df.columns:
            # Drop EXACTLY the columns TabSketchFM's preprocess_cols drops.
            # Delegate to the adapter's classifier so the fixture cannot drift
            # from the production drop predicate (single source of truth).
            if TabSketchFMAdapter._is_degenerate_column(df[col].replace("", np.nan)):
                # TabSketchFM emits no token -> no embedding for this column.
                continue
            kept_embs[local] = np.full(_DIM, float(local + 1), dtype=np.float32)
            local += 1
        return {"column_embeddings": kept_embs}


def _make_adapter() -> TabSketchFMAdapter:
    adapter = TabSketchFMAdapter(checkpoint_path="/unused.ckpt", device="cpu")
    adapter._embedder = _FakeTSFMEmbedder()  # bypass _load_model / checkpoint
    return adapter


def _write_csv(tmp_path: Path, name: str, frame: dict) -> str:
    df = pd.DataFrame(frame)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# embed_columns-level behaviour
# ---------------------------------------------------------------------------

def test_embed_columns_degenerate_numeric_gets_zero_vector(tmp_path):
    """A single-unique numeric column TabSketchFM drops must come back as a
    zero-vector (benign fallback), NOT be left missing/flagged as a failure."""
    # col0: kept (2 distinct strings), col1: degenerate numeric (all 5),
    # col2: single-row-equivalent degenerate numeric (only one non-null value).
    csv = _write_csv(
        tmp_path,
        "t.csv",
        {
            "a": ["alpha", "beta", "gamma"],
            "const_num": ["5", "5", "5"],
            "one_num": ["7", "", ""],
        },
    )
    adapter = _make_adapter()
    mapping = adapter.embed_columns(csv, [0, 1, 2], max_rows=None)

    # All three indices must be present (none left "missing").
    assert set(mapping.keys()) == {0, 1, 2}, mapping.keys()
    # The two degenerate numeric columns are zero-vectors.
    assert np.allclose(mapping[1], 0.0), "const numeric col must be zero-vector"
    assert np.allclose(mapping[2], 0.0), "single-value numeric col must be zero-vector"
    # The genuine kept column is NOT a zero-vector.
    assert not np.allclose(mapping[0], 0.0), "real column must keep its embedding"


def test_embed_columns_all_blank_still_zero_vector(tmp_path):
    """Pre-existing behaviour preserved: a fully blank column -> zero-vector."""
    csv = _write_csv(
        tmp_path,
        "blank.csv",
        {"a": ["x", "y"], "blank": ["", ""]},
    )
    adapter = _make_adapter()
    mapping = adapter.embed_columns(csv, [0, 1], max_rows=None)
    assert set(mapping.keys()) == {0, 1}
    assert np.allclose(mapping[1], 0.0)


def test_embed_columns_genuine_missing_not_zero_vectored(tmp_path):
    """A GENUINE missing column (numeric, >=2 distinct values) that the model
    fails to embed must be left MISSING (so the core repair raises) -- it must
    NOT be silently zero-vectored. We force the failure by making the fake
    embedder drop an otherwise-keepable column."""
    csv = _write_csv(
        tmp_path,
        "g.csv",
        {"a": ["alpha", "beta"], "real_num": ["1", "2"]},
    )

    class _DropRealEmbedder(_FakeTSFMEmbedder):
        def encode_csv(self, csv_path):
            # Emulate a model that returns NOTHING (a real extraction failure),
            # even though both columns are non-degenerate and should embed.
            return {"column_embeddings": {}}

    adapter = TabSketchFMAdapter(checkpoint_path="/unused.ckpt", device="cpu")
    adapter._embedder = _DropRealEmbedder()
    mapping = adapter.embed_columns(csv, [0, 1], max_rows=None)
    # Neither column is degenerate, so neither may be zero-vectored: both stay
    # missing and surface as failures upstream.
    assert 0 not in mapping and 1 not in mapping, (
        f"genuine non-degenerate columns must NOT be zero-vectored: {mapping}"
    )


# ---------------------------------------------------------------------------
# End-to-end repair_embeddings behaviour (the gate run.py hits)
# ---------------------------------------------------------------------------

def _write_pickle(path: Path, records: list) -> None:
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=4)


def test_repair_does_not_raise_for_degenerate_columns(tmp_path):
    """The one-command path's gate: a table whose only 'missing' columns are
    TabSketchFM-degenerate must repair cleanly (no RuntimeError), since
    ``run.py`` invokes repair WITHOUT ``--allow_failures``."""
    csv = _write_csv(
        tmp_path,
        "deg.csv",
        {"a": ["alpha", "beta", "gamma"], "const_num": ["5", "5", "5"]},
    )
    # Record already has the kept column (idx 0) but is missing the degenerate
    # numeric column (idx 1) -- exactly what extraction produces.
    record = {
        "table": csv,
        "table_id": "deg",
        "table_name": "deg",
        "column_names": ["a", "const_num"],
        "column_embeddings": {0: np.ones(_DIM, dtype=np.float32)},
        "table_embedding": {
            "cls_embedding": np.ones(_DIM, dtype=np.float32),
            "table_embedding": None,
            "column_mean": np.ones(_DIM, dtype=np.float32),
            "token_mean": np.ones(_DIM, dtype=np.float32),
        },
    }
    pkl = tmp_path / "deg.pkl"
    _write_pickle(pkl, [record])

    adapter = _make_adapter()
    summary = repair_embeddings(
        adapter, pkl, allow_failures=False, chunk_size=64,
    )
    assert summary["failures"] == {}, summary["failures"]

    with open(pkl, "rb") as f:
        out = pickle.load(f)
    embs = out[0]["column_embeddings"]
    # Both columns now present; the degenerate one is a zero-vector.
    assert 0 in embs and 1 in embs
    assert np.allclose(embs[1], 0.0)


def test_repair_still_raises_for_genuine_missing_column(tmp_path):
    """A genuine non-degenerate missing column must STILL raise RuntimeError
    under the same no-allow_failures gate (no blanket suppression)."""
    csv = _write_csv(
        tmp_path,
        "real.csv",
        {"a": ["alpha", "beta", "gamma"], "real_num": ["1", "2", "3"]},
    )
    record = {
        "table": csv,
        "table_id": "real",
        "table_name": "real",
        "column_names": ["a", "real_num"],
        "column_embeddings": {0: np.ones(_DIM, dtype=np.float32)},
        "table_embedding": {
            "cls_embedding": np.ones(_DIM, dtype=np.float32),
            "table_embedding": None,
            "column_mean": np.ones(_DIM, dtype=np.float32),
            "token_mean": np.ones(_DIM, dtype=np.float32),
        },
    }
    pkl = tmp_path / "real.pkl"
    _write_pickle(pkl, [record])

    class _DropRealEmbedder(_FakeTSFMEmbedder):
        def encode_csv(self, csv_path):
            return {"column_embeddings": {}}  # genuine failure: nothing back

    adapter = TabSketchFMAdapter(checkpoint_path="/unused.ckpt", device="cpu")
    adapter._embedder = _DropRealEmbedder()

    with pytest.raises(RuntimeError):
        repair_embeddings(adapter, pkl, allow_failures=False, chunk_size=64)
