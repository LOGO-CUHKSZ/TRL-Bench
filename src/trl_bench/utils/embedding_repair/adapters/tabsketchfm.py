"""TabSketchFM adapter for embedding repair."""

from __future__ import annotations

import csv
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np

from .base import BaseAdapter
from ..io_utils import read_csv_subset


def _safe_float(value: str):
    """Parse ``value`` as a finite float, else return None.

    Used to mirror TabSketchFM's numeric-column detection. NaN/inf tokens parse
    to non-finite floats; we reject them so such a column is treated as STRING
    (which TabSketchFM keeps), never as a degenerate numeric drop.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


class TabSketchFMAdapter(BaseAdapter):
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self._embedder = None

    def name(self) -> str:
        return "tabsketchfm"

    def _infer_offset(self, record, header_len: Optional[int] = None) -> int:
        col_embs = record.get("column_embeddings", {}) if isinstance(record, dict) else {}
        if header_len is None:
            names = record.get("column_names") if isinstance(record, dict) else None
            header_len = len(names) if names else None
        if header_len is None or not col_embs:
            return 0
        try:
            max_key = max(int(k) for k in col_embs.keys())
        except Exception:
            return 0
        if 0 in col_embs and max_key >= header_len:
            return 1
        return 0

    def get_column_embeddings(self, record):
        col_embs = record.get("column_embeddings", {}) if isinstance(record, dict) else {}
        offset = self._infer_offset(record)
        if offset == 1:
            shifted = {}
            for key, value in col_embs.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if idx == 0:
                    continue
                shifted[idx - 1] = value
            return shifted
        return super().get_column_embeddings(record)

    def detect_missing(self, record, header):
        col_embs = self.get_column_embeddings(record)
        missing = []
        for i in range(len(header)):
            if i not in col_embs:
                missing.append(i)
        return missing

    def _load_model(self):
        if self._embedder is not None:
            return
        from importlib import util as importlib_util
        from pathlib import Path

        script_path = Path(__file__).resolve().parents[3] / "models" / "tabsketchfm" / "generate_column_embeddings.py"
        spec = importlib_util.spec_from_file_location("tabsketchfm_generate_column_embeddings", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load TabSketchFM module from {script_path}")
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TabSketchFMEmbedder = getattr(module, "TabSketchFMEmbedder")
        self._embedder = TabSketchFMEmbedder(self.checkpoint_path, self.device)

    def embed_columns(
        self,
        table_path: str,
        col_indices: List[int],
        max_rows: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        self._load_model()
        ordered = sorted(set(int(i) for i in col_indices))
        df = read_csv_subset(table_path, ordered, max_rows=max_rows)
        if df.empty or len(df.columns) == 0:
            return {}
        # Classify each requested column as DEGENERATE iff TabSketchFM's
        # ``preprocess_cols`` would intentionally drop it (producing no token
        # and therefore no column embedding). That routine drops a *numeric*
        # (INTEGER/FLOAT) column when it is ``all_na or len(col) <= 1 or
        # unique == 1`` (data_prep.py), and a fully-blank column of any type
        # never yields content tokens either. A degenerate column legitimately
        # comes back empty from the model, so it must get a ZERO-VECTOR
        # fallback rather than be flagged a repair failure. A non-degenerate
        # column that comes back empty is a GENUINE failure and must propagate
        # (so the no-``allow_failures`` repair raises, as before).
        degenerate_by_idx: Dict[int, bool] = {}
        kept_indices: List[int] = []  # requested cols TabSketchFM will tokenize
        for pos, col in enumerate(df.columns):
            idx = ordered[pos]
            degenerate_by_idx[idx] = self._is_degenerate_column(df[col])
            if not degenerate_by_idx[idx]:
                kept_indices.append(idx)

        with tempfile.TemporaryDirectory(prefix="tabsketchfm_repair_") as tmpdir:
            tmp_csv = f"{tmpdir}/subset.csv"
            df = df.fillna("")
            # Quote all fields so blank/whitespace-only rows aren't dropped on read.
            df.to_csv(tmp_csv, index=False, quoting=csv.QUOTE_ALL)
            result = self._embedder.encode_csv(tmp_csv)

        col_embeddings = result.get("column_embeddings", {})
        mapping: Dict[int, np.ndarray] = {}
        # Drop the leading "table name" metadata segment (local key 0) when the
        # model emitted one extra embedding beyond the columns it tokenized.
        local_offset = 1 if (
            len(col_embeddings) in (len(df.columns) + 1, len(kept_indices) + 1)
            and 0 in col_embeddings
        ) else 0
        content_keys = sorted(k for k in col_embeddings if not (local_offset == 1 and k == 0))
        content_embs = [col_embeddings[k] for k in content_keys]

        if len(content_embs) == len(ordered):
            # No columns were dropped: positional 1:1 mapping over all requested.
            for adj_idx, vec in enumerate(content_embs):
                mapping[int(ordered[adj_idx])] = vec
        elif len(content_embs) == len(kept_indices):
            # Some degenerate columns were dropped: the model returns one
            # embedding per KEPT column, densely keyed in column order. Map them
            # back to the kept original indices (the degenerate ones are handled
            # by the zero-vector fallback below). This avoids the off-by-N
            # mis-assignment a naive ``ordered[local_idx]`` map would cause.
            for adj_idx, vec in enumerate(content_embs):
                mapping[int(kept_indices[adj_idx])] = vec
        else:
            # Ambiguous count (an unexpected drop/extra). Map only the prefix we
            # can trust positionally over kept columns; leave the rest missing so
            # the core's per-column retry resolves them one at a time (where the
            # 1-in / <=1-out keying is unambiguous).
            for adj_idx, vec in enumerate(content_embs):
                if adj_idx < len(kept_indices):
                    mapping[int(kept_indices[adj_idx])] = vec

        missing = [idx for idx in ordered if idx not in mapping]
        if missing:
            if mapping:
                sample = next(iter(mapping.values()))
                dim = int(sample.shape[-1]) if hasattr(sample, "shape") else len(sample)
            else:
                dim = getattr(getattr(getattr(self._embedder, "model", None), "model", None), "config", None)
                dim = int(getattr(dim, "hidden_size", 768))
            for idx in missing:
                if degenerate_by_idx.get(idx, False):
                    print(
                        f"[repair][tabsketchfm] Degenerate column zero-vector fallback "
                        f"for {table_path} idx={idx}",
                        file=sys.stderr,
                    )
                    mapping[idx] = np.zeros(dim, dtype=np.float32)
                else:
                    print(
                        f"[repair][tabsketchfm] Missing non-degenerate column for {table_path} idx={idx}; "
                        "will fail after retries.",
                        file=sys.stderr,
                    )
        return mapping

    @staticmethod
    def _is_degenerate_column(series) -> bool:
        """Return True iff TabSketchFM's ``preprocess_cols`` would drop this
        column (no token => no embedding).

        Mirrors ``data_prep.preprocess_cols``: a numeric column is dropped when
        ``all_na or len(non_null) <= 1 or n_unique == 1``; a fully-blank column
        (any type) likewise contributes no content tokens. The subset is read
        with ``dtype=str``, so numeric detection coerces and requires *every*
        non-blank cell to parse as a number (matching pandas' ``infer_dtype``
        treating any non-numeric token as a string column, which TabSketchFM
        keeps and tokenizes by words).
        """
        non_null = series.dropna()
        # Treat whitespace-only cells as blanks, consistent with the CSV the
        # model is fed (blank strings became NaN -> '' on write).
        non_blank = non_null[non_null.astype(str).str.strip() != ""]
        if non_blank.empty:
            return True  # all-blank / all-NaN: no content tokens
        coerced = np.asarray(
            [_safe_float(v) for v in non_blank.astype(str).tolist()], dtype=object
        )
        if any(v is None for v in coerced):
            # A non-numeric token => TabSketchFM types it STRING and keeps it
            # (tokenized by words), so it is NOT a degenerate-drop candidate.
            return False
        numeric_vals = [float(v) for v in coerced]
        if len(numeric_vals) <= 1:
            return True  # len(df[col]) <= 1
        if len(set(numeric_vals)) == 1:
            return True  # c['unique'] == 1
        return False

    def update_record(self, record, header, column_embeddings):
        offset = self._infer_offset(record, len(header))
        if offset == 1:
            raw = record.get("column_embeddings", {})
            extra = raw.get(0)
            rebuilt = {}
            if extra is not None:
                rebuilt[0] = extra
            for idx, vec in column_embeddings.items():
                rebuilt[int(idx) + 1] = vec
            record["column_embeddings"] = rebuilt
            record["column_names"] = [str(c) for c in header]
            record["table_embedding"] = self.recompute_table_embedding(record, rebuilt)
            return record

        return super().update_record(record, header, column_embeddings)
