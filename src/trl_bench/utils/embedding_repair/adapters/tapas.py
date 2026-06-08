"""TAPAS adapter for embedding repair."""

from __future__ import annotations

import tempfile
from typing import Dict, List, Optional

import numpy as np

from .base import BaseAdapter
from ..io_utils import read_csv_subset


class TapasAdapter(BaseAdapter):
    def __init__(
        self,
        model_name: str = "google/tapas-base",
        max_rows: int = 100,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.max_rows = max_rows
        self.max_length = max_length
        self._embedder = None

    def name(self) -> str:
        return "tapas"

    def _load_model(self):
        if self._embedder is not None:
            return
        from trl_bench.models.tapas.generate_column_embeddings import TAPASEmbedder
        self._embedder = TAPASEmbedder(
            model_name=self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    def detect_missing(self, record, header):
        col_embs = self.get_column_embeddings(record)
        missing = []
        for i in range(len(header)):
            vec = col_embs.get(i)
            if vec is None:
                missing.append(i)
                continue
            if isinstance(vec, np.ndarray):
                if np.linalg.norm(vec) < 1e-8:
                    missing.append(i)
            else:
                try:
                    arr = np.array(vec)
                    if np.linalg.norm(arr) < 1e-8:
                        missing.append(i)
                except Exception:
                    missing.append(i)
        return missing

    def is_missing_vector(self, vec: np.ndarray) -> bool:
        try:
            arr = np.array(vec)
            return np.linalg.norm(arr) < 1e-8
        except Exception:
            return True

    def embed_columns(
        self,
        table_path: str,
        col_indices: List[int],
        max_rows: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        self._load_model()
        ordered = sorted(set(int(i) for i in col_indices))
        df = read_csv_subset(table_path, ordered, max_rows=max_rows or self.max_rows)
        if df.empty or len(df.columns) == 0:
            return {}

        with tempfile.TemporaryDirectory(prefix="tapas_repair_") as tmpdir:
            tmp_csv = f"{tmpdir}/subset.csv"
            df.to_csv(tmp_csv, index=False)
            result = self._embedder.encode_csv(
                tmp_csv,
                question=None,
                max_rows=max_rows or self.max_rows,
            )

        col_embeddings = result.get("column_embeddings", {})
        mapping: Dict[int, np.ndarray] = {}
        for local_idx in sorted(col_embeddings.keys()):
            if local_idx >= len(ordered):
                break
            mapping[int(ordered[local_idx])] = col_embeddings[local_idx]
        return mapping
