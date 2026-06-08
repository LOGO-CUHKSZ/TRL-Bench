"""TaBERT adapter for embedding repair."""

from __future__ import annotations

import csv
import tempfile
from typing import Dict, List, Optional

import numpy as np

from .base import BaseAdapter
from ..io_utils import read_csv_subset


class TabertAdapter(BaseAdapter):
    def __init__(
        self,
        checkpoint_path: str,
        max_rows: int = 100,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self.max_rows = max_rows
        self._embedder = None

    def name(self) -> str:
        return "tabert"

    def _load_model(self):
        if self._embedder is not None:
            return
        from trl_bench.models.tabert.generate_column_embeddings import TaBERTEmbedder
        self._embedder = TaBERTEmbedder(self.checkpoint_path, self.device)

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

        with tempfile.TemporaryDirectory(prefix="tabert_repair_") as tmpdir:
            tmp_csv = f"{tmpdir}/subset.csv"
            df = df.fillna("")
            # Quote all fields so blank/whitespace-only rows aren't dropped on read.
            df.to_csv(tmp_csv, index=False, quoting=csv.QUOTE_ALL)
            result = self._embedder.encode_csv(
                tmp_csv,
                context_mode="column",
                max_rows=max_rows or self.max_rows,
                trim_long_table=True,
            )

        col_embeddings = result.get("column_embeddings", {})
        mapping: Dict[int, np.ndarray] = {}
        for local_idx in sorted(col_embeddings.keys()):
            if local_idx >= len(ordered):
                break
            mapping[int(ordered[local_idx])] = col_embeddings[local_idx]
        return mapping
