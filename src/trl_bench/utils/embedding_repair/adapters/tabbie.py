"""TABBIE adapter for embedding repair."""

from __future__ import annotations

import sys
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from .base import BaseAdapter
from ..io_utils import read_csv_subset


class TabbieAdapter(BaseAdapter):
    def __init__(
        self,
        checkpoint_path: str,
        max_rows: int = 30,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self.max_rows = max_rows
        self._embedder = None

    def name(self) -> str:
        return "tabbie"

    def recommended_chunk_size(self) -> Optional[int]:
        # TABBIE's grid is capped at 20 columns
        return 20

    def _load_model(self):
        if self._embedder is not None:
            return

        # TABBIE's internal imports need its directory on sys.path,
        # and it has a local utils/ that shadows the project-level one.
        # The TABBIEEmbedder import must happen with TABBIE's dir on path.
        tabbie_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "models", "tabbie"
        )
        tabbie_dir = os.path.abspath(tabbie_dir)

        # Temporarily add TABBIE's dir and import
        old_path = sys.path[:]
        sys.path.insert(0, tabbie_dir)
        try:
            from csv_to_embeddings import TABBIEEmbedder
        finally:
            sys.path[:] = old_path

        # Translate device string to device_id for TABBIEEmbedder
        device_id = None
        if self.device == "cpu":
            device_id = -1
        elif self.device and self.device.startswith("cuda"):
            # "cuda:0" -> 0, "cuda" -> 0
            parts = self.device.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else 0

        self._embedder = TABBIEEmbedder(
            model_path=self.checkpoint_path,
            device_id=device_id,
            max_rows=self.max_rows,
        )

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

        with tempfile.TemporaryDirectory(prefix="tabbie_repair_") as tmpdir:
            tmp_csv = f"{tmpdir}/subset.csv"
            df.to_csv(tmp_csv, index=False)
            result = self._embedder.csv_to_embeddings(
                tmp_csv,
                aggregate="column",
                output_format="numpy",
            )

        col_embeddings = result.get("column_embeddings", {})
        mapping: Dict[int, np.ndarray] = {}
        for local_idx in sorted(col_embeddings.keys()):
            if local_idx >= len(ordered):
                break
            mapping[int(ordered[local_idx])] = col_embeddings[local_idx]
        return mapping
