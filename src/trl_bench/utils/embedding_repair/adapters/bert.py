"""BERT adapter for embedding repair."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..io_utils import read_csv_subset


class BertAdapter(BaseAdapter):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self._embedder = None

    def name(self) -> str:
        return "bert"

    def _load_model(self):
        if self._embedder is not None:
            return
        from trl_bench.models.bert.generate_column_embeddings import BERTEmbedder

        self._embedder = BERTEmbedder(
            model_name=self.model_name,
            device=self.device,
        )

    def embed_columns(
        self,
        table_path: str,
        col_indices: List[int],
        max_rows: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        self._load_model()
        ordered = sorted(set(int(i) for i in col_indices))
        df = read_csv_subset(table_path, ordered, max_rows=max_rows or 100)
        if df.empty or len(df.columns) == 0:
            return {}

        from trl_bench.models.bert.generate_column_embeddings import serialize_column

        mapping: Dict[int, np.ndarray] = {}
        for idx, orig_idx in enumerate(ordered):
            if idx >= len(df.columns):
                break
            col_name = str(df.columns[idx])
            col_text = serialize_column(col_name, df.iloc[:, idx])
            mapping[orig_idx] = self._embedder._encode_text(col_text)
        return mapping
