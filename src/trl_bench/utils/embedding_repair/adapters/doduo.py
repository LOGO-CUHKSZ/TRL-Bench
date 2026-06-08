"""Doduo adapter for embedding repair."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch

from .base import BaseAdapter
from ..io_utils import read_csv_subset


class DoduoAdapter(BaseAdapter):
    def __init__(
        self,
        checkpoint_path: str,
        model_variant: str = "wikitable",
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self.model_variant = model_variant
        self._model = None

    def name(self) -> str:
        return "doduo"

    def recommended_chunk_size(self) -> int:
        # Doduo tablewise encoding uses max_length=32 per column (34 tokens incl. CLS/SEP).
        # Keep chunks <= 15 columns to stay under the 512-token limit.
        return 15

    def _load_model(self):
        if self._model is not None:
            return
        import sys
        from pathlib import Path

        doduo_root = Path(__file__).resolve().parents[3] / "models" / "doduo"
        if str(doduo_root) not in sys.path:
            sys.path.insert(0, str(doduo_root))
        from doduo.doduo import Doduo
        args = argparse.Namespace(model=self.model_variant)
        model = Doduo(
            args,
            coltype_model_path=self.checkpoint_path,
            load_colrel=False,
            load_vocab=False,
        )
        if self.device is not None:
            dev = torch.device(self.device)
            model.device = dev
            if hasattr(model, "coltype_model") and model.coltype_model is not None:
                model.coltype_model = model.coltype_model.to(dev)
            if hasattr(model, "colrel_model") and model.colrel_model is not None:
                model.colrel_model = model.colrel_model.to(dev)
        self._model = model

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
        embeddings = self._model.get_column_embeddings(df)
        mapping: Dict[int, np.ndarray] = {}
        for idx, emb in enumerate(embeddings):
            if idx >= len(ordered):
                break
            mapping[int(ordered[idx])] = emb
        return mapping
