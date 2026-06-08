"""Starmie adapter for embedding repair."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import BaseAdapter
from ..io_utils import read_csv_subset


class StarmieAdapter(BaseAdapter):
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._dataset = None

    def name(self) -> str:
        return "starmie"

    def _load_model(self):
        if self._model is not None:
            return
        from trl_bench.models.starmie.sdd.pretrain import load_checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model, dataset = load_checkpoint(ckpt)
        if self.device is not None:
            model = model.to(self.device)
        self._model = model
        self._dataset = dataset

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

        from trl_bench.models.starmie.sdd.pretrain import inference_on_tables

        vectors = inference_on_tables(
            [df],
            self._model,
            self._dataset,
            batch_size=1,
            total=1,
        )
        if not vectors:
            return {}

        col_vectors = vectors[0]
        mapping: Dict[int, np.ndarray] = {}
        for idx, vec in enumerate(col_vectors):
            if idx >= len(ordered):
                break
            mapping[int(ordered[idx])] = vec
        return mapping
