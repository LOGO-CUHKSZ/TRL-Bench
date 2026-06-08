"""Base adapter interface for embedding repair."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

import numpy as np

from trl_bench.utils.aggregation import aggregate_embeddings


class BaseAdapter(ABC):
    """Adapter interface for model-specific embedding repair."""

    def __init__(self, device: Optional[str] = None):
        self.device = device

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def recommended_chunk_size(self) -> Optional[int]:
        """Optional hint for repair chunk sizing."""
        return None

    def get_table_path(self, record: Dict[str, Any]) -> str:
        return record.get("table") or record.get("table_path")

    def get_column_embeddings(self, record: Dict[str, Any]) -> Dict[int, np.ndarray]:
        col_embs = record.get("column_embeddings")
        if col_embs is None:
            col_embs = record.get("column_embedding", {})
        if isinstance(col_embs, list):
            return {i: np.array(v) for i, v in enumerate(col_embs)}
        if isinstance(col_embs, dict):
            normalized: Dict[int, np.ndarray] = {}
            for key, value in col_embs.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                normalized[idx] = value
            return normalized
        return {}

    def detect_missing(
        self,
        record: Dict[str, Any],
        header: List[str],
    ) -> List[int]:
        """Default missing detection: any index not present in embeddings."""
        col_embs = self.get_column_embeddings(record)
        present = set(col_embs.keys())
        return [i for i in range(len(header)) if i not in present]

    def is_missing_vector(self, vec: np.ndarray) -> bool:
        """Override for model-specific missing/zero vector detection."""
        return False

    def recompute_table_embedding(
        self,
        record: Dict[str, Any],
        column_embeddings: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Recompute table_embedding with column_mean."""
        table_embedding = record.get("table_embedding")
        cls_embedding = None
        base_table_embedding = None
        if isinstance(table_embedding, dict):
            cls_embedding = table_embedding.get("cls_embedding")
            base_table_embedding = table_embedding.get("table_embedding")
        else:
            cls_embedding = record.get("cls_embedding")
            base_table_embedding = table_embedding

        updated = {
            "cls_embedding": cls_embedding,
            "table_embedding": base_table_embedding,
            "column_mean": aggregate_embeddings(column_embeddings, "mean"),
            "token_mean": table_embedding.get("token_mean") if isinstance(table_embedding, dict) else None,
        }
        return updated

    def update_record(
        self,
        record: Dict[str, Any],
        header: List[str],
        column_embeddings: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        record["column_embeddings"] = column_embeddings
        record["column_names"] = [str(c) for c in header]
        record["table_embedding"] = self.recompute_table_embedding(record, column_embeddings)
        return record

    @abstractmethod
    def embed_columns(
        self,
        table_path: str,
        col_indices: List[int],
        max_rows: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Embed a subset of columns. Returns mapping from original column index to vector."""
        raise NotImplementedError
