"""Adapter registry for embedding repair."""

from __future__ import annotations

import importlib
from typing import Dict, Type

from .adapters.base import BaseAdapter


ADAPTERS: Dict[str, str] = {
    "bert": "trl_bench.utils.embedding_repair.adapters.bert:BertAdapter",
    "doduo": "trl_bench.utils.embedding_repair.adapters.doduo:DoduoAdapter",
    "gte": "trl_bench.utils.embedding_repair.adapters.gte:GteAdapter",
    "starmie": "trl_bench.utils.embedding_repair.adapters.starmie:StarmieAdapter",
    "tabert": "trl_bench.utils.embedding_repair.adapters.tabert:TabertAdapter",
    "tabsketchfm": "trl_bench.utils.embedding_repair.adapters.tabsketchfm:TabSketchFMAdapter",
    "tabbie": "trl_bench.utils.embedding_repair.adapters.tabbie:TabbieAdapter",
    "tapas": "trl_bench.utils.embedding_repair.adapters.tapas:TapasAdapter",
}


def _load_adapter(path: str) -> Type[BaseAdapter]:
    module_path, class_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_adapter(model: str, **kwargs) -> BaseAdapter:
    if model not in ADAPTERS:
        raise ValueError(f"Unsupported model for repair: {model}")
    cls = _load_adapter(ADAPTERS[model])
    return cls(**kwargs)
