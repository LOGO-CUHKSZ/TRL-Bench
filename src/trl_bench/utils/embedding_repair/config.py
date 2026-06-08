"""Configuration helpers for embedding repair."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def get_project_root() -> Path:
    # config.py lives at src/trl_bench/utils/embedding_repair/config.py
    # -> parents[4] is the project root.
    return Path(__file__).resolve().parents[4]


def load_models_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root or get_project_root()
    path = root / "slurm" / "config" / "models.yaml"
    return yaml.safe_load(path.read_text())


def load_datasets_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root or get_project_root()
    path = root / "slurm" / "config" / "datasets.yaml"
    return yaml.safe_load(path.read_text())


def resolve_tables_dir(dataset: str, datasets_cfg: Dict[str, Any], project_root: Path) -> Path:
    ds_cfg = datasets_cfg["datasets"][dataset]
    tables_dir = ds_cfg.get("tables_dir")
    if tables_dir is None:
        source = ds_cfg.get("tables_source")
        if source is None:
            raise ValueError(f"Dataset {dataset} has no tables_dir or tables_source")
        ds_cfg = datasets_cfg["datasets"][source]
        tables_dir = ds_cfg.get("tables_dir")
        dataset = source
    return project_root / "datasets" / dataset / tables_dir


def default_embeddings_path(model: str, dataset: str, project_root: Path) -> Path:
    # TRL-Bench layout: embeddings/column/{model}/{dataset}.pkl (no `assets/`
    # prefix).
    return project_root / "embeddings" / "column" / model / f"{dataset}.pkl"


def default_checkpoint_path(model: str, models_cfg: Dict[str, Any], project_root: Path) -> str:
    ckpt = models_cfg["models"][model]["checkpoint"]
    # Treat any value containing a forward-slash without a leading path char
    # OR no slash at all as a Hugging Face model ID (e.g. `bert-base-uncased`,
    # `google/tapas-base`, `thenlper/gte-base`). Otherwise resolve against
    # project_root.
    if "/" not in ckpt or ckpt.split("/", 1)[0] in {"google", "facebook", "thenlper", "microsoft", "sentence-transformers"}:
        if not ckpt.startswith(("checkpoints/", "/")) and ".pt" not in ckpt and ".ckpt" not in ckpt and ".bin" not in ckpt:
            return ckpt
    return str(project_root / ckpt)


def default_model_args(model: str, models_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return models_cfg["models"][model].get("defaults", {})
