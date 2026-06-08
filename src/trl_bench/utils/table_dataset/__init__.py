"""
Table dataset abstraction for unified data loading across model pipelines.

Supports three layouts:
- Canonical: dataset.json + data.csv + splits.npz (with SHA256 integrity)
- Legacy: train.csv / val.csv / test.csv (pre-split files)
- Single CSV: one file, single "all" split
"""

from .core import TableDataset, SplitView
from .resolver import load_table_dataset, resolve_label_columns_cli
from .preprocessor import Preprocessor, SSLPreprocessor, PretrainedPreprocessor


def is_regression_label(y, label_task_types=None, label_col=None) -> bool:
    """Detect if a label series contains regression (continuous) values.

    Resolution order:
    1. **Manifest metadata** (authoritative): if *label_task_types* contains
       an entry for *label_col*, use it.  ``"regression"`` → True,
       ``"classification"`` → False.  This correctly handles int64 regression
       labels (e.g. ``price``, ``credit_amount``) that dtype alone cannot
       distinguish from classification targets.
    2. **Dtype heuristic** (fallback): float-typed columns → regression,
       everything else → classification.  Used when manifest metadata is
       unavailable (legacy layout, single CSV, or missing ``labels`` section).

    Args:
        y: Label series or array.
        label_task_types: ``dataset.label_task_types`` mapping from column
            name to ``"classification"`` or ``"regression"``.
        label_col: The label column name to look up in *label_task_types*.
    """
    if y is None:
        return False
    # 1. Manifest metadata (authoritative)
    if label_task_types and label_col:
        task_type = label_task_types.get(label_col, "")
        if task_type:
            return task_type == "regression"
    # 2. Dtype fallback
    if hasattr(y, 'dtype'):
        return y.dtype.kind == 'f'
    return False


__all__ = [
    'TableDataset',
    'SplitView',
    'load_table_dataset',
    'resolve_label_columns_cli',
    'Preprocessor',
    'SSLPreprocessor',
    'PretrainedPreprocessor',
    'is_regression_label',
]
