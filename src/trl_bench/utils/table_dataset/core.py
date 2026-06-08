"""
Core TableDataset and SplitView classes.

TableDataset holds a full dataframe with split information, providing
uniform access regardless of the underlying layout (canonical, legacy, single CSV).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitView:
    """A view into a single split of a TableDataset.

    Attributes:
        X: Feature columns as a DataFrame.
        y: Label column(s) as a Series/DataFrame, or None if no labels.
        row_indices: Integer indices into the canonical data.csv.
            None for legacy layout (no stable global indexing).
    """
    X: pd.DataFrame
    y: Optional[Union[pd.Series, pd.DataFrame]] = None
    row_indices: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        parts = [f"SplitView(n={len(self)}, features={self.X.shape[1]}"]
        if self.y is not None:
            parts.append(f"has_labels=True")
        if self.row_indices is not None:
            parts.append(f"has_row_indices=True")
        return ", ".join(parts) + ")"


class TableDataset:
    """Unified dataset container for tabular data with split information.

    Holds the full dataframe internally and provides split-level access
    via ``get_split(name)`` and whole-dataset access via ``get_full()``.

    Parameters:
        full_df: The complete dataframe (all rows, all columns including labels).
        label_columns: Column name(s) to treat as labels. Empty list means no labels.
        split_indices: Mapping from split name to row indices (into full_df).
            For canonical layout, these come from splits.npz.
            For legacy layout, these are synthetic ranges from concatenation.
        layout: One of "canonical", "legacy", "single".
        source_path: Path to the dataset root directory or CSV file.
        fingerprint: SHA256 hex digest of data.csv (canonical only).
        label_task_types: Mapping from label column name to task type
            ("classification" or "regression"), sourced from canonical manifest.
            Empty dict for legacy/single layouts.
    """

    def __init__(
        self,
        full_df: pd.DataFrame,
        label_columns: Optional[List[str]] = None,
        split_indices: Optional[Dict[str, Optional[np.ndarray]]] = None,
        layout: str = "single",
        source_path: str = "",
        fingerprint: str = "",
        label_task_types: Optional[Dict[str, str]] = None,
    ):
        self._full_df = full_df.reset_index(drop=True)
        self._label_columns = label_columns or []
        self._split_indices = split_indices or {"all": np.arange(len(full_df))}
        self._layout = layout
        self.source_path = source_path
        self.fingerprint = fingerprint
        self._label_task_types = label_task_types or {}

        # Validate label columns exist (warn and skip missing ones)
        valid_labels = []
        for col in self._label_columns:
            if col in self._full_df.columns:
                valid_labels.append(col)
            else:
                logger.warning(
                    "Label column '%s' not found in dataset (available: %s), ignoring",
                    col, list(self._full_df.columns)[:10],
                )
        self._label_columns = valid_labels

    @property
    def split_names(self) -> List[str]:
        """Names of available splits."""
        return list(self._split_indices.keys())

    @property
    def label_columns(self) -> List[str]:
        """Label column name(s)."""
        return self._label_columns

    @property
    def label_task_types(self) -> Dict[str, str]:
        """Per-label task type from manifest: ``{"col": "classification"|"regression"}``."""
        return self._label_task_types

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names (all columns except labels)."""
        return [c for c in self._full_df.columns if c not in self._label_columns]

    @property
    def layout(self) -> str:
        """Dataset layout: 'canonical', 'legacy', or 'single'."""
        return self._layout

    def get_split(self, name: str) -> SplitView:
        """Get a view of a single split.

        Args:
            name: Split name (e.g. "train", "test", "val", "all").

        Returns:
            SplitView with features, optional labels, and optional row indices.

        Raises:
            KeyError: If split name not found.
        """
        if name not in self._split_indices:
            raise KeyError(
                f"Split '{name}' not found. Available splits: {self.split_names}"
            )

        indices = self._split_indices[name]
        if indices is not None:
            split_df = self._full_df.iloc[indices].reset_index(drop=True)
        else:
            split_df = self._full_df.copy()

        X = split_df[self.feature_columns]

        y = None
        if self._label_columns:
            if len(self._label_columns) == 1:
                y = split_df[self._label_columns[0]]
            else:
                y = split_df[self._label_columns]

        # Row indices: available for canonical and single layouts
        if self._layout in ("canonical", "single") and indices is not None:
            row_indices = indices.copy()
        else:
            # Legacy layout: synthetic indices, not stable
            row_indices = None

        return SplitView(X=X, y=y, row_indices=row_indices)

    def get_full(self) -> SplitView:
        """Get a view over all rows.

        For canonical/single layouts, row_indices is np.arange(n).
        For legacy layout, row_indices is None (synthetic ordering).
        """
        X = self._full_df[self.feature_columns]

        y = None
        if self._label_columns:
            if len(self._label_columns) == 1:
                y = self._full_df[self._label_columns[0]]
            else:
                y = self._full_df[self._label_columns]

        if self._layout in ("canonical", "single"):
            row_indices = np.arange(len(self._full_df))
        else:
            row_indices = None
            logger.warning(
                "get_full() on legacy dataset: row_indices is None because "
                "the concatenation order is synthetic."
            )

        return SplitView(X=X, y=y, row_indices=row_indices)

    def apply_train_test_split(
        self,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        stratify_on_label: bool = True,
    ) -> None:
        """Split an 'all'-only dataset into train/test in-place.

        This is meant for ``--input`` single-CSV mode where the old scripts
        did ``train_test_split`` internally.  If the dataset already has
        train/test splits, this is a no-op.

        Args:
            train_ratio: Fraction of data for training.
            random_seed: Random seed for reproducibility.
            stratify_on_label: Use stratified split if a label column exists.
        """
        if set(self._split_indices.keys()) != {"all"}:
            return  # already has named splits

        from sklearn.model_selection import train_test_split

        n = len(self._full_df)
        indices = np.arange(n)

        stratify = None
        if stratify_on_label and self._label_columns:
            stratify = self._full_df[self._label_columns[0]]

        try:
            train_idx, test_idx = train_test_split(
                indices,
                train_size=train_ratio,
                random_state=random_seed,
                stratify=stratify,
            )
        except ValueError:
            # Stratification fails for continuous (regression) targets or
            # classes with too few members; fall back to random split.
            if stratify is not None:
                logger.warning(
                    "Stratified split failed for label '%s'; "
                    "falling back to random split",
                    self._label_columns[0],
                )
            train_idx, test_idx = train_test_split(
                indices,
                train_size=train_ratio,
                random_state=random_seed,
            )

        self._split_indices = {
            "train": np.sort(train_idx),
            "test": np.sort(test_idx),
        }

    def __len__(self) -> int:
        return len(self._full_df)

    def __repr__(self) -> str:
        return (
            f"TableDataset(layout={self._layout!r}, rows={len(self)}, "
            f"features={len(self.feature_columns)}, "
            f"labels={self._label_columns}, "
            f"splits={self.split_names})"
        )
