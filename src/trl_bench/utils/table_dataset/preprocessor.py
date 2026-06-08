"""
Preprocessors for tabular data.

Two concrete implementations:
- SSLPreprocessor: OrdinalEncoder + median imputation + MinMaxScaler (for SCARF, DAE, etc.)
- PretrainedPreprocessor: OrdinalEncoder only, no scaling (for TabPFN, TabICL)

Both are fully pickle-serializable for saving in training_config.pkl.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

logger = logging.getLogger(__name__)


class Preprocessor:
    """Base preprocessor interface for tabular features."""

    def __init__(self):
        self._category_cols: List[str] = []
        self._continuous_cols: List[str] = []
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, X_df: pd.DataFrame) -> "Preprocessor":
        """Fit the preprocessor on training data. Must be overridden."""
        raise NotImplementedError

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters. Must be overridden."""
        raise NotImplementedError

    def fit_transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X_df).transform(X_df)

    @property
    def category_cols(self) -> List[str]:
        """Categorical column names (after fit)."""
        return self._category_cols

    @property
    def continuous_cols(self) -> List[str]:
        """Continuous column names (after fit)."""
        return self._continuous_cols

    @property
    def input_dim(self) -> int:
        """Total number of feature columns (after fit)."""
        return len(self._feature_names)

    @property
    def categorical_indices(self) -> List[int]:
        """Positional indices of categorical columns in feature order."""
        return [
            i for i, col in enumerate(self._feature_names)
            if col in self._category_cols
        ]

    def _detect_types(self, X_df: pd.DataFrame) -> None:
        """Detect categorical vs continuous columns from dtypes."""
        self._category_cols = X_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self._continuous_cols = X_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self._feature_names = list(X_df.columns)


class SSLPreprocessor(Preprocessor):
    """Preprocessor for SSL models (SCARF, DAE, SubTab, VIME).

    Categorical: OrdinalEncoder (handles unseen values with -1).
    Continuous: median imputation + MinMaxScaler.
    """

    def __init__(self):
        super().__init__()
        self._ordinal_encoder: Optional[OrdinalEncoder] = None
        self._scaler: Optional[MinMaxScaler] = None
        self._medians: Optional[pd.Series] = None

    def fit(self, X_df: pd.DataFrame) -> "SSLPreprocessor":
        self._detect_types(X_df)

        # Fit OrdinalEncoder on categorical columns
        if self._category_cols:
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            self._ordinal_encoder.fit(X_df[self._category_cols].astype(str))
            logger.info(
                "OrdinalEncoder fitted on %d categorical columns",
                len(self._category_cols),
            )

        # Fit MinMaxScaler and compute medians on continuous columns
        if self._continuous_cols:
            self._medians = X_df[self._continuous_cols].median()
            filled = X_df[self._continuous_cols].fillna(self._medians)
            self._scaler = MinMaxScaler()
            self._scaler.fit(filled)
            logger.info(
                "MinMaxScaler fitted on %d continuous columns",
                len(self._continuous_cols),
            )

        self._fitted = True
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        result = X_df.copy()

        # Encode categoricals
        if self._category_cols and self._ordinal_encoder is not None:
            encoded = self._ordinal_encoder.transform(
                result[self._category_cols].astype(str)
            )
            result[self._category_cols] = encoded

        # Impute + scale continuous
        if self._continuous_cols and self._scaler is not None:
            result[self._continuous_cols] = result[self._continuous_cols].fillna(
                self._medians
            )
            result[self._continuous_cols] = self._scaler.transform(
                result[self._continuous_cols]
            )

        return result

    @property
    def cat_cardinalities(self) -> List[int]:
        """Number of categories per categorical column (after fit)."""
        if self._ordinal_encoder is None:
            return []
        return [len(cats) for cats in self._ordinal_encoder.categories_]


class PretrainedPreprocessor(Preprocessor):
    """Preprocessor for pretrained models (TabPFN, TabICL).

    Categorical: OrdinalEncoder (needed because np.isnan fails on object dtype).
    Continuous: no scaling (TabPFN/TabICL handle normalization internally).
    """

    def __init__(self):
        super().__init__()
        self._ordinal_encoder: Optional[OrdinalEncoder] = None

    def fit(self, X_df: pd.DataFrame) -> "PretrainedPreprocessor":
        self._detect_types(X_df)

        if self._category_cols:
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            self._ordinal_encoder.fit(X_df[self._category_cols].astype(str))
            logger.info(
                "OrdinalEncoder fitted on %d categorical columns",
                len(self._category_cols),
            )

        self._fitted = True
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        result = X_df.copy()

        if self._category_cols and self._ordinal_encoder is not None:
            encoded = self._ordinal_encoder.transform(
                result[self._category_cols].astype(str)
            )
            result[self._category_cols] = encoded

        return result
