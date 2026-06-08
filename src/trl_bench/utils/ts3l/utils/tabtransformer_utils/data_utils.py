from typing import Tuple, Union, Optional, List
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from trl_bench.utils.ts3l.utils.tabtransformer_utils import TabTransformerSSLConfig


class TabTransformerSSLDataset(Dataset):
    """Dataset for TabTransformer-SSL.

    Stores categorical data (as float-encoded ints) and continuous data
    separately, concatenating them in __getitem__. Categories come first,
    matching the DAE convention.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        Y: Optional[Union[NDArray[np.int_], NDArray[np.float64]]] = None,
        unlabeled_data: Optional[pd.DataFrame] = None,
        continuous_cols: Optional[List] = None,
        category_cols: Optional[List] = None,
        is_regression: Optional[bool] = False,
    ) -> None:
        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])

        self.len = len(X)

        self.cont_data = torch.FloatTensor(
            X[continuous_cols].values if continuous_cols else np.empty((len(X), 0))
        )
        self.cat_data = torch.FloatTensor(
            X[category_cols].values if category_cols else np.empty((len(X), 0))
        )

        self.continuous_cols = continuous_cols or []
        self.category_cols = category_cols or []

        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor

        if Y is None:
            self.label = None
        else:
            self.label = self.label_class(Y)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cat_samples = self.cat_data[idx]
        cont_samples = self.cont_data[idx]
        x = torch.concat((cat_samples, cont_samples))

        if self.label is None:
            return x

        return x, self.label[idx]

    def __len__(self) -> int:
        return self.len


class TabTransformerSSLCollateFN(object):
    """Collate function for TabTransformer-SSL pretraining.

    Produces (x_original, x_corrupted, mlm_mask, rtd_labels) per batch.
    MLM and RTD are mutually exclusive per column per sample.
    """

    def __init__(self, config: TabTransformerSSLConfig) -> None:
        self.n_cat = len(config.cat_cardinality)
        self.cat_cardinality = config.cat_cardinality
        self.mlm_probability = config.mlm_probability
        self.rtd_probability = config.rtd_probability

    def __call__(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_original = torch.stack(batch)  # (B, n_cat + n_cont)
        B = x_original.size(0)
        n_cat = self.n_cat

        x_corrupted = x_original.clone()
        mlm_mask = torch.zeros(B, n_cat, dtype=torch.bool)
        rtd_labels = torch.zeros(B, n_cat, dtype=torch.float)

        for i in range(B):
            # --- MLM: select columns to mask ---
            if self.mlm_probability > 0 and n_cat > 0:
                n_mlm = max(1, int(n_cat * self.mlm_probability))
            else:
                n_mlm = 0

            all_cols = torch.randperm(n_cat)
            mlm_cols = all_cols[:n_mlm]
            remaining_cols = all_cols[n_mlm:]

            # Set MLM mask
            if n_mlm > 0:
                mlm_mask[i, mlm_cols] = True

            # --- RTD: select from remaining columns ---
            n_remaining = len(remaining_cols)
            if self.rtd_probability > 0 and n_remaining > 0:
                n_rtd = max(1, int(n_remaining * self.rtd_probability))
            else:
                n_rtd = 0

            if n_rtd > 0:
                rtd_perm = torch.randperm(n_remaining)[:n_rtd]
                rtd_cols = remaining_cols[rtd_perm]

                for col_idx in rtd_cols:
                    col = col_idx.item()
                    card = self.cat_cardinality[col]
                    original_val = int(x_original[i, col].item())
                    replacement = torch.randint(0, card, (1,)).item()
                    x_corrupted[i, col] = replacement
                    # Only mark as replaced if the value actually changed
                    if replacement != original_val:
                        rtd_labels[i, col] = 1.0

        return x_original, x_corrupted, mlm_mask, rtd_labels
