from typing import Union, Tuple, Optional, List
from numpy.typing import NDArray

from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch

from trl_bench.utils.ts3l.utils.saint_utils.saint_config import SAINTConfig


class SAINTDataset(Dataset):
    """Dataset for SAINT with CutMix augmentation for SSL phase.

    First phase: returns (x, x_cutmix) where x_cutmix has random features
    swapped from another row.
    Second phase: returns (x, y) for supervised learning.

    Args:
        X: DataFrame containing features.
        Y: Labels (optional). Integers for classification, floats for regression.
        unlabeled_data: Additional unlabeled data (optional).
        config: SAINTConfig with cutmix_probability.
        continuous_cols: List of continuous column names.
        category_cols: List of categorical column names.
        is_regression: Whether the task is regression.
        is_second_phase: Whether this is for second phase (supervised) learning.
    """

    def __init__(self, X: pd.DataFrame,
                 Y: Optional[Union[NDArray[np.int_],
                                   NDArray[np.float64]]] = None,
                 unlabeled_data: Optional[pd.DataFrame] = None,
                 config: Optional[SAINTConfig] = None,
                 continuous_cols: Optional[List] = None,
                 category_cols: Optional[List] = None,
                 is_regression: Optional[bool] = False,
                 is_second_phase: Optional[bool] = False,
                 ) -> None:

        if config is not None:
            self.config = config

        if unlabeled_data is not None:
            X = pd.concat([X, unlabeled_data])

        cat_data = torch.FloatTensor(X[category_cols].values)
        cont_data = torch.FloatTensor(X[continuous_cols].values)

        self.data = torch.concat([cat_data, cont_data], dim=1)

        self.cutmix_probability = self.config.cutmix_probability if not is_second_phase else 0.0
        self.cutmix_len = int(X.shape[1] * self.cutmix_probability)
        if not is_second_phase:
            self.cutmix_len = max(1, self.cutmix_len)

        self.n_samples, self.n_features = X.shape
        self.is_second_phase = is_second_phase
        self.is_regression = is_regression

        self.label_class = torch.FloatTensor if is_regression else torch.LongTensor

        if Y is None:
            self.label = None
        else:
            self.label = self.label_class(Y)

            if self.label_class == torch.LongTensor:
                class_counts = [sum((self.label == i))
                                for i in set(self.label.numpy())]
                num_samples = len(self.label)
                class_weights = [num_samples / class_counts[i]
                                 for i in range(len(class_counts))]
                self.weights = [class_weights[self.label[i]]
                                for i in range(int(num_samples))]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.label is None:
            if self.cutmix_len > 0 and not self.is_second_phase:
                # CutMix: randomly select another row
                other_idx = torch.randint(0, self.n_samples, (1,)).item()

                # Randomly select columns to swap
                swap_mask = torch.zeros(self.n_features, dtype=torch.bool)
                swap_idx = torch.randperm(self.n_features)[:self.cutmix_len]
                swap_mask[swap_idx] = True

                # Apply CutMix
                x_cutmix = self.data[idx].clone()
                x_cutmix[swap_mask] = self.data[other_idx][swap_mask]

                return self.data[idx], x_cutmix
            return self.data[idx], torch.tensor(-1)
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
