from typing import Tuple
import torch
from torch import nn

from .base_module import TS3LLightining
from trl_bench.utils.ts3l.models import SAINT
from trl_bench.utils.ts3l.models.scarf import NTXentLoss
from trl_bench.utils.ts3l.utils.saint_utils import SAINTConfig
from trl_bench.utils.ts3l import functional as F
from trl_bench.utils.ts3l.utils import BaseConfig


class SAINTLightning(TS3LLightining):

    def __init__(self, config: SAINTConfig) -> None:
        """Initialize the PyTorch Lightning module for SAINT.

        Args:
            config: The configuration for SAINTLightning.
        """
        super(SAINTLightning, self).__init__(config)

    def _initialize(self, config: BaseConfig) -> None:
        """Initialize SAINT-specific components.

        Args:
            config: SAINTConfig instance.
        """
        if not isinstance(config, SAINTConfig):
            raise TypeError(f"Expected SAINTConfig, got {type(config)}")

        self.lambda_denoise = config.lambda_denoise
        self.num_categoricals = len(config.cat_cardinality)
        self.num_continuous = config.num_continuous

        # Contrastive loss (reuse from SCARF)
        self.contrastive_loss = NTXentLoss(config.tau)

        # Denoising losses (follow DAE pattern)
        self.categorical_feature_loss = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none')
        self.continuous_feature_loss = nn.MSELoss()

        self._init_model(SAINT, config)

    def _get_first_phase_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate the first phase loss (contrastive + denoising).

        Args:
            batch: (x, x_cutmix) tensors.

        Returns:
            Combined loss scalar.
        """
        x, x_cutmix = batch

        proj_orig, proj_mixed, cat_preds, cont_preds = F.saint.first_phase_step(
            self.model, batch)

        contrastive = F.saint.contrastive_loss(
            proj_orig, proj_mixed, self.contrastive_loss)

        denoising = F.saint.denoising_loss(
            x[:, :self.num_categoricals],
            x[:, self.num_categoricals:],
            cat_preds, cont_preds,
            self.categorical_feature_loss,
            self.continuous_feature_loss)

        return contrastive + self.lambda_denoise * denoising

    def _get_second_phase_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the second phase loss (supervised).

        Args:
            batch: (x, y) tensors.

        Returns:
            Tuple of (loss, ground truth y, predicted y_hat).
        """
        _, y = batch

        y_hat = F.saint.second_phase_step(self.model, batch)

        loss = F.saint.second_phase_loss(y, y_hat, self.task_loss_fn)

        return loss, y, y_hat

    def set_second_phase(self, freeze_encoder: bool = True) -> None:
        """Set the module to fine-tuning.

        Args:
            freeze_encoder: If True, freeze the encoder during fine-tuning.
        """
        return super().set_second_phase(freeze_encoder)

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Predict step for SAINT.

        Args:
            batch: Input batch.
            batch_idx: For compatibility, do not use.

        Returns:
            Predicted output (logits).
        """
        y_hat = F.saint.second_phase_step(self.model, batch)
        return y_hat
