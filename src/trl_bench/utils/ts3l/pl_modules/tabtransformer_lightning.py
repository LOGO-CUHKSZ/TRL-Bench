from typing import Tuple
import torch
from torch import nn

from .base_module import TS3LLightining
from trl_bench.utils.ts3l.models import TabTransformerSSL
from trl_bench.utils.ts3l.utils.tabtransformer_utils import TabTransformerSSLConfig
from trl_bench.utils.ts3l import functional as F
from trl_bench.utils.ts3l.utils import BaseConfig


class TabTransformerSSLLightning(TS3LLightining):

    def __init__(self, config: TabTransformerSSLConfig) -> None:
        super(TabTransformerSSLLightning, self).__init__(config)

    def _initialize(self, config: BaseConfig) -> None:
        if not isinstance(config, TabTransformerSSLConfig):
            raise TypeError(f"Expected TabTransformerSSLConfig, got {type(config)}")

        self.mlm_weight = config.mlm_weight
        self.rtd_weight = config.rtd_weight
        self.n_cat = len(config.cat_cardinality)

        self.mlm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none'
        )
        self.rtd_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self._init_model(TabTransformerSSL, config)

    def _get_first_phase_loss(self, batch):
        x_original, x_corrupted, mlm_mask, rtd_labels = batch

        mlm_logits, rtd_logits = F.tabtransformer.first_phase_step(
            self.model, batch)

        loss = F.tabtransformer.first_phase_loss(
            x_original, mlm_mask, rtd_labels,
            mlm_logits, rtd_logits,
            self.mlm_loss_fn, self.rtd_loss_fn,
            self.mlm_weight, self.rtd_weight,
            self.n_cat,
        )
        return loss

    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        _, y = batch

        y_hat = F.tabtransformer.second_phase_step(self.model, batch)

        loss = F.tabtransformer.second_phase_loss(y, y_hat, self.task_loss_fn)

        return loss, y, y_hat

    def set_second_phase(self, freeze_encoder: bool = True) -> None:
        super().set_second_phase(freeze_encoder)
        self.model.tabtransformer_embedding.requires_grad_(not freeze_encoder)

    def predict_step(self, batch, batch_idx: int) -> torch.FloatTensor:
        y_hat = F.tabtransformer.second_phase_step(self.model, batch)
        return y_hat
