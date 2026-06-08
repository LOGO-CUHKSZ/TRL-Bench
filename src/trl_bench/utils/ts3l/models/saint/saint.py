from typing import OrderedDict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from trl_bench.utils.ts3l.models.common import TS3LModule
from trl_bench.utils.ts3l.models.common import MLP
from trl_bench.utils.ts3l.models.common.reconstruction_head import ReconstructionHead
from trl_bench.utils.ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig
from .encoder import SAINTEncoder


class SAINT(TS3LModule):
    def __init__(
        self,
        embedding_config: BaseEmbeddingConfig,
        backbone_config: BaseBackboneConfig,
        num_continuous: int,
        cat_cardinality: List[int],
        pretraining_head_dim: int = 256,
        output_dim: int = 2,
        head_depth: int = 2,
        dropout_rate: float = 0.0,
        mixup_alpha: float = 0.2,
        **kwargs
    ) -> None:
        """SAINT: Self-Attention and INtersample attention Transformer.

        Args:
            embedding_config: Configuration for the FeatureTokenizer embedding.
            backbone_config: Configuration for the SAINTEncoder backbone.
            num_continuous: Number of continuous features.
            cat_cardinality: Cardinality of each categorical feature.
            pretraining_head_dim: Dimension of the contrastive projection head.
            output_dim: Output dimension for downstream task head.
            head_depth: Number of layers in the projection head.
            dropout_rate: Dropout rate for heads.
            mixup_alpha: Alpha parameter for Beta distribution in Mixup.
        """
        super(SAINT, self).__init__(embedding_config, backbone_config)

        d = self.backbone_module.output_dim

        # Contrastive projection head
        self.pretraining_head = MLP(
            input_dim=d, hidden_dims=pretraining_head_dim,
            n_hiddens=head_depth, dropout_rate=dropout_rate)

        # Feature reconstruction head (same as DAE)
        self.feature_predictor = ReconstructionHead(
            d, num_continuous, cat_cardinality)

        # Downstream classification/regression head
        self.head = nn.Sequential(
            OrderedDict([
                ("head_activation", nn.ReLU(inplace=True)),
                ("head_batchnorm", nn.BatchNorm1d(d)),
                ("head_dropout", nn.Dropout(dropout_rate)),
                ("head_linear", nn.Linear(d, output_dim)),
            ])
        )

        self.mixup_alpha = mixup_alpha

    def _set_backbone_module(self, backbone_config):
        """Override to use SAINTEncoder instead of default TS3LBackboneModule."""
        self.backbone_module = SAINTEncoder(**backbone_config.__dict__)

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module

    def _first_phase_step(self, x: torch.Tensor, x_cutmix: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor,
                                     List[torch.Tensor], torch.Tensor]:
        # Embed both views via FeatureTokenizer
        z_orig = self.embedding_module(x)          # (B, N+1, d)
        z_cutmix = self.embedding_module(x_cutmix) # (B, N+1, d)

        # Mixup in embedding space (augmented view only)
        lam = Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
        perm = torch.randperm(len(x), device=x.device)
        z_mixed = lam * z_cutmix + (1 - lam) * z_cutmix[perm]

        # Encode both views
        emb_orig = self.backbone_module(z_orig)    # (B, d) CLS tokens
        emb_mixed = self.backbone_module(z_mixed)  # (B, d) CLS tokens

        # Contrastive projections
        proj_orig = F.normalize(self.pretraining_head(emb_orig), p=2)
        proj_mixed = F.normalize(self.pretraining_head(emb_mixed), p=2)

        # Feature reconstruction from augmented view
        cat_preds, cont_preds = self.feature_predictor(emb_mixed)

        return proj_orig, proj_mixed, cat_preds, cont_preds

    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding_module(x)
        emb = self.backbone_module(z)
        return self.head(emb)
