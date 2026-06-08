from typing import OrderedDict, Tuple, List, Optional

import torch
import torch.nn as nn

from trl_bench.utils.ts3l.models.common import TS3LModule
from trl_bench.utils.ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig


class TabTransformerEmbedding(nn.Module):
    """Embedding module for TabTransformer-SSL.

    Applies learned embeddings + column positional embeddings + transformer
    self-attention to categorical features, while continuous features are
    passed through LayerNorm only. Outputs a flat concatenation of the two.
    """

    def __init__(
        self,
        emb_dim: int,
        num_continuous: int,
        cat_cardinality: List[int],
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_cat = len(cat_cardinality)
        self.num_continuous = num_continuous
        self.emb_dim = emb_dim
        self.cat_cardinality = cat_cardinality

        # Offset indexing for shared embedding table
        total_categories = sum(cat_cardinality)
        self.cat_embedding = nn.Embedding(total_categories, emb_dim)
        offsets = torch.tensor([0] + cat_cardinality[:-1]).cumsum(0)
        self.register_buffer("offsets", offsets)

        # Column positional embeddings (not named 'weight'/'bias', so
        # initialize_weights won't touch them — init explicitly)
        self.column_embedding = nn.Parameter(torch.empty(self.n_cat, emb_dim))
        nn.init.xavier_uniform_(self.column_embedding)

        # Learnable [MASK] token for MLM
        self.mask_token = nn.Parameter(torch.empty(1, emb_dim))
        nn.init.xavier_uniform_(self.mask_token)

        # Auto-correct n_heads if emb_dim % n_heads != 0
        if emb_dim % n_heads != 0:
            divisors = [n for n in range(1, emb_dim + 1) if emb_dim % n == 0]
            closest = min(divisors, key=lambda x: abs(x - n_heads))
            if closest != n_heads:
                print(f"Adjusting n_heads from {n_heads} to {closest} "
                      f"(closest valid divisor of {emb_dim})")
            n_heads = closest

        if ffn_dim is None:
            ffn_dim = 4 * emb_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm for continuous features
        if num_continuous > 0:
            self.cont_norm = nn.LayerNorm(num_continuous)
        else:
            self.cont_norm = None

        self.output_dim = self.n_cat * emb_dim + num_continuous

    def forward(self, x: torch.Tensor, mlm_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, n_cat + n_cont) — cats as float-encoded ints, conts as floats.
            mlm_mask: (B, n_cat) bool tensor. True = masked for MLM.

        Returns:
            (B, n_cat * emb_dim + n_cont)
        """
        B = x.size(0)
        x_cat = x[:, :self.n_cat].long()    # (B, n_cat)
        x_cont = x[:, self.n_cat:]           # (B, n_cont)

        # Clamp unknown categories (-1 from OrdinalEncoder) to 0 for safe
        # embedding lookup; masked positions are replaced by mask_token anyway
        x_cat = x_cat.clamp(min=0)

        # Categorical embeddings with offset indexing
        cat_emb = self.cat_embedding(x_cat + self.offsets)  # (B, n_cat, emb_dim)
        cat_emb = cat_emb + self.column_embedding           # broadcast (n_cat, emb_dim)

        # Apply MLM mask: replace masked positions with mask_token
        if mlm_mask is not None:
            mask_expanded = mlm_mask.unsqueeze(-1)  # (B, n_cat, 1)
            cat_emb = torch.where(mask_expanded, self.mask_token, cat_emb)

        # Transformer on categorical embeddings
        cat_emb = self.transformer(cat_emb)  # (B, n_cat, emb_dim)
        cat_flat = cat_emb.reshape(B, -1)    # (B, n_cat * emb_dim)

        # Continuous features
        if self.num_continuous > 0 and self.cont_norm is not None:
            x_cont = self.cont_norm(x_cont)  # (B, n_cont)
            return torch.cat([cat_flat, x_cont], dim=1)

        return cat_flat


class TabTransformerSSL(TS3LModule):
    """TabTransformer with MLM + RTD self-supervised pretraining.

    Uses IdentityEmbedding (unused pass-through) as the TS3L embedding module,
    and stores the actual TabTransformerEmbedding separately to handle the
    extra mlm_mask argument during phase 1.
    """

    def __init__(
        self,
        embedding_config: BaseEmbeddingConfig,
        backbone_config: BaseBackboneConfig,
        num_continuous: int,
        cat_cardinality: List[int],
        emb_dim: int = 32,
        n_transformer_layers: int = 6,
        n_heads: int = 8,
        transformer_ffn_dim: Optional[int] = None,
        transformer_dropout: float = 0.0,
        output_dim: int = 2,
        dropout_rate: float = 0.04,
        **kwargs,
    ):
        super(TabTransformerSSL, self).__init__(embedding_config, backbone_config)

        self.tabtransformer_embedding = TabTransformerEmbedding(
            emb_dim=emb_dim,
            num_continuous=num_continuous,
            cat_cardinality=cat_cardinality,
            n_layers=n_transformer_layers,
            n_heads=n_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
        )

        backbone_out = self.backbone_module.output_dim
        n_cat = len(cat_cardinality)

        # MLM heads: one per categorical column, predicts the original category
        self.mlm_heads = nn.ModuleList([
            nn.Linear(backbone_out, card) for card in cat_cardinality
        ])

        # RTD head: binary prediction per categorical column
        self.rtd_head = nn.Linear(backbone_out, n_cat)

        # Downstream classification/regression head (phase 2)
        self.head = nn.Sequential(OrderedDict([
            ("head_activation", nn.ReLU(inplace=True)),
            ("head_batchnorm", nn.BatchNorm1d(backbone_out)),
            ("head_dropout", nn.Dropout(dropout_rate)),
            ("head_linear", nn.Linear(backbone_out, output_dim)),
        ]))

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module

    def set_second_phase(self, freeze_encoder: bool = True):
        super().set_second_phase(freeze_encoder)
        self.tabtransformer_embedding.requires_grad_(not freeze_encoder)

    def _first_phase_step(
        self, x_corrupted: torch.Tensor, mlm_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x_emb = self.tabtransformer_embedding(x_corrupted, mlm_mask=mlm_mask)
        h = self.backbone_module(x_emb)
        mlm_logits = [head(h) for head in self.mlm_heads]
        rtd_logits = self.rtd_head(h)
        return mlm_logits, rtd_logits

    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.tabtransformer_embedding(x)
        h = self.backbone_module(x_emb)
        return self.head(h)
