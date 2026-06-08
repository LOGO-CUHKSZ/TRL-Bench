from dataclasses import dataclass, field
from trl_bench.utils.ts3l.utils import BaseConfig

from typing import List, Optional


@dataclass
class TabTransformerSSLConfig(BaseConfig):
    """Configuration for TabTransformer-SSL (MLM + RTD pretraining).

    Inherits all BaseConfig fields (task, embedding_config, backbone_config,
    output_dim, loss_fn, etc.).

    New Attributes:
        emb_dim: Embedding dimension for categorical features.
        n_transformer_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (auto-corrected if emb_dim % n_heads != 0).
        transformer_ffn_dim: FFN dimension in transformer (defaults to 4 * emb_dim).
        transformer_dropout: Dropout rate in transformer layers.
        cat_cardinality: List of cardinalities for each categorical column.
        num_continuous: Number of continuous features.
        mlm_probability: Fraction of categorical columns to mask for MLM.
        rtd_probability: Fraction of remaining columns to corrupt for RTD.
        mlm_weight: Weight for MLM loss.
        rtd_weight: Weight for RTD loss.
        dropout_rate: Dropout rate for downstream head.
    """

    emb_dim: int = field(default=32)

    n_transformer_layers: int = field(default=6)

    n_heads: int = field(default=8)

    transformer_ffn_dim: Optional[int] = field(default=None)

    transformer_dropout: float = field(default=0.0)

    cat_cardinality: List[int] = field(default_factory=lambda: [])

    num_continuous: Optional[int] = field(default=None)

    mlm_probability: float = field(default=0.15)

    rtd_probability: float = field(default=0.15)

    mlm_weight: float = field(default=1.0)

    rtd_weight: float = field(default=1.0)

    dropout_rate: float = field(default=0.04)

    def __post_init__(self):
        super().__post_init__()

        if len(self.cat_cardinality) == 0:
            raise ValueError(
                "TabTransformer requires at least 1 categorical column "
                "(cat_cardinality must be non-empty)."
            )

        if self.num_continuous is None:
            self.num_continuous = 0

        if self.transformer_ffn_dim is None:
            self.transformer_ffn_dim = 4 * self.emb_dim

        # Auto-correct n_heads to nearest divisor of emb_dim
        if self.emb_dim % self.n_heads != 0:
            divisors = [n for n in range(1, self.emb_dim + 1)
                        if self.emb_dim % n == 0]
            closest = min(divisors, key=lambda x: abs(x - self.n_heads))
            if closest != self.n_heads:
                print(f"Config: adjusting n_heads from {self.n_heads} to "
                      f"{closest} (closest valid divisor of {self.emb_dim})")
            self.n_heads = closest

        if not (0 <= self.mlm_probability <= 1):
            raise ValueError(
                f"mlm_probability must be in [0, 1], got {self.mlm_probability}")

        if not (0 <= self.rtd_probability <= 1):
            raise ValueError(
                f"rtd_probability must be in [0, 1], got {self.rtd_probability}")
