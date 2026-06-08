from dataclasses import dataclass, field
from trl_bench.utils.ts3l.utils import BaseConfig

from typing import List


@dataclass
class SAINTConfig(BaseConfig):
    """Configuration for SAINT (Self-Attention and INtersample attention Transformer).

    Inherits all BaseConfig attributes (task, embedding_config, backbone_config,
    output_dim, loss_fn, etc.).

    New Attributes:
        num_continuous: Number of continuous features.
        cat_cardinality: Cardinality of each categorical feature.
        saint_variant: SAINT variant ('saint', 'saint_s', 'saint_i').
        pretraining_head_dim: Dimension of contrastive projection head.
        head_depth: Number of layers in projection head.
        dropout_rate: Dropout rate for heads.
        cutmix_probability: Proportion of features to swap in CutMix.
        mixup_alpha: Alpha for Beta distribution in Mixup augmentation.
        tau: Temperature for NTXent contrastive loss.
        lambda_denoise: Weight for denoising loss component.
    """

    num_continuous: int = field(default=0)
    cat_cardinality: List[int] = field(default_factory=list)

    saint_variant: str = field(default="saint")
    pretraining_head_dim: int = field(default=256)
    head_depth: int = field(default=2)
    dropout_rate: float = field(default=0.0)

    cutmix_probability: float = field(default=0.3)
    mixup_alpha: float = field(default=0.2)

    tau: float = field(default=0.7)
    lambda_denoise: float = field(default=10.0)

    def __post_init__(self):
        super().__post_init__()

        if self.saint_variant not in ("saint", "saint_s", "saint_i"):
            raise ValueError(
                f"Invalid saint_variant '{self.saint_variant}'. "
                f"Choices: 'saint', 'saint_s', 'saint_i'")

        if self.embedding_config.name == "identity":
            raise ValueError(
                "SAINT requires FeatureTokenizer embedding, not Identity. "
                "Use FTEmbeddingConfig.")

        if getattr(self.embedding_config, 'required_token_dim', 1) != 2:
            raise ValueError(
                "SAINT requires FTEmbeddingConfig with required_token_dim=2 "
                f"(got {self.embedding_config.required_token_dim})")

        if self.backbone_config.name != "saint":
            raise ValueError(
                f"SAINT requires SAINTBackboneConfig (name='saint'), "
                f"got name='{self.backbone_config.name}'")

        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        if not 0 < self.cutmix_probability < 1:
            raise ValueError(
                f"cutmix_probability must be in (0, 1), got {self.cutmix_probability}")
        if self.mixup_alpha <= 0:
            raise ValueError(f"mixup_alpha must be positive, got {self.mixup_alpha}")
