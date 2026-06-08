from dataclasses import dataclass, field
from .base_backbone_config import BaseBackboneConfig
from typing import Optional


@dataclass
class SAINTBackboneConfig(BaseBackboneConfig):
    d_model: Optional[int] = field(default=None)
    encoder_depth: int = field(default=6)
    n_head: int = field(default=8)
    ffn_factor: float = field(default=4.0)
    saint_variant: str = field(default="saint")

    def __post_init__(self):
        self.name = "saint"

        if self.d_model is None:
            raise TypeError(
                "__init__ missing 1 required positional argument: 'd_model'")

        if self.saint_variant not in ("saint", "saint_s", "saint_i"):
            raise ValueError(
                f"Invalid saint_variant '{self.saint_variant}'. "
                f"Choices: 'saint', 'saint_s', 'saint_i'")
