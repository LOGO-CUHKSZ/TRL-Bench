import torch
from torch import nn


class SAINTBlock(nn.Module):
    """A single SAINT block with self-attention and/or intersample attention."""

    def __init__(self, d_model, n_head, ffn_dim, dropout, variant="saint"):
        super().__init__()
        self.variant = variant

        if variant in ("saint", "saint_s"):
            self.self_attn_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True)

        if variant in ("saint", "saint_i"):
            self.inter_attn_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=False)

    def forward(self, x):
        # x: (B, N+1, d)
        if self.variant in ("saint", "saint_s"):
            x = self.self_attn_layer(x)         # attention across features
        if self.variant in ("saint", "saint_i"):
            x = x.permute(1, 0, 2)              # (N+1, B, d)
            x = self.inter_attn_layer(x)         # attention across rows
            x = x.permute(1, 0, 2)              # (B, N+1, d)
        return x


class SAINTEncoder(nn.Module):
    """Stack of SAINTBlock layers. Returns CLS token x[:, 0]."""

    def __init__(self, d_model, encoder_depth=6, n_head=8, ffn_factor=4.0,
                 dropout_rate=0.3, saint_variant="saint", **kwargs):
        super().__init__()

        # Auto-adjust n_head if d_model is not divisible
        if d_model % n_head != 0:
            divisors = [n for n in range(1, d_model + 1) if d_model % n == 0]
            closest_num_heads = min(divisors, key=lambda x: abs(x - n_head))
            if closest_num_heads != n_head:
                print(
                    f"Adjusting num_heads from {n_head} to {closest_num_heads} "
                    f"(closest valid divisor of {d_model})")
            n_head = closest_num_heads

        ffn_dim = int(d_model * ffn_factor)

        self.blocks = nn.ModuleList([
            SAINTBlock(d_model, n_head, ffn_dim, dropout_rate, saint_variant)
            for _ in range(encoder_depth)
        ])

        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N+1, d) from FeatureTokenizer
        for block in self.blocks:
            x = block(x)
        # Extract CLS token (position 0)
        return x[:, 0]
