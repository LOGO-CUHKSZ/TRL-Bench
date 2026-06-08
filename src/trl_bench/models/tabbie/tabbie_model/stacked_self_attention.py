"""
Stacked Self-Attention for TABBIE

Patched from the original TABBIE repo (AllenNLP-based).
Module names are chosen to match the checkpoint key structure:

  Checkpoint pattern (per transformer block, new-style keys):
    self_attention_0._combined_projection.{weight,bias}  (2304, 768) / (2304,)
    self_attention_0._output_projection.{weight,bias}    (768, 768) / (768,)
    feedforward_0._linear_layers.0.{weight,bias}         (3072, 768) / (3072,)
    feedforward_0._linear_layers.1.{weight,bias}         (768, 3072) / (768,)
    layer_norm_0.{gamma,beta}                            (768,)
    feedforward_layer_norm_0.{gamma,beta}                (768,)

  Note: gamma/beta are old PyTorch LayerNorm names for weight/bias.
  The load_weights() remap handles this.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with combined Q/K/V projection.

    The checkpoint uses a single combined projection of size (3*hidden_dim, hidden_dim)
    that packs Q, K, V together, rather than separate projections.
    """

    def __init__(self, input_dim: int, num_heads: int, dropout_prob: float = 0.1):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = input_dim // num_heads
        self._input_dim = input_dim

        # Combined Q/K/V projection: (3*input_dim, input_dim)
        self._combined_projection = nn.Linear(input_dim, 3 * input_dim)
        self._output_projection = nn.Linear(input_dim, input_dim)
        self._attention_dropout = nn.Dropout(dropout_prob)
        self._scale = math.sqrt(self._head_dim)

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, input_dim)
            mask: (batch, seq_len) bool tensor, True = keep
        Returns:
            (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = inputs.size()
        num_heads = self._num_heads
        head_dim = self._head_dim

        # Combined projection → split into Q, K, V
        combined = self._combined_projection(inputs)  # (batch, seq_len, 3*input_dim)
        q, k, v = combined.chunk(3, dim=-1)  # each (batch, seq_len, input_dim)

        # Reshape to (batch*heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.contiguous().view(batch_size * num_heads, seq_len, head_dim)

        k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.contiguous().view(batch_size * num_heads, seq_len, head_dim)

        v = v.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.contiguous().view(batch_size * num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        attn = torch.bmm(q, k.transpose(1, 2)) / self._scale

        # Mask
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, num_heads, -1)
            expanded_mask = expanded_mask.reshape(batch_size * num_heads, 1, seq_len)
            attn = attn.masked_fill(~expanded_mask.bool(), -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self._attention_dropout(attn)

        # Weighted sum
        out = torch.bmm(attn, v)  # (batch*heads, seq_len, head_dim)

        # Reshape back
        out = out.view(batch_size, num_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self._input_dim)

        return self._output_projection(out)


class FeedForward(nn.Module):
    """Feedforward network matching checkpoint's _linear_layers ModuleList naming."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self._linear_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        ])
        self._activation = nn.ReLU()

    def forward(self, x):
        x = self._activation(self._linear_layers[0](x))
        return self._linear_layers[1](x)


class TransformerBlock(nn.Module):
    """Single transformer block matching TABBIE checkpoint naming.

    Module names match the checkpoint's new-style keys:
      self_attention_0, feedforward_0, layer_norm_0, feedforward_layer_norm_0
    """

    def __init__(
        self,
        input_dim: int = 768,
        feedforward_hidden_dim: int = 3072,
        num_attention_heads: int = 12,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        # Names must match checkpoint keys exactly
        self.self_attention_0 = CombinedMultiHeadSelfAttention(
            input_dim=input_dim,
            num_heads=num_attention_heads,
            dropout_prob=dropout_prob,
        )
        self.feedforward_0 = FeedForward(input_dim, feedforward_hidden_dim)
        self.layer_norm_0 = nn.LayerNorm(input_dim)
        self.feedforward_layer_norm_0 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, input_dim)
            mask: (batch, seq_len) bool tensor
        Returns:
            (batch, seq_len, input_dim)
        """
        # Self-attention with residual + layernorm
        attn_out = self.self_attention_0(inputs, mask)
        attn_out = self.dropout(attn_out)
        x = self.layer_norm_0(inputs + attn_out)

        # Feedforward with residual + layernorm
        ff_out = self.feedforward_0(x)
        ff_out = self.dropout(ff_out)
        out = self.feedforward_layer_norm_0(x + ff_out)

        return out
