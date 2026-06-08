"""
Loss function factory and custom losses for downstream tasks.
"""

import torch
import torch.nn as nn
from typing import Any, Dict


class MaskedBCELoss(nn.Module):
    """Masked BCE with logits for variable-length multi-label classification.

    Used by column type prediction where tables have variable numbers of
    columns and a binary mask indicates which columns are valid.

    Input shapes:
        logits: (B, T, C) - batch, tokens/columns, classes
        targets: (B, T, C) - multi-hot labels
        mask: (B, T) - 1 for valid columns, 0 for padding

    Loss = sum(BCE_per_column * mask) / sum(mask)
    where BCE_per_column is averaged over the class dimension first.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> per-element loss
        loss = self.bce(logits, targets)
        # Average over class dim -> (B, T)
        loss = loss.mean(dim=-1)
        # Mask and average over valid positions
        mask_sum = mask.sum().clamp(min=1)
        return torch.sum(loss * mask) / mask_sum


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build a loss function from config.

    Args:
        config: Dict with at minimum a 'type' key. Supported types:
            - cross_entropy
            - bce_with_logits
            - mse
            - masked_bce

    Returns:
        nn.Module loss function.
    """
    loss_type = config.get('type', 'cross_entropy')

    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'bce_with_logits':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'masked_bce':
        return MaskedBCELoss()
    elif loss_type == 'auto':
        raise ValueError(
            "Loss type 'auto' must be resolved before calling build_loss(). "
            "Use resolve_auto_values() first."
        )
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")
