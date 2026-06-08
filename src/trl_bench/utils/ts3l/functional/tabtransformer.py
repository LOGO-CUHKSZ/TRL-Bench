from typing import Tuple, List
import torch
from torch import nn


def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Forward step for TabTransformer-SSL during pretraining.

    Args:
        model: TabTransformerSSL instance.
        batch: (x_original, x_corrupted, mlm_mask, rtd_labels)

    Returns:
        (mlm_logits, rtd_logits) — mlm_logits is a list of per-column logits.
    """
    _, x_corrupted, mlm_mask, _ = batch
    mlm_logits, rtd_logits = model(x_corrupted, mlm_mask)
    return mlm_logits, rtd_logits


def first_phase_loss(
    x_original: torch.Tensor,
    mlm_mask: torch.Tensor,
    rtd_labels: torch.Tensor,
    mlm_logits: List[torch.Tensor],
    rtd_logits: torch.Tensor,
    mlm_loss_fn: nn.Module,
    rtd_loss_fn: nn.Module,
    mlm_weight: float,
    rtd_weight: float,
    n_cat: int,
) -> torch.Tensor:
    """Compute combined MLM + RTD loss.

    MLM: cross-entropy on masked column positions only.
    RTD: BCE on non-masked column positions only (since the model never saw
         the masked positions, it can't detect replacement there).

    Args:
        x_original: (B, n_cat + n_cont). First n_cat columns are categorical.
        mlm_mask: (B, n_cat) bool tensor. True = masked for MLM.
        rtd_labels: (B, n_cat) float tensor. 1.0 = replaced.
        mlm_logits: List of (B, card_i) tensors, one per categorical column.
        rtd_logits: (B, n_cat) logits.
        mlm_loss_fn: CrossEntropyLoss.
        rtd_loss_fn: BCEWithLogitsLoss(reduction='none').
        mlm_weight: Weight for MLM loss term.
        rtd_weight: Weight for RTD loss term.
        n_cat: Number of categorical columns.

    Returns:
        Combined scalar loss.
    """
    device = rtd_logits.device

    # --- MLM loss: CE averaged over all masked positions (per-position) ---
    mlm_loss = torch.tensor(0.0, device=device)
    all_mlm_losses = []
    for col_idx in range(n_cat):
        col_mask = mlm_mask[:, col_idx]  # (B,) bool
        if col_mask.sum() == 0:
            continue
        targets = x_original[col_mask, col_idx].long()  # ground truth category
        valid = targets >= 0
        if valid.sum() == 0:
            continue
        preds = mlm_logits[col_idx][col_mask]            # (n_masked, card_i)
        per_sample = mlm_loss_fn(preds, targets)         # reduction='none' → (n_masked,)
        all_mlm_losses.append(per_sample[valid])

    if all_mlm_losses:
        mlm_loss = torch.cat(all_mlm_losses).mean()

    # --- RTD loss: BCE only on non-masked positions ---
    rtd_loss = torch.tensor(0.0, device=device)
    not_mlm = ~mlm_mask  # (B, n_cat) — positions where RTD is valid
    n_valid = not_mlm.sum()
    if n_valid > 0:
        raw_rtd = rtd_loss_fn(rtd_logits, rtd_labels)  # (B, n_cat)
        rtd_loss = (raw_rtd * not_mlm.float()).sum() / n_valid

    return mlm_weight * mlm_loss + rtd_weight * rtd_loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step for TabTransformer-SSL during fine-tuning.

    Args:
        model: TabTransformerSSL instance.
        batch: (x, y) tuple.

    Returns:
        Predicted logits (squeezed).
    """
    x, _ = batch
    return model(x).squeeze()


def second_phase_loss(
    y: torch.Tensor, y_hat: torch.Tensor, loss_fn: nn.Module
) -> torch.Tensor:
    """Standard task loss for phase 2."""
    return loss_fn(y_hat, y)
