from typing import Tuple, List
import torch
from torch import nn


def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Forward step of SAINT during the first phase (SSL).

    Args:
        model: An instance of SAINT.
        batch: (x, x_cutmix) tensors.

    Returns:
        Tuple of (proj_orig, proj_mixed, cat_preds, cont_preds).
    """
    x, x_cutmix = batch
    proj_orig, proj_mixed, cat_preds, cont_preds = model(x, x_cutmix)
    return proj_orig, proj_mixed, cat_preds, cont_preds


def contrastive_loss(
    proj_orig: torch.Tensor,
    proj_mixed: torch.Tensor,
    loss_fn: nn.Module
) -> torch.Tensor:
    """Compute contrastive loss between original and mixed projections.

    Args:
        proj_orig: L2-normalized projections from original view.
        proj_mixed: L2-normalized projections from mixed view.
        loss_fn: NTXentLoss instance.

    Returns:
        Contrastive loss scalar.
    """
    return loss_fn(proj_orig, proj_mixed)


def denoising_loss(
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
    cat_preds: List[torch.Tensor],
    cont_preds: torch.Tensor,
    cat_loss_fn: nn.Module,
    cont_loss_fn: nn.Module,
) -> torch.Tensor:
    """Compute denoising (reconstruction) loss.

    Follows DAE pattern: per-feature CE for categoricals + MSE for continuous.

    Args:
        x_cat: Original categorical features (B, n_cat).
        x_cont: Original continuous features (B, n_cont).
        cat_preds: List of per-category logit tensors.
        cont_preds: Predicted continuous features (B, n_cont).
        cat_loss_fn: CrossEntropyLoss with ignore_index=-1, reduction='none'.
        cont_loss_fn: MSELoss.

    Returns:
        Reconstruction loss scalar.
    """
    feature_loss = torch.tensor(0.0, device=x_cat.device)

    if x_cat.shape[1] > 0:
        for idx in range(x_cat.shape[1]):
            targets = x_cat[:, idx].long()
            valid = targets >= 0
            if valid.sum() > 0:
                per_sample = cat_loss_fn(cat_preds[idx], targets)
                feature_loss = feature_loss + per_sample[valid].mean()

    if x_cont.shape[1] > 0:
        feature_loss = feature_loss + cont_loss_fn(cont_preds, x_cont)

    return feature_loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of SAINT during the second phase (supervised).

    Args:
        model: An instance of SAINT.
        batch: (x, y) tensors.

    Returns:
        Predicted labels (logits).
    """
    x, _ = batch
    return model(x).squeeze()


def second_phase_loss(
    y: torch.Tensor, y_hat: torch.Tensor, loss_fn: nn.Module
) -> torch.Tensor:
    """Calculate the second phase loss.

    Args:
        y: Ground truth labels.
        y_hat: Predicted labels.
        loss_fn: Task loss function.

    Returns:
        Task loss scalar.
    """
    return loss_fn(y_hat, y)
