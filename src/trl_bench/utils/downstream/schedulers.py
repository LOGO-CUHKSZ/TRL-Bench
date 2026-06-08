"""
Learning rate scheduler factory for downstream tasks.

Returns (scheduler, step_event) where step_event is either 'batch' or 'epoch',
indicating when the scheduler's .step() should be called.
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from typing import Any, Dict, Optional, Tuple


def _build_linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then linear decay to 0. Same as HuggingFace's get_linear_schedule_with_warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    num_training_steps: Optional[int] = None,
) -> Tuple[Optional[Any], str]:
    """Build a scheduler from config.

    Args:
        config: Scheduler config dict with 'type' and type-specific params.
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of training steps (batches), needed
            for linear_warmup_decay and cosine schedulers.

    Returns:
        (scheduler, step_event) where step_event is 'batch', 'epoch', or
        'metric' (for ReduceLROnPlateau which needs a metric value).
        Returns (None, 'none') if type is 'none'.
    """
    sched_type = config.get('type', 'none')

    if sched_type == 'none':
        return None, 'none'

    elif sched_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            patience=config.get('patience', 5),
            factor=config.get('factor', 0.5),
        )
        return scheduler, 'metric'

    elif sched_type == 'linear_warmup_decay':
        if num_training_steps is None:
            raise ValueError("linear_warmup_decay requires num_training_steps")
        warmup_steps = config.get('warmup_steps', 0)
        scheduler = _build_linear_warmup_decay(
            optimizer, warmup_steps, num_training_steps
        )
        return scheduler, 'batch'

    elif sched_type == 'cosine':
        if num_training_steps is None:
            raise ValueError("cosine scheduler requires num_training_steps")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.get('eta_min', 0),
        )
        return scheduler, 'batch'

    else:
        raise ValueError(f"Unknown scheduler type: '{sched_type}'")
