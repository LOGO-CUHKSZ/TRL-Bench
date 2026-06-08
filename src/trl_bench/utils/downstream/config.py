"""
Config loading and validation for downstream tasks.

Uses OmegaConf for YAML loading, CLI overrides, and struct mode
(typo detection on undefined keys).
"""

import os
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf, DictConfig


# Default config schema — tasks override these via YAML
_DEFAULT_CONFIG = {
    'task_name': '???',  # required
    'task_type': 'classification',

    'head': {
        'type': 'mlp',
        'input_dim': 'auto',
        'output_dim': 2,
        'hidden_dim': 256,
        'num_layers': 2,
        'activation': 'relu',
        'dropout': 0.1,
        'dropout_first': False,
    },

    'training': {
        'backend': 'pytorch',
        'combination_method': 'concat',

        'optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },

        'scheduler': {
            'type': 'none',
        },

        'loss': {
            'type': 'cross_entropy',
        },

        'batch_size': 32,
        'max_epochs': 10,
        'seed': 42,
        'device': 'auto',
        'deterministic': True,
        'debug': False,
        'eval_test_every_epoch': False,

        'early_stopping': None,
        'checkpointing': {
            'monitor': 'val_loss',
            'mode': 'min',
            'save_per_epoch': False,
        },

        'logging': {
            'wandb': False,
        },
    },

    'evaluation': {
        'metrics': ['accuracy'],
        'threshold': 0.5,
    },

    'linear_probe': {
        'C_values': [0.01, 0.1, 1.0, 10.0, 100.0],
        'alpha_values': [0.01, 0.1, 1.0, 10.0, 100.0],
        'fixed_C': 1.0,
        'fixed_alpha': 1.0,
        'max_iter': 5000,
        'normalize': True,
        'sweep': True,
        'refit_trainval': True,
    },
}


def load_config(
    path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    task_name: Optional[str] = None,
) -> DictConfig:
    """Load config from YAML file with optional CLI overrides.

    Args:
        path: Path to YAML config file. If None, uses defaults only.
        overrides: List of OmegaConf-style overrides like
            ['training.optimizer.lr=1e-4', 'head.hidden_dim=512'].
        task_name: Task name to set if not in config.

    Returns:
        Resolved DictConfig with struct mode enabled.
    """
    # Start with defaults
    cfg = OmegaConf.create(_DEFAULT_CONFIG)

    # Merge YAML if provided
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        yaml_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Set task name if provided and not already set
    if task_name is not None and cfg.task_name == '???':
        cfg.task_name = task_name

    # Enable struct mode to catch typos
    OmegaConf.set_struct(cfg, True)

    return cfg


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' device to actual device string."""
    if device_str == 'auto':
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def resolve_auto_values(
    cfg: DictConfig,
    input_dim: int,
    output_dim: int,
    multi_label: bool = False,
) -> DictConfig:
    """Resolve 'auto' values in config using runtime info.

    Modifies cfg in place (with struct temporarily disabled).

    Args:
        cfg: The config to resolve.
        input_dim: Detected input dimension from data.
        output_dim: Detected output dimension from data.
        multi_label: Whether this is a multi-label classification task.

    Returns:
        The same config with 'auto' values resolved.
    """
    # Temporarily disable struct to allow mutation
    was_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)

    if cfg.head.input_dim == 'auto':
        cfg.head.input_dim = input_dim
    if cfg.head.output_dim == 'auto':
        cfg.head.output_dim = output_dim

    # Resolve device
    cfg.training.device = resolve_device(cfg.training.device)

    # Resolve auto loss type
    if cfg.training.loss.type == 'auto':
        if cfg.task_type == 'regression':
            cfg.training.loss.type = 'mse'
        elif cfg.task_type == 'classification' and multi_label:
            cfg.training.loss.type = 'bce_with_logits'
        elif cfg.task_type == 'classification':
            cfg.training.loss.type = 'cross_entropy'

    OmegaConf.set_struct(cfg, was_struct)
    return cfg


def validate_monitor_keys(cfg: DictConfig) -> None:
    """Validate that monitor keys reference metrics that will be produced.

    The Trainer prefixes metric names with train_/val_/test_.
    This checks that scheduler.monitor, early_stopping.monitor, and
    checkpointing.monitor all reference valid prefixed metric names.

    Raises:
        ValueError if a monitor key references a metric that won't exist.
    """
    # Build the set of metrics that will be produced
    base_metrics = list(cfg.evaluation.metrics)
    valid_keys = {'train_loss', 'val_loss', 'test_loss'}
    for m in base_metrics:
        for prefix in ('train_', 'val_', 'test_'):
            valid_keys.add(f'{prefix}{m}')

    # Check each monitor key
    monitors_to_check = []

    # NB: pass an explicit default to .get(). Under OmegaConf struct mode
    # (set in load_config), `cfg.get('missing')` WITHOUT a default raises
    # ConfigAttributeError rather than returning None -- which broke every
    # row_prediction run (scheduler is `type: none`, no monitor key). The
    # explicit `None` default restores the intended "skip if absent" behavior
    # and also covers early_stopping/checkpointing blocks that omit monitor.
    if cfg.training.scheduler.get('monitor', None):
        monitors_to_check.append(
            ('scheduler.monitor', cfg.training.scheduler.monitor)
        )

    if cfg.training.early_stopping is not None and \
            cfg.training.early_stopping.get('monitor', None):
        monitors_to_check.append(
            ('early_stopping.monitor', cfg.training.early_stopping.monitor)
        )

    if cfg.training.checkpointing is not None and \
            cfg.training.checkpointing.get('monitor', None):
        monitors_to_check.append(
            ('checkpointing.monitor', cfg.training.checkpointing.monitor)
        )

    for source, key in monitors_to_check:
        if key not in valid_keys:
            raise ValueError(
                f"{source}='{key}' references a metric that won't be produced. "
                f"Valid keys: {sorted(valid_keys)}"
            )


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert config to a plain dict for serialization."""
    return OmegaConf.to_container(cfg, resolve=True)
