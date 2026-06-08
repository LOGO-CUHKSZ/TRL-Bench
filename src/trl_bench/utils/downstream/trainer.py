"""
Config-driven PyTorch Trainer for downstream tasks.

Replaces both the Lightning-based pipeline (run_task.py) and the manual
training loops in individual task scripts with a single, unified loop.

The Trainer owns:
    - model.train() / model.eval() toggling
    - torch.no_grad() context for evaluation
    - Gradient zeroing, backward, optimizer step
    - Scheduler stepping (per-batch or per-epoch)
    - Early stopping, checkpointing
    - Metric computation and logging
    - Deterministic seeding

TaskSpec implementations own:
    - forward_and_loss(): forward computation + loss calculation
    - compute_metrics(): metric computation from raw outputs
"""

import inspect
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from .config import config_to_dict, validate_monitor_keys
from .heads import build_head
from .losses import build_loss
from .metrics import compute_metrics as _compute_metrics, METRIC_DIRECTIONS
from .schedulers import build_scheduler


# ---------------------------------------------------------------------------
# TaskSpec protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TaskSpec(Protocol):
    """Protocol for task-specific forward and metric logic.

    Implementations return (loss, logits). They must NOT call model.train(),
    model.eval(), or torch.no_grad() — the Trainer owns those.
    Outputs must always be raw logits (no sigmoid/softmax).
    """

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (loss, logits).

        Args:
            model: The head model.
            batch: A single batch from the DataLoader.
            device: Target device.
            criterion: The loss module.

        Returns:
            (loss, logits) where loss is a scalar tensor and logits
            are raw model outputs (no activation applied).
        """
        ...

    def extract_targets(self, batch: Any) -> Any:
        """Extract target labels from a batch for metric computation.

        Default: batch[1] for tuple batches, batch['label'] for dicts.
        Override for custom batch formats.
        """
        ...

    def compute_metrics(
        self,
        outputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics from collected outputs and targets.

        Args:
            outputs: Concatenated raw logits from all batches.
            targets: Concatenated ground truth labels.

        Returns:
            Dict of metric_name -> value (unprefixed names).
        """
        ...


# ---------------------------------------------------------------------------
# DefaultTaskSpec — handles standard (emb, label) batches
# ---------------------------------------------------------------------------

class DefaultTaskSpec:
    """Default TaskSpec for simple (embedding, label) batches.

    Covers: single-label classification (CE), multi-label (BCE),
    and regression (MSE). The loss type is inferred from the config.
    """

    def __init__(self, cfg: DictConfig):
        self.loss_type = cfg.training.loss.type
        self.metric_names = list(cfg.evaluation.metrics)
        self.threshold = cfg.evaluation.get('threshold', 0.5)

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        criterion: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, labels = batch
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)

        if self.loss_type == 'mse':
            loss = criterion(logits.squeeze(-1), labels)
        else:
            loss = criterion(logits, labels)

        return loss, logits

    def extract_targets(self, batch: Any) -> Any:
        """Extract targets from batch. Handles tuple and dict formats."""
        if isinstance(batch, (tuple, list)):
            return batch[1]
        elif isinstance(batch, dict):
            return batch.get('label', batch.get('labels'))
        return None

    def compute_metrics(
        self,
        outputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        return _compute_metrics(
            outputs, targets, self.metric_names, threshold=self.threshold
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a value or tuple of values to numpy."""
    if isinstance(x, tuple):
        return tuple(_to_numpy(v) for v in x)
    elif hasattr(x, 'cpu'):
        return x.cpu().numpy()
    elif hasattr(x, 'numpy'):
        return x.numpy()
    else:
        return np.asarray(x)


def _padded_concat(arrays, axis=0):
    """Concatenate arrays, padding to max shape along non-concat axes.

    Handles the case where arrays have different sizes on axes other than
    the concatenation axis (e.g., variable-length columns in CT prediction
    where each batch has a different max number of columns).
    """
    if not arrays:
        return None
    if arrays[0].ndim <= 1:
        return np.concatenate(arrays, axis=axis)

    # Check if all shapes match on non-concat axes
    shapes = [a.shape for a in arrays]
    non_concat_dims = [i for i in range(arrays[0].ndim) if i != axis]
    needs_padding = any(
        shapes[0][d] != s[d] for s in shapes[1:] for d in non_concat_dims
    )

    if not needs_padding:
        return np.concatenate(arrays, axis=axis)

    # Compute max shape for non-concat dims
    max_shape = list(shapes[0])
    for s in shapes[1:]:
        for d in non_concat_dims:
            max_shape[d] = max(max_shape[d], s[d])

    padded = []
    for a in arrays:
        if list(a.shape) == max_shape[:axis] + [a.shape[axis]] + max_shape[axis+1:]:
            padded.append(a)
        else:
            pad_width = [(0, max_shape[d] - a.shape[d]) if d != axis else (0, 0)
                         for d in range(a.ndim)]
            padded.append(np.pad(a, pad_width, mode='constant', constant_values=0))

    return np.concatenate(padded, axis=axis)


def _concat_targets(target_list):
    """Concatenate collected targets. Handles tuples (e.g., (labels, masks))."""
    if not target_list:
        return None
    first = target_list[0]
    if isinstance(first, tuple):
        # Zip and concatenate each element separately
        return tuple(
            _padded_concat([t[i] for t in target_list], axis=0)
            for i in range(len(first))
        )
    return _padded_concat(target_list, axis=0)


def move_batch_to_device(batch, device):
    """Recursively move a batch (tensor, tuple, list, dict) to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        moved = [move_batch_to_device(x, device) for x in batch]
        return type(batch)(moved)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    else:
        return batch  # non-tensor (e.g., list of strings)


def seed_everything(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    """Seed DataLoader workers for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class _EarlyStopping:
    """Track a monitored metric and signal when to stop."""

    def __init__(self, patience: int, mode: str = 'min'):
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (value < self.best) if self.mode == 'min' else (value > self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

class _Checkpointer:
    """Track best metric and save model checkpoint."""

    def __init__(self, output_dir: str, mode: str = 'min',
                 save_per_epoch: bool = False,
                 format_fn: Optional[Callable] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.save_per_epoch = save_per_epoch
        self.format_fn = format_fn
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1

        # Detect whether format_fn accepts `is_best`
        self._fmt_accepts_is_best = False
        if format_fn is not None:
            try:
                sig = inspect.signature(format_fn)
                self._fmt_accepts_is_best = (
                    'is_best' in sig.parameters
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in sig.parameters.values())
                )
            except (TypeError, ValueError):
                pass

    def step(self, value: float, epoch: int, model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             metrics: Optional[Dict] = None,
             config: Optional[Dict] = None) -> bool:
        """Check if metric improved and save checkpoint if so.

        Returns True if this is a new best.
        """
        improved = (value < self.best) if self.mode == 'min' else (value > self.best)

        if improved:
            self.best = value
            self.best_epoch = epoch
            self._save(
                self.output_dir / 'best_model.pt',
                epoch, model, optimizer, metrics, config,
                is_best=True,
            )

        if self.save_per_epoch:
            self._save(
                self.output_dir / f'checkpoint_epoch_{epoch}.pt',
                epoch, model, optimizer, metrics, config,
                is_best=False,
            )

        return improved

    def _save(self, path: Path, epoch: int, model: nn.Module,
              optimizer: Optional[torch.optim.Optimizer],
              metrics: Optional[Dict], config: Optional[Dict],
              is_best: bool = False):
        if self.format_fn is not None:
            if self._fmt_accepts_is_best:
                checkpoint = self.format_fn(
                    model=model, optimizer=optimizer, epoch=epoch,
                    metrics=metrics, config=config, is_best=is_best,
                )
            else:
                checkpoint = self.format_fn(
                    model=model, optimizer=optimizer, epoch=epoch,
                    metrics=metrics, config=config,
                )
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'metrics': metrics,
            }
        torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# Debug assertions
# ---------------------------------------------------------------------------

def _debug_check_batch(logits, labels, loss, loss_type, mask=None):
    """Validate shapes and dtypes. Called once per epoch in debug mode."""
    assert logits.shape[0] == labels.shape[0], \
        f"Batch size mismatch: logits {logits.shape[0]} vs labels {labels.shape[0]}"

    if loss_type == 'cross_entropy':
        assert labels.dtype == torch.long, f"CE requires long labels, got {labels.dtype}"
        assert labels.min() >= 0, f"Labels must be non-negative, got min={labels.min()}"
        assert labels.max() < logits.shape[-1], \
            f"Label {labels.max()} >= num_classes {logits.shape[-1]}"
    elif loss_type in ('bce_with_logits', 'masked_bce'):
        assert labels.dtype == torch.float, f"BCE requires float labels, got {labels.dtype}"
        assert labels.min() >= 0, f"Labels must be >= 0, got min={labels.min()}"
        assert labels.max() <= 1, f"Labels must be <= 1, got max={labels.max()}"
    elif loss_type == 'mse':
        assert labels.dtype == torch.float, f"MSE requires float labels, got {labels.dtype}"

    if mask is not None:
        assert mask.dtype in (torch.bool, torch.float), \
            f"Mask dtype must be bool or float, got {mask.dtype}"
        assert mask.any(), "Mask is all-zero — no valid elements"

    assert torch.isfinite(loss), f"Loss is NaN or Inf: {loss.item()}"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Config-driven PyTorch training loop.

    Usage:
        trainer = Trainer(config, output_dir)
        trainer.setup(train_loader, val_loader, test_loader,
                      input_dim=768, output_dim=2)
        result = trainer.fit(task_spec=None)
        test_result = trainer.test(task_spec=None)
    """

    def __init__(
        self,
        config: DictConfig,
        output_dir: str,
        checkpoint_format_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config: Full task config (from load_config).
            output_dir: Directory for checkpoints and results.
            checkpoint_format_fn: Optional callable(model, optimizer, epoch,
                metrics, config) -> dict for custom checkpoint format.
        """
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_format_fn = checkpoint_format_fn

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scheduler_step_event = 'none'
        self.criterion = None
        self.device = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        model: Optional[nn.Module] = None,
        multi_label: bool = False,
    ):
        """Initialize model, optimizer, scheduler, and loss.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader (can be None).
            test_loader: Test DataLoader (can be None).
            input_dim: Input dimension (used to resolve head.input_dim='auto').
            output_dim: Output dimension (used to resolve head.output_dim='auto').
            model: Optional pre-built model. If None, builds from config.
            multi_label: Whether this is a multi-label classification task.
                Used to resolve loss.type='auto' -> 'bce_with_logits'.
        """
        cfg = self.cfg

        # Seed
        seed_everything(cfg.training.seed, cfg.training.deterministic)

        # Resolve auto values
        from .config import resolve_auto_values
        if input_dim is not None or output_dim is not None:
            resolve_auto_values(
                cfg,
                input_dim=input_dim or 0,
                output_dim=output_dim or 0,
                multi_label=multi_label,
            )

        self.device = torch.device(cfg.training.device)

        # Build or use provided model
        if model is not None:
            self.model = model.to(self.device)
        else:
            head_cfg = config_to_dict(cfg.head)
            self.model = build_head(head_cfg).to(self.device)

        # Build loss
        loss_cfg = config_to_dict(cfg.training.loss)
        self.criterion = build_loss(loss_cfg)

        # Build optimizer
        opt_cfg = cfg.training.optimizer
        opt_type = opt_cfg.type.lower()
        opt_params = {
            'lr': opt_cfg.lr,
            'weight_decay': opt_cfg.get('weight_decay', 0),
        }
        if opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_params)
        elif opt_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **opt_params)
        elif opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.get('weight_decay', 0),
                momentum=opt_cfg.get('momentum', 0),
            )
        else:
            raise ValueError(f"Unknown optimizer: '{opt_type}'")

        # Build scheduler
        sched_cfg = config_to_dict(cfg.training.scheduler)
        num_training_steps = len(train_loader) * cfg.training.max_epochs
        self.scheduler, self.scheduler_step_event = build_scheduler(
            sched_cfg, self.optimizer, num_training_steps
        )

        # Validate monitor keys
        validate_monitor_keys(cfg)

        # Store loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Print summary
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainer setup complete:")
        print(f"  Model: {n_params:,} parameters")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: {opt_type} (lr={opt_cfg.lr})")
        print(f"  Scheduler: {sched_cfg.get('type', 'none')}")
        print(f"  Loss: {cfg.training.loss.type}")

    def fit(
        self,
        task_spec: Optional[TaskSpec] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            task_spec: TaskSpec implementation. If None, uses DefaultTaskSpec.

        Returns:
            Dict with training history and best metrics.
        """
        cfg = self.cfg
        if task_spec is None:
            task_spec = DefaultTaskSpec(cfg)

        # Early stopping
        early_stopper = None
        es_cfg = cfg.training.early_stopping
        if es_cfg is not None:
            # Auto-fallback: if monitoring val_* but no val loader, switch to train_*
            if self.val_loader is None and es_cfg.monitor.startswith('val_'):
                fallback = 'train_' + es_cfg.monitor[len('val_'):]
                print(f"Warning: no val loader, early_stopping monitor "
                      f"'{es_cfg.monitor}' -> '{fallback}'")
                OmegaConf.set_struct(cfg, False)
                cfg.training.early_stopping.monitor = fallback
                OmegaConf.set_struct(cfg, True)
            early_stopper = _EarlyStopping(
                patience=es_cfg.patience,
                mode=es_cfg.mode,
            )

        # Checkpointing
        ckpt = None
        ckpt_cfg = cfg.training.checkpointing
        if ckpt_cfg is not None:
            # Auto-fallback: if monitoring val_* but no val loader, switch to train_*
            if self.val_loader is None and ckpt_cfg.monitor.startswith('val_'):
                fallback = 'train_' + ckpt_cfg.monitor[len('val_'):]
                print(f"Warning: no val loader, checkpointing monitor "
                      f"'{ckpt_cfg.monitor}' -> '{fallback}'")
                OmegaConf.set_struct(cfg, False)
                cfg.training.checkpointing.monitor = fallback
                OmegaConf.set_struct(cfg, True)
            ckpt = _Checkpointer(
                output_dir=str(self.output_dir),
                mode=ckpt_cfg.mode,
                save_per_epoch=ckpt_cfg.get('save_per_epoch', False),
                format_fn=self.checkpoint_format_fn,
            )

        # Wandb
        wandb_run = None
        if cfg.training.logging.get('wandb', False):
            try:
                import wandb
                wandb_run = wandb.init(
                    project=cfg.training.logging.get('wandb_project', cfg.task_name),
                    name=cfg.training.logging.get('wandb_run_name'),
                    config=config_to_dict(cfg),
                    dir=str(self.output_dir),
                )
            except ImportError:
                print("Warning: wandb not installed, skipping wandb logging")

        history = []
        debug_mode = cfg.training.get('debug', False)
        _warned_missing_monitors = set()

        for epoch in range(1, cfg.training.max_epochs + 1):
            # --- Train ---
            train_metrics = self.train_epoch(
                task_spec, epoch, debug_mode=debug_mode
            )

            # --- Validate ---
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.eval_epoch(
                    self.val_loader, task_spec, prefix='val'
                )

            # --- Test (optional, every epoch) ---
            test_metrics = {}
            if (self.test_loader is not None
                    and cfg.training.get('eval_test_every_epoch', False)):
                test_metrics = self.eval_epoch(
                    self.test_loader, task_spec, prefix='test'
                )

            # Combine metrics for this epoch
            epoch_metrics = {**train_metrics, **val_metrics, **test_metrics}

            # Scheduler step
            if self.scheduler is not None and self.scheduler_step_event == 'metric':
                monitor_key = cfg.training.scheduler.get('monitor', 'val_loss')
                if monitor_key in epoch_metrics:
                    self.scheduler.step(epoch_metrics[monitor_key])

            # Log
            epoch_log = {'epoch': epoch, **epoch_metrics}
            history.append(epoch_log)
            self._print_epoch(epoch, cfg.training.max_epochs, epoch_metrics)

            if wandb_run is not None:
                wandb_run.log(epoch_log)

            # Checkpointing
            if ckpt is not None:
                ckpt_monitor = ckpt_cfg.monitor
                if ckpt_monitor in epoch_metrics:
                    is_best = ckpt.step(
                        epoch_metrics[ckpt_monitor], epoch, self.model,
                        self.optimizer, epoch_metrics, config_to_dict(cfg)
                    )
                    if is_best:
                        print(f"  -> New best model (epoch {epoch}, "
                              f"{ckpt_monitor}={epoch_metrics[ckpt_monitor]:.4f})")
                elif ckpt_monitor not in _warned_missing_monitors:
                    print(f"  Warning: checkpointing monitor '{ckpt_monitor}' "
                          f"not in epoch metrics {list(epoch_metrics.keys())}. "
                          f"No checkpoint will be saved until this key appears.")
                    _warned_missing_monitors.add(ckpt_monitor)

            # Early stopping
            if early_stopper is not None:
                es_monitor = es_cfg.monitor
                if es_monitor in epoch_metrics:
                    if early_stopper.step(epoch_metrics[es_monitor]):
                        print(f"  Early stopping triggered (patience={es_cfg.patience})")
                        break
                elif es_monitor not in _warned_missing_monitors:
                    print(f"  Warning: early_stopping monitor '{es_monitor}' "
                          f"not in epoch metrics {list(epoch_metrics.keys())}. "
                          f"Early stopping is inactive.")
                    _warned_missing_monitors.add(es_monitor)

        if wandb_run is not None:
            wandb_run.finish()

        # Build result
        result = {
            'history': history,
            'best_epoch': ckpt.best_epoch if ckpt else len(history),
            'best_value': ckpt.best if ckpt else None,
        }

        return result

    def test(
        self,
        task_spec: Optional[TaskSpec] = None,
        load_best: bool = True,
    ) -> Dict[str, float]:
        """Evaluate on the test set.

        Args:
            task_spec: TaskSpec implementation. If None, uses DefaultTaskSpec.
            load_best: If True, load best_model.pt before testing.

        Returns:
            Dict of test metrics (test_loss, test_accuracy, etc.).
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided to setup()")

        if task_spec is None:
            task_spec = DefaultTaskSpec(self.cfg)

        if load_best:
            best_path = self.output_dir / 'best_model.pt'
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt)
                self.model.load_state_dict(state_dict)
                print(f"Loaded best model from {best_path}")

        return self.eval_epoch(self.test_loader, task_spec, prefix='test')

    # -------------------------------------------------------------------
    # Internal training/eval loops
    # -------------------------------------------------------------------

    def train_epoch(
        self,
        task_spec: TaskSpec,
        epoch: int,
        debug_mode: bool = False,
    ) -> Dict[str, float]:
        """Run one training epoch. Returns dict with train_loss and train metrics."""
        self.model.train()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        n_batches = 0
        checked_debug = False

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()

            loss, logits = task_spec.forward_and_loss(
                self.model, batch, self.device, self.criterion
            )

            # Debug checks (once per epoch)
            if debug_mode and not checked_debug:
                # Extract labels from batch for checking
                if isinstance(batch, (tuple, list)):
                    labels_for_check = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    labels_for_check = None
                if labels_for_check is not None:
                    _debug_check_batch(
                        logits, labels_for_check, loss,
                        self.cfg.training.loss.type
                    )
                checked_debug = True

            loss.backward()

            # Debug: check gradients are finite
            if debug_mode:
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), \
                            f"Non-finite gradient in {name}"

            self.optimizer.step()

            # Per-batch scheduler step
            if self.scheduler is not None and self.scheduler_step_event == 'batch':
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            # Collect outputs for metrics (detach to avoid graph retention)
            all_outputs.append(logits.detach().cpu().numpy())
            raw_targets = task_spec.extract_targets(batch)
            if raw_targets is not None:
                all_targets.append(_to_numpy(raw_targets))

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(n_batches, 1)
        result = {'train_loss': avg_loss}

        # Compute train metrics
        if all_outputs and all_targets:
            outputs = _padded_concat(all_outputs, axis=0)
            targets = _concat_targets(all_targets)
            train_metrics = task_spec.compute_metrics(outputs, targets)
            for k, v in train_metrics.items():
                result[f'train_{k}'] = v

        return result

    def eval_epoch(
        self,
        loader: DataLoader,
        task_spec: TaskSpec,
        prefix: str = 'val',
    ) -> Dict[str, float]:
        """Run one evaluation epoch. Returns dict with prefixed metrics."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                loss, logits = task_spec.forward_and_loss(
                    self.model, batch, self.device, self.criterion
                )

                total_loss += loss.item()
                n_batches += 1
                all_outputs.append(logits.cpu().numpy())

                raw_targets = task_spec.extract_targets(batch)
                if raw_targets is not None:
                    all_targets.append(_to_numpy(raw_targets))

        avg_loss = total_loss / max(n_batches, 1)
        result = {f'{prefix}_loss': avg_loss}

        if all_outputs and all_targets:
            outputs = _padded_concat(all_outputs, axis=0)
            targets = _concat_targets(all_targets)
            metrics = task_spec.compute_metrics(outputs, targets)
            for k, v in metrics.items():
                result[f'{prefix}_{k}'] = v

        return result

    def _print_epoch(self, epoch: int, max_epochs: int,
                     metrics: Dict[str, float]):
        """Print a single-line epoch summary."""
        parts = [f"Epoch {epoch}/{max_epochs}"]
        for key in sorted(metrics.keys()):
            parts.append(f"{key}={metrics[key]:.4f}")
        print("  ".join(parts))
