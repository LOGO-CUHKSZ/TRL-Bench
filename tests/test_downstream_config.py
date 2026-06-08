"""Regression tests for the downstream-probe config loader.

Guards the OmegaConf struct-mode interaction in ``validate_monitor_keys``:
a scheduler/early_stopping/checkpointing block that omits ``monitor`` must be
skipped, not raise. Recent OmegaConf versions make ``cfg.get('missing')``
raise under struct mode unless an explicit default is passed -- this broke
every row_prediction MLP run (whose scheduler is ``type: none``, no monitor).
"""
from omegaconf import OmegaConf

from trl_bench.utils.downstream.config import load_config, validate_monitor_keys


def test_validate_monitor_keys_scheduler_without_monitor_struct_mode():
    """scheduler={type: none} (no monitor) under struct mode must not raise."""
    cfg = OmegaConf.create({
        "training": {
            "scheduler": {"type": "none"},                 # no 'monitor' key
            "early_stopping": None,
            "checkpointing": {"monitor": "val_loss", "mode": "min"},
        },
        "evaluation": {"metrics": ["accuracy"]},
    })
    OmegaConf.set_struct(cfg, True)
    validate_monitor_keys(cfg)  # must not raise


def test_validate_monitor_keys_early_stopping_without_monitor_struct_mode():
    """early_stopping present but missing monitor must be skipped, not raise."""
    cfg = OmegaConf.create({
        "training": {
            "scheduler": {"type": "none"},
            "early_stopping": {"patience": 15},            # no 'monitor' key
            "checkpointing": None,
        },
        "evaluation": {"metrics": ["accuracy"]},
    })
    OmegaConf.set_struct(cfg, True)
    validate_monitor_keys(cfg)  # must not raise


def test_validate_monitor_keys_invalid_monitor_still_raises():
    """A monitor referencing a metric that won't be produced must still error."""
    cfg = OmegaConf.create({
        "training": {
            "scheduler": {"type": "plateau", "monitor": "val_bogus"},
            "early_stopping": None,
            "checkpointing": None,
        },
        "evaluation": {"metrics": ["accuracy"]},
    })
    OmegaConf.set_struct(cfg, True)
    try:
        validate_monitor_keys(cfg)
    except ValueError as e:
        assert "val_bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for an invalid monitor key")


def test_row_prediction_yaml_passes_monitor_validation():
    """The shipped row_prediction.yaml must validate cleanly (e2e regression)."""
    cfg = load_config(
        path="configs/downstream/row_prediction.yaml", task_name="row_smoke"
    )
    validate_monitor_keys(cfg)  # must not raise
