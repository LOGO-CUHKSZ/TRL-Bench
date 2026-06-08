"""
Dataset resolver: loads tabular datasets from multiple layout conventions.

Resolution precedence:
1. Canonical: <dir>/dataset.json  (data.csv + splits.npz + SHA256 fingerprint)
2. Legacy:    <dir>/train.csv     (+ optional val.csv, test.csv)
3. Single:    path ends with .csv (or dir has data.csv but no dataset.json)
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .core import TableDataset

logger = logging.getLogger(__name__)


def resolve_label_columns_cli(
    label_column: Optional[str],
    label_policy: str = "auto",
) -> Optional[List[str]]:
    """Resolve ``label_columns_cli`` from CLI label flags.

    Policies:
    - ``auto``: Backward compatible behavior. Use CLI label if provided,
      otherwise force unlabeled mode.
    - ``none``: Force unlabeled mode (no labels).
    - ``manifest``: Use labels from canonical ``dataset.json``.
    - ``cli``: Require ``--label_column`` and use it.

    Returns:
        Value for ``label_columns_cli`` passed into ``load_table_dataset``.
    """
    policy = (label_policy or "auto").strip().lower()
    if policy == "auto":
        return [label_column] if label_column else []
    if policy == "none":
        if label_column is not None:
            raise ValueError("--label_policy=none cannot be used with --label_column")
        return []
    if policy == "manifest":
        if label_column is not None:
            raise ValueError("--label_policy=manifest cannot be used with --label_column")
        return None
    if policy == "cli":
        if label_column is None:
            raise ValueError("--label_policy=cli requires --label_column")
        return [label_column]
    raise ValueError(
        f"Invalid label_policy '{label_policy}'. "
        "Expected one of: auto, none, manifest, cli."
    )


def load_table_dataset(
    path: str,
    label_columns_cli: Optional[List[str]] = None,
    ignore_fingerprint: bool = False,
) -> TableDataset:
    """Load a TableDataset from disk, auto-detecting the layout.

    Args:
        path: Path to a directory or CSV file.
        label_columns_cli: Label column names from CLI. Overrides manifest
            ``label_columns`` for canonical layout (with a warning).
        ignore_fingerprint: If True, skip SHA256 verification for canonical layout.

    Returns:
        Populated TableDataset instance.

    Raises:
        FileNotFoundError: If path doesn't exist or required files are missing.
        ValueError: If SHA256 fingerprint doesn't match (canonical, unless ignored).
    """
    path = str(path)

    # Determine layout
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "dataset.json")
        train_csv_path = os.path.join(path, "train.csv")
        data_csv_path = os.path.join(path, "data.csv")

        if os.path.exists(manifest_path):
            return _load_canonical(path, manifest_path, label_columns_cli, ignore_fingerprint)
        elif os.path.exists(train_csv_path):
            return _load_legacy(path, label_columns_cli)
        elif os.path.exists(data_csv_path):
            # data.csv without dataset.json → treat as single CSV
            return _load_single(data_csv_path, label_columns_cli)
        else:
            raise FileNotFoundError(
                f"Cannot resolve dataset at '{path}': no dataset.json, "
                f"train.csv, or data.csv found."
            )
    elif path.lower().endswith(".csv"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        return _load_single(path, label_columns_cli)
    else:
        raise FileNotFoundError(
            f"Cannot resolve dataset: '{path}' is neither a directory nor a .csv file."
        )


# ---------------------------------------------------------------------------
# Canonical layout
# ---------------------------------------------------------------------------

def _load_canonical(
    dir_path: str,
    manifest_path: str,
    label_columns_cli: Optional[List[str]],
    ignore_fingerprint: bool,
) -> TableDataset:
    """Load canonical layout: dataset.json + data.csv + splits.npz."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Support both nested format (data.file, splits.file, data.fingerprint.sha256)
    # and flat format (data_file, splits_file, fingerprint_sha256)
    data_section = manifest.get("data", {})
    splits_section = manifest.get("splits", {})

    data_file = data_section.get("file") or manifest.get("data_file", "data.csv")
    splits_file = splits_section.get("file") or manifest.get("splits_file", "splits.npz")
    manifest_labels = manifest.get("label_columns", [])
    expected_sha = (
        data_section.get("fingerprint", {}).get("sha256")
        or manifest.get("fingerprint_sha256", "")
    )

    data_csv_path = os.path.join(dir_path, data_file)
    splits_npz_path = os.path.join(dir_path, splits_file)

    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Data file not found: {data_csv_path}")
    if not os.path.exists(splits_npz_path):
        raise FileNotFoundError(f"Splits file not found: {splits_npz_path}")

    # SHA256 verification
    if expected_sha and not ignore_fingerprint:
        actual_sha = _compute_sha256(data_csv_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA256 mismatch for {data_csv_path}:\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual_sha}\n"
                f"Use ignore_fingerprint=True to skip this check."
            )
        logger.info("SHA256 fingerprint verified for %s", data_csv_path)
    elif expected_sha and ignore_fingerprint:
        logger.warning("Skipping SHA256 verification for %s (ignore_fingerprint=True)", data_csv_path)

    # Load data
    full_df = pd.read_csv(data_csv_path)

    # Load splits
    splits_data = np.load(splits_npz_path)
    split_indices: Dict[str, np.ndarray] = {
        name: splits_data[name].astype(np.int64)
        for name in splits_data.files
    }

    # Resolve label columns
    label_columns = _resolve_labels(manifest_labels, label_columns_cli, "canonical")

    # Extract per-label task_type from manifest "labels" array
    label_task_types: Dict[str, str] = {}
    for entry in manifest.get("labels", []):
        name = entry.get("name", "")
        task_type = entry.get("task_type", "")
        if name and task_type:
            label_task_types[name] = task_type

    if ignore_fingerprint:
        # Don't store unverified manifest fingerprint; compute from actual file
        fingerprint = _compute_sha256(data_csv_path)
    else:
        fingerprint = expected_sha or _compute_sha256(data_csv_path)

    return TableDataset(
        full_df=full_df,
        label_columns=label_columns,
        split_indices=split_indices,
        layout="canonical",
        source_path=dir_path,
        fingerprint=fingerprint,
        label_task_types=label_task_types,
    )


# ---------------------------------------------------------------------------
# Legacy layout
# ---------------------------------------------------------------------------

def _load_legacy(
    dir_path: str,
    label_columns_cli: Optional[List[str]],
) -> TableDataset:
    """Load legacy layout: train.csv (+ optional val.csv, test.csv)."""
    frames: List[pd.DataFrame] = []
    split_indices: Dict[str, np.ndarray] = {}
    offset = 0

    for split_name in ("train", "val", "test"):
        csv_path = os.path.join(dir_path, f"{split_name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            n = len(df)
            split_indices[split_name] = np.arange(offset, offset + n)
            frames.append(df)
            offset += n
            logger.info("Loaded %s: %d rows from %s", split_name, n, csv_path)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in legacy layout at '{dir_path}'")

    full_df = pd.concat(frames, ignore_index=True)
    label_columns = _resolve_labels([], label_columns_cli, "legacy")

    return TableDataset(
        full_df=full_df,
        label_columns=label_columns,
        split_indices=split_indices,
        layout="legacy",
        source_path=dir_path,
        fingerprint="",
    )


# ---------------------------------------------------------------------------
# Single CSV layout
# ---------------------------------------------------------------------------

def _load_single(
    csv_path: str,
    label_columns_cli: Optional[List[str]],
) -> TableDataset:
    """Load single-CSV layout: one file, one 'all' split."""
    full_df = pd.read_csv(csv_path)
    label_columns = _resolve_labels([], label_columns_cli, "single")

    return TableDataset(
        full_df=full_df,
        label_columns=label_columns,
        split_indices={"all": np.arange(len(full_df))},
        layout="single",
        source_path=csv_path,
        fingerprint="",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_labels(
    manifest_labels: List[str],
    cli_labels: Optional[List[str]],
    layout: str,
) -> List[str]:
    """Determine the effective label columns.

    CLI labels override manifest labels (with a warning).
    """
    if cli_labels is not None:
        if manifest_labels and cli_labels != manifest_labels:
            logger.warning(
                "CLI label_columns %s override manifest label_columns %s",
                cli_labels,
                manifest_labels,
            )
        return cli_labels
    return manifest_labels


def _compute_sha256(filepath: str) -> str:
    """Compute SHA256 hex digest of a file using streaming reads."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)  # 64KB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
