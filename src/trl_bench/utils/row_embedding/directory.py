"""
Shared utilities for directory-mode row embedding generation.

Provides common infrastructure for processing a directory of CSV files
into an aggregate pickle of row embeddings. Used by all 6 row embedding
model scripts (TabPFN, TabICL, SCARF, SubTab, VIME, DAE).
"""

import os
import pickle
import signal
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def discover_csv_files(input_dir: str, table_list_path: Optional[str] = None) -> List[Path]:
    """
    Find all .csv files in a directory (non-recursive, sorted by name).

    Args:
        input_dir: Directory to scan for CSV files.
        table_list_path: Optional path to a table list file for shard filtering.
            When provided, only CSV files whose basename appears in the list
            are returned.

    Returns:
        Sorted list of Path objects for each CSV file found.
    """
    input_path = Path(input_dir)
    results = []
    with os.scandir(input_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".csv"):
                results.append(Path(entry.path))
    results = sorted(results)

    if table_list_path is not None:
        from trl_bench.utils.table_list import load_table_list, filter_csv_files
        table_list = load_table_list(table_list_path)
        results = filter_csv_files(results, table_list)

    return results


def check_checkpoint_complete(checkpoint_dir: str, model_name: str) -> bool:
    """
    Check whether a per-table training checkpoint is complete.

    A checkpoint is complete if:
    1. training_config.pkl exists and is loadable
    2. The expected checkpoint file ({model_name}_self_supervised.ckpt) exists

    Args:
        checkpoint_dir: Directory containing the checkpoint files.
        model_name: Lowercase model name (e.g., 'scarf', 'subtab').

    Returns:
        True if checkpoint is complete and usable.
    """
    ckpt_dir = Path(checkpoint_dir)
    config_path = ckpt_dir / "training_config.pkl"
    ckpt_path = ckpt_dir / f"{model_name}_self_supervised.ckpt"

    if not config_path.exists() or not ckpt_path.exists():
        return False

    # Verify the config pickle is loadable (not corrupted)
    try:
        with open(config_path, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def clean_partial_checkpoint(checkpoint_dir: str) -> None:
    """
    Remove an incomplete checkpoint directory.

    Called before retraining when a previous SLURM job was interrupted,
    leaving behind partial files (e.g., training_config.pkl without
    the corresponding .ckpt, or a corrupted pickle).

    Args:
        checkpoint_dir: Directory to clean up.
    """
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"  Cleaned partial checkpoint: {ckpt_dir}")


def build_table_result(
    table_path: str,
    row_embeddings: np.ndarray,
    column_names: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """
    Build one entry for the aggregate pickle output.

    Args:
        table_path: Absolute path to the source CSV file.
        row_embeddings: Row embeddings array of shape (n_rows, embedding_dim).
        column_names: List of column header names from the CSV.
        model_name: Name of the embedding model.

    Returns:
        Dict matching the aggregate pickle schema.
    """
    table_id = Path(table_path).stem
    # Strip .csv extension if present in stem (e.g., from .csv.gz)
    if table_id.endswith(".csv"):
        table_id = table_id[:-4]

    return {
        "table": str(table_path),
        "table_id": table_id,
        "row_embeddings": row_embeddings,
        "column_names": column_names,
        "model_name": model_name,
        "embedding_dim": row_embeddings.shape[1] if row_embeddings.ndim > 1 else 0,
        "num_rows": row_embeddings.shape[0],
    }


def save_aggregate_pickle(
    results: List[Dict[str, Any]],
    output_path: str,
    protocol: int = 4,
) -> None:
    """
    Atomically save aggregate results to a pickle file.

    Writes to a temporary file first, then renames to the final path.
    This prevents corrupt output if a SLURM job is interrupted mid-write.

    Args:
        results: List of table result dicts.
        output_path: Final output pickle path.
        protocol: Pickle protocol version.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out.with_suffix(".pkl.tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(results, f, protocol=protocol)

    os.rename(tmp_path, out)


def register_save_on_signal(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Install a SIGTERM handler that flushes results to disk before exiting.

    SLURM sends SIGTERM before SIGKILL when a job hits its time limit or
    is cancelled. This handler saves the in-memory results list during
    the grace period, closing the gap where up to checkpoint_interval - 1
    tables would otherwise be lost.

    Since ``results`` is a mutable list, the handler closure always sees
    the latest state.

    Args:
        results: The shared, mutable results list being built up in main().
        output_path: Path to the aggregate pickle file.
    """
    def _handler(signum, frame):
        if results:
            print(f"\nSignal {signum} received — saving {len(results)} tables to {output_path}...")
            save_aggregate_pickle(results, output_path)
            print("Emergency save complete.")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)


def load_existing_results(output_path: str) -> List[Dict[str, Any]]:
    """
    Load previously saved results for resume support.

    If the output file exists and is loadable, returns its contents.
    Otherwise returns an empty list. Also cleans up any stale .tmp files.

    Args:
        output_path: Path to the aggregate pickle.

    Returns:
        List of table result dicts, or empty list.
    """
    out = Path(output_path)

    # Clean up stale tmp file from a previous interrupted write
    tmp_path = out.with_suffix(".pkl.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    if not out.exists():
        return []

    try:
        with open(out, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def get_completed_table_ids(results: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract the set of table_ids already in the results.

    Used to skip tables that were already processed in a previous run.

    Args:
        results: List of table result dicts.

    Returns:
        Set of table_id strings.
    """
    return {r["table_id"] for r in results if "table_id" in r}


def cleanup_checkpoints(
    checkpoint_base_dir: str,
    table_ids: Optional[List[str]] = None,
) -> None:
    """
    Clean up per-table checkpoint directories after successful embedding.

    When ``table_ids`` is provided, only the subdirectories for those
    tables are removed.  This is the shard-safe mode: each shard cleans
    up only its own tables, avoiding races with concurrent shards that
    share the same ``checkpoint_base_dir``.

    When ``table_ids`` is ``None``, the entire directory tree is removed
    (legacy non-sharded behavior).

    Args:
        checkpoint_base_dir: Root checkpoint directory
            (e.g., assets/checkpoints/row/scarf/sato/).
        table_ids: If given, only remove subdirectories matching these
            table IDs.  If None, remove the entire tree.
    """
    ckpt_dir = Path(checkpoint_base_dir)
    if not ckpt_dir.exists():
        return

    if table_ids is not None:
        cleaned = 0
        for table_id in table_ids:
            table_ckpt = ckpt_dir / table_id
            if table_ckpt.exists():
                shutil.rmtree(table_ckpt)
                cleaned += 1
        if cleaned:
            print(f"Cleaned up {cleaned} table checkpoints from {ckpt_dir}")
    else:
        n_entries = sum(1 for _ in ckpt_dir.iterdir())
        shutil.rmtree(ckpt_dir)
        print(f"Cleaned up {n_entries} table checkpoints from {ckpt_dir}")


def preprocess_table(
    df: pd.DataFrame,
    label_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], MinMaxScaler, List[str], List[str]]:
    """
    Preprocess a single table for ts3l model training.

    Detects categorical/continuous columns, fits LabelEncoder per categorical
    column and MinMaxScaler on continuous columns. Optionally strips label columns.

    Args:
        df: Raw DataFrame from CSV.
        label_columns: If provided, these columns are dropped from features.

    Returns:
        Tuple of:
        - X_encoded: Preprocessed feature DataFrame
        - categorical_encoders: Dict mapping column name to fitted LabelEncoder
        - scaler: Fitted MinMaxScaler for continuous columns
        - category_cols: List of categorical column names
        - continuous_cols: List of continuous column names
    """
    X = df.copy()
    if label_columns:
        cols_to_drop = [c for c in label_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

    category_cols = X.select_dtypes(include=["object"]).columns.tolist()
    continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Fit and apply categorical encoders
    categorical_encoders = {}
    for col in category_cols:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        categorical_encoders[col] = le
        X[col] = le.transform(X[col].astype(str))

    # Fit and apply scaler on continuous columns
    # Replace inf with NaN first, then fill with column median (robust to outliers),
    # fallback to 0.  Some web tables contain inf from division-by-zero in source data.
    scaler = MinMaxScaler()
    if continuous_cols:
        X[continuous_cols] = X[continuous_cols].replace([np.inf, -np.inf], np.nan)
        medians = X[continuous_cols].median()
        X[continuous_cols] = X[continuous_cols].fillna(medians).fillna(0)
        scaler.fit(X[continuous_cols])
        X[continuous_cols] = scaler.transform(X[continuous_cols])

    return X, categorical_encoders, scaler, category_cols, continuous_cols


# ─── Raw PyTorch training utilities ──────────────────────────────────

def _batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move a batch (tensor or tuple/list of tensors) to a device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(
            t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
        )
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    return batch


def train_raw_loop(
    pl_model: torch.nn.Module,
    dataloader: Any,
    max_epochs: int,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> None:
    """
    Train a ts3l model using a raw PyTorch loop.

    Replaces PL Trainer for per-table training where tables are small
    and the PL overhead (~25s/table) dominates actual computation.

    Uses the PL module's ``_get_first_phase_loss(batch)`` for exact
    loss computation — same math as PL Trainer, no approximation.

    A DataLoader is used (rather than a pre-collated batch) so that
    noise-based models (DAE, SCARF, VIME) get fresh corruption each
    epoch, matching PL Trainer behavior.

    No validation split, no early stopping. Epoch count is the knob
    for quality vs speed.

    Args:
        pl_model: A ts3l LightningModule with ``set_first_phase()`` called.
        dataloader: DataLoader yielding batches for ``_get_first_phase_loss``.
            For small tables, use ``batch_size=len(dataset)`` for a single batch.
        max_epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        device: Device to train on. If None, uses CUDA if available, else CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model = pl_model.to(device)

    optimizer = torch.optim.Adam(pl_model.parameters(), lr=lr)
    for _ in range(max_epochs):
        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            pl_model.train()
            optimizer.zero_grad()
            loss = pl_model._get_first_phase_loss(batch)
            loss.backward()
            optimizer.step()


def save_model_checkpoint(
    pl_model: torch.nn.Module,
    config_dict: Dict[str, Any],
    checkpoint_dir: str,
    ckpt_filename: str,
) -> None:
    """
    Save model state_dict and training config for later embedding extraction.

    Uses plain ``torch.save(state_dict)`` instead of PL checkpoints,
    avoiding the PyTorch 2.6 ``weights_only=True`` compatibility issue.

    ``training_config.pkl`` is written last and serves as the completeness
    marker — ``check_checkpoint_complete()`` checks for it.

    Args:
        pl_model: Trained PL module.
        config_dict: Training metadata (encoders, scaler, model config, etc.).
        checkpoint_dir: Directory to save into.
        ckpt_filename: Filename for the state_dict checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
    torch.save(pl_model.model.state_dict(), ckpt_path)

    # Write config last (completeness marker)
    with open(os.path.join(checkpoint_dir, "training_config.pkl"), "wb") as f:
        pickle.dump(config_dict, f)


def load_model_from_checkpoint(
    lightning_cls: type,
    config: Any,
    checkpoint_dir: str,
    ckpt_filename: str,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Recreate a ts3l model from saved config + state_dict.

    Args:
        lightning_cls: The LightningModule class (e.g., DAELightning).
        config: The model Config object (e.g., DAEConfig).
        checkpoint_dir: Directory containing the checkpoint.
        ckpt_filename: Filename of the state_dict checkpoint.
        device: Device to load model onto.

    Returns:
        Loaded PL module in eval mode with second phase set.
    """
    pl_model = lightning_cls(config)
    ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pl_model.model.load_state_dict(state_dict)
    pl_model.eval()
    pl_model.set_second_phase()

    if device is not None:
        pl_model = pl_model.to(device)

    return pl_model
