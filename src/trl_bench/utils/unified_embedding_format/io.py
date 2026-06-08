"""
I/O utilities for saving and loading embeddings in unified format.

Supports:
- Pickle format for column/table embeddings
- NumPy arrays + JSON metadata for row embeddings
"""

import os
import json
import pickle
from pathlib import Path
from typing import Union, Dict, Any, List, Optional

import numpy as np

import re

from .schema import (
    TableEmbeddingResult,
    TableLevelEmbedding,
    EmbeddingBatch,
    RowEmbeddingMetadata,
    RowEmbeddingMetadataV2,
    SplitInfo,
    UNIFIED_TABLE_VERSION,
    UNIFIED_ROW_VERSION_V2,
)


def _sanitize_label_name(name: str) -> str:
    """Sanitize a label column name for use in filenames.

    Replaces non-alphanumeric chars (except ``_`` and ``-``) with ``_``.
    """
    return re.sub(r'[^A-Za-z0-9_\-]', '_', str(name))


def save_embeddings(
    data: Union[TableEmbeddingResult, EmbeddingBatch, List[Dict], Dict],
    output_path: str,
    protocol: int = 4
) -> None:
    """
    Save table/column embeddings to pickle file.

    Accepts:
    - TableEmbeddingResult: Single table result
    - EmbeddingBatch: Batch of results
    - List[Dict]: Legacy list of dicts format
    - Dict: Legacy dict format

    Args:
        data: Embedding data to save
        output_path: Output pickle file path
        protocol: Pickle protocol version (default: 4)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass objects to dicts
    if isinstance(data, TableEmbeddingResult):
        to_save = data.to_dict()
    elif isinstance(data, EmbeddingBatch):
        to_save = data.to_dict()
    elif isinstance(data, list):
        # List of dicts - convert TableEmbeddingResult items if present
        to_save = []
        for item in data:
            if isinstance(item, TableEmbeddingResult):
                to_save.append(item.to_dict())
            elif isinstance(item, dict):
                to_save.append(item)
            else:
                raise TypeError(f"Unsupported item type in list: {type(item)}")
    elif isinstance(data, dict):
        to_save = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    with open(output_path, 'wb') as f:
        pickle.dump(to_save, f, protocol=protocol)


def load_embeddings(
    input_path: str,
    as_dataclass: bool = False
) -> Union[TableEmbeddingResult, EmbeddingBatch, List[Dict], Dict]:
    """
    Load table/column embeddings from pickle file.

    Automatically detects format:
    - Unified batch format (with 'format': 'unified_batch_embedding')
    - Unified table format (with 'format': 'unified_table_embedding')
    - Legacy list format (list of dicts with 'column_embedding' or 'column_embeddings')
    - Legacy dict format (dict with table_name keys)

    Args:
        input_path: Input pickle file path
        as_dataclass: If True, convert to TableEmbeddingResult/EmbeddingBatch
                     If False, return raw dict/list

    Returns:
        Loaded embedding data
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    if not as_dataclass:
        return data

    # Detect format and convert to dataclass
    if isinstance(data, dict):
        fmt = data.get('format', '')

        if fmt == 'unified_batch_embedding':
            return EmbeddingBatch.from_dict(data)
        elif fmt == 'unified_table_embedding':
            return TableEmbeddingResult.from_dict(data)
        else:
            # Legacy dict format - try to detect structure
            if 'column_embeddings' in data or 'column_embedding' in data:
                # Single table dict
                return _legacy_dict_to_result(data)
            else:
                # Dict with table_name keys
                return data

    elif isinstance(data, list):
        # List of dicts - convert to EmbeddingBatch
        results = []
        model_name = ''
        embedding_dim = 0

        for item in data:
            if isinstance(item, dict):
                result = _legacy_dict_to_result(item)
                if result:
                    results.append(result)
                    if not model_name:
                        model_name = result.model_name
                    if not embedding_dim:
                        embedding_dim = result.embedding_dim

        return EmbeddingBatch(
            model_name=model_name,
            embedding_dim=embedding_dim,
            results=results
        )

    return data


def _legacy_dict_to_result(data: Dict) -> Optional[TableEmbeddingResult]:
    """Convert legacy dict format to TableEmbeddingResult."""
    # Handle 'column_embedding' (singular) -> 'column_embeddings' (plural)
    col_emb = data.get('column_embeddings') or data.get('column_embedding') or {}

    if not col_emb:
        return None

    # Convert to proper format if needed
    column_embeddings = {}
    for k, v in col_emb.items():
        col_idx = int(k) if isinstance(k, str) else k
        if isinstance(v, np.ndarray):
            column_embeddings[col_idx] = v.astype(np.float32)
        elif isinstance(v, list):
            column_embeddings[col_idx] = np.array(v, dtype=np.float32)
        else:
            column_embeddings[col_idx] = v

    # Extract embedding dim
    embedding_dim = data.get('embedding_dim', 0)
    if not embedding_dim and column_embeddings:
        first_emb = next(iter(column_embeddings.values()))
        if isinstance(first_emb, np.ndarray):
            embedding_dim = first_emb.shape[0]
        elif isinstance(first_emb, list):
            embedding_dim = len(first_emb)

    # Get table_id from various possible keys
    table_id = data.get('table_id') or data.get('table_name') or data.get('table') or ''
    if isinstance(table_id, str):
        if '/' in table_id or '\\' in table_id:
            # Extract filename from path
            table_id = Path(table_id).stem
        # Handle double extensions like .csv.gz / .csv.bz2
        if table_id.endswith('.csv'):
            table_id = table_id[:-4]

    # Convert table-level embeddings (v1.0 and v2.0)
    table_embedding_obj: Optional[TableLevelEmbedding] = None
    table_embedding_data = data.get('table_embedding')

    if isinstance(table_embedding_data, dict):
        # v2.0 format: dict with cls_embedding, table_embedding, column_mean
        table_embedding_obj = TableLevelEmbedding.from_dict(table_embedding_data)
    elif table_embedding_data is not None:
        # v1.0 format: array/list (typically mean-pooled columns)
        table_embedding_obj = TableLevelEmbedding(
            column_mean=_to_numpy(table_embedding_data)
        )

    # Backward-compat fields at the top level
    top_level_cls = data.get('cls_embedding')
    top_level_column_mean = data.get('column_mean')
    if (
        table_embedding_obj is None
        and (top_level_cls is not None or top_level_column_mean is not None)
    ):
        table_embedding_obj = TableLevelEmbedding()

    if table_embedding_obj is not None:
        if table_embedding_obj.cls_embedding is None and top_level_cls is not None:
            table_embedding_obj.cls_embedding = _to_numpy(top_level_cls)
        if table_embedding_obj.column_mean is None and top_level_column_mean is not None:
            table_embedding_obj.column_mean = _to_numpy(top_level_column_mean)

        # Keep output schema clean: drop empty table_embedding blocks
        if not table_embedding_obj.has_any():
            table_embedding_obj = None

    return TableEmbeddingResult(
        table_id=table_id,
        model_name=data.get('model_name', ''),
        embedding_dim=int(embedding_dim) if embedding_dim else 0,
        column_embeddings=column_embeddings,
        table_embedding=table_embedding_obj,
        context_embedding=_to_numpy(data.get('context_embedding')),
        column_names=data.get('column_names', []),
        source_path=data.get('source_path') or data.get('table'),
        version=data.get('version', UNIFIED_TABLE_VERSION),
        format=data.get('format', 'unified_table_embedding'),
    )


def _to_numpy(data: Any) -> Optional[np.ndarray]:
    """Convert to numpy array if not None."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    if isinstance(data, list):
        return np.array(data, dtype=np.float32)
    return data


def encode_label_column(y_col, label_encoder=None, split_name='', col_name='', logger=None):
    """Encode a label column for saving, handling NaN correctly.

    For classification (label_encoder is not None):
        - Detects NaN positions and marks them with -1 sentinel
        - Encodes only non-NaN values using the LabelEncoder
        - Handles unseen labels by mapping to first known class

    For regression (label_encoder is None):
        - Converts to float64, preserving NaN naturally

    Args:
        y_col: pandas Series or array-like of label values.
        label_encoder: Fitted sklearn LabelEncoder for classification columns,
            or None for regression columns.
        split_name: Split name for log messages (e.g. 'train', 'test').
        col_name: Column name for log messages.
        logger: Optional logger instance.

    Returns:
        np.ndarray: Encoded labels. int64 with -1 for NaN (classification)
            or float64 with NaN preserved (regression).
    """
    import pandas as pd

    if label_encoder is None:
        # Regression: save raw continuous values (NaN preserved naturally in float64)
        if hasattr(y_col, 'values'):
            return y_col.values.astype(np.float64)
        return np.asarray(y_col, dtype=np.float64)

    # Classification: detect NaN before encoding
    nan_mask = pd.isna(y_col)
    if hasattr(nan_mask, 'values'):
        nan_mask = nan_mask.values  # convert to numpy for consistent indexing
    n_nan = int(nan_mask.sum())
    n_total = len(y_col)

    result = np.full(n_total, -1, dtype=np.int64)

    if n_nan < n_total:
        # Extract non-NaN values
        if hasattr(y_col, 'iloc'):
            valid_vals = y_col[~nan_mask]
        else:
            valid_vals = np.asarray(y_col)[~nan_mask]

        try:
            result[~nan_mask] = label_encoder.transform(valid_vals)
        except ValueError:
            # Unseen labels in this split — map to first known class
            known = set(label_encoder.classes_)
            if hasattr(valid_vals, 'where'):
                valid_mapped = valid_vals.where(valid_vals.isin(known), label_encoder.classes_[0])
            else:
                valid_mapped = np.array(valid_vals, copy=True)
                for v in set(valid_vals) - known:
                    valid_mapped[valid_mapped == v] = label_encoder.classes_[0]
            result[~nan_mask] = label_encoder.transform(valid_mapped)
            if logger:
                logger.warning(
                    "Some labels in split '%s' col '%s' were unseen, mapped to first class",
                    split_name, col_name,
                )

    if n_nan > 0 and logger:
        logger.info(
            "NaN labels in split '%s' col '%s': %d of %d (%.1f%%) encoded as -1",
            split_name, col_name, n_nan, n_total, 100.0 * n_nan / n_total,
        )

    return result


def save_row_embeddings(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    metadata: RowEmbeddingMetadata,
    output_dir: str,
    train_labels: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Save row-level embeddings with unified metadata format.

    Creates:
    - train_embeddings.npy
    - test_embeddings.npy
    - train_labels.npy (if provided)
    - test_labels.npy (if provided)
    - metadata.json

    Args:
        train_embeddings: Training set embeddings (num_samples, embedding_dim)
        test_embeddings: Test set embeddings (num_samples, embedding_dim)
        metadata: Embedding metadata
        output_dir: Output directory
        train_labels: Optional training labels
        test_labels: Optional test labels

    Returns:
        Dict mapping file types to their paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Save embeddings
    train_emb_path = output_path / 'train_embeddings.npy'
    test_emb_path = output_path / 'test_embeddings.npy'
    np.save(train_emb_path, train_embeddings)
    np.save(test_emb_path, test_embeddings)
    output_files['train_embeddings'] = str(train_emb_path)
    output_files['test_embeddings'] = str(test_emb_path)

    # Update metadata with actual sample counts
    metadata.train_samples = train_embeddings.shape[0]
    metadata.test_samples = test_embeddings.shape[0]
    if train_embeddings.ndim > 1:
        metadata.embedding_dim = train_embeddings.shape[1]

    # Save labels if provided
    if train_labels is not None:
        train_labels_path = output_path / 'train_labels.npy'
        np.save(train_labels_path, train_labels)
        metadata.has_labels = True
        output_files['train_labels'] = str(train_labels_path)

    if test_labels is not None:
        test_labels_path = output_path / 'test_labels.npy'
        np.save(test_labels_path, test_labels)
        output_files['test_labels'] = str(test_labels_path)

    # Save metadata as JSON
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    output_files['metadata'] = str(metadata_path)

    return output_files


def load_row_embeddings(
    input_dir: str
) -> Dict[str, Any]:
    """
    Load row-level embeddings from directory.

    Returns:
        Dict with keys:
        - 'train_embeddings': np.ndarray
        - 'test_embeddings': np.ndarray
        - 'train_labels': np.ndarray (if exists)
        - 'test_labels': np.ndarray (if exists)
        - 'metadata': RowEmbeddingMetadata
    """
    input_path = Path(input_dir)

    result = {}

    # Load embeddings
    train_path = input_path / 'train_embeddings.npy'
    test_path = input_path / 'test_embeddings.npy'

    if train_path.exists():
        result['train_embeddings'] = np.load(train_path)
    if test_path.exists():
        result['test_embeddings'] = np.load(test_path)

    # Load labels if they exist
    train_labels_path = input_path / 'train_labels.npy'
    test_labels_path = input_path / 'test_labels.npy'

    if train_labels_path.exists():
        result['train_labels'] = np.load(train_labels_path)
    if test_labels_path.exists():
        result['test_labels'] = np.load(test_labels_path)

    # Load metadata
    metadata_json = input_path / 'metadata.json'
    metadata_pkl = input_path / 'embedding_metadata.pkl'  # Legacy format

    if metadata_json.exists():
        with open(metadata_json, 'r') as f:
            metadata_dict = json.load(f)
        result['metadata'] = RowEmbeddingMetadata.from_dict(metadata_dict)
    elif metadata_pkl.exists():
        # Legacy pickle format
        with open(metadata_pkl, 'rb') as f:
            metadata_dict = pickle.load(f)
        result['metadata'] = RowEmbeddingMetadata.from_dict(metadata_dict)

    return result


def save_split_embeddings(
    embeddings: Dict[str, np.ndarray],
    metadata: RowEmbeddingMetadataV2,
    output_dir: str,
    labels: Optional[Dict[str, Any]] = None,
    row_indices: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, str]:
    """
    Save split-aware row-level embeddings with v2.0 metadata.

    Creates per-split .npy files and a metadata.json with the splits map.

    Args:
        embeddings: split_name -> (N, D) array of embeddings.
        metadata: V2 metadata object (splits field will be populated).
        output_dir: Output directory.
        labels: Optional split_name -> labels.  Each value may be either:
            - np.ndarray (single-label, backward compat)
            - dict mapping column_name -> np.ndarray (multi-label)
        row_indices: Optional split_name -> (N,) array of row indices.

    Returns:
        Dict mapping file descriptions to their paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = {}
    if row_indices is None:
        row_indices = {}

    output_files = {}
    splits_map: Dict[str, Dict[str, Any]] = {}

    # Build sanitized label name mapping (used for multi-label filenames)
    # Deduplicate sanitized names with numeric suffixes if needed
    label_filename_map: Dict[str, str] = {}
    if metadata.label_columns:
        seen_sanitized: Dict[str, int] = {}
        for col in metadata.label_columns:
            base = _sanitize_label_name(col)
            if base in seen_sanitized:
                seen_sanitized[base] += 1
                sanitized = f"{base}_{seen_sanitized[base]}"
            else:
                seen_sanitized[base] = 0
                sanitized = base
            label_filename_map[col] = sanitized
        metadata.label_filename_map = label_filename_map

    for split_name, emb_arr in embeddings.items():
        # Save embeddings
        emb_filename = f"{split_name}_embeddings.npy"
        emb_path = output_path / emb_filename
        np.save(emb_path, emb_arr)
        output_files[f"{split_name}_embeddings"] = str(emb_path)

        split_info: Dict[str, Any] = {
            "num_samples": emb_arr.shape[0],
            "embeddings_file": emb_filename,
        }

        # Update embedding_dim from first split
        if emb_arr.ndim > 1 and metadata.embedding_dim == 0:
            metadata.embedding_dim = emb_arr.shape[1]

        # Save labels if provided for this split
        if split_name in labels and labels[split_name] is not None:
            split_labels = labels[split_name]

            if isinstance(split_labels, dict):
                # Multi-label: save per-column .npy files
                labels_files: Dict[str, str] = {}
                for col_name, col_arr in split_labels.items():
                    sanitized = label_filename_map.get(col_name, _sanitize_label_name(col_name))
                    lbl_filename = f"{split_name}_labels_{sanitized}.npy"
                    lbl_path = output_path / lbl_filename
                    np.save(lbl_path, col_arr)
                    output_files[f"{split_name}_labels_{sanitized}"] = str(lbl_path)
                    labels_files[col_name] = lbl_filename
                split_info["labels_files"] = labels_files

                # Backward compat: set labels_file to first label column
                # (deterministic from metadata.label_columns order)
                if metadata.label_columns:
                    first_col = metadata.label_columns[0]
                    if first_col in labels_files:
                        split_info["labels_file"] = labels_files[first_col]
            else:
                # Single ndarray (backward compat)
                lbl_filename = f"{split_name}_labels.npy"
                lbl_path = output_path / lbl_filename
                np.save(lbl_path, split_labels)
                output_files[f"{split_name}_labels"] = str(lbl_path)
                split_info["labels_file"] = lbl_filename

        # Save row indices if provided for this split
        if split_name in row_indices and row_indices[split_name] is not None:
            idx_filename = f"{split_name}_row_indices.npy"
            idx_path = output_path / idx_filename
            np.save(idx_path, row_indices[split_name])
            output_files[f"{split_name}_row_indices"] = str(idx_path)
            split_info["row_indices_file"] = idx_filename

        splits_map[split_name] = split_info

    metadata.splits = splits_map

    # Write metadata.json
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    output_files["metadata"] = str(metadata_path)

    return output_files


def _version_gte(version_str: str, target: str) -> bool:
    """Numeric version comparison: '2.0' >= '2.0' is True, '2' >= '2.0' is True."""
    def _parse(v):
        try:
            return tuple(int(x) for x in v.split('.'))
        except (ValueError, AttributeError):
            return (0,)
    a, b = _parse(version_str), _parse(target)
    # Zero-pad shorter tuple so (2,) and (2,0) compare as equal
    maxlen = max(len(a), len(b))
    a = a + (0,) * (maxlen - len(a))
    b = b + (0,) * (maxlen - len(b))
    return a >= b


def load_split_embeddings(
    input_dir: str,
) -> Dict[str, Any]:
    """
    Load split-aware row-level embeddings from a directory.

    Auto-detects format version:
    - v2.0 with 'splits' key: loads per-split files from the splits map.
    - v1.0 / no splits key: falls back to load_row_embeddings() and wraps.

    Returns:
        Dict with keys:
        - 'embeddings': {split_name: np.ndarray}
        - 'labels': {split_name: np.ndarray}  (may be empty)
        - 'row_indices': {split_name: np.ndarray}  (may be empty)
        - 'metadata': dict (raw metadata from JSON)
    """
    input_path = Path(input_dir)
    metadata_json = input_path / "metadata.json"

    metadata = {}
    if metadata_json.exists():
        with open(metadata_json, "r") as f:
            metadata = json.load(f)

    version = metadata.get("version", "1.0")
    splits_map = metadata.get("splits", {})

    if _version_gte(version, "2.0") and splits_map:
        # V2 format: load per-split files
        emb_dict: Dict[str, np.ndarray] = {}
        lbl_dict: Dict[str, np.ndarray] = {}
        idx_dict: Dict[str, np.ndarray] = {}

        for split_name, split_info in splits_map.items():
            emb_file = split_info.get("embeddings_file", "")
            if emb_file:
                emb_path = input_path / emb_file
                if emb_path.exists():
                    emb_dict[split_name] = np.load(emb_path)

            lbl_files_dict = split_info.get("labels_files")
            lbl_file = split_info.get("labels_file")
            if lbl_files_dict:
                # Multi-label: load per-column arrays into a dict
                per_col: Dict[str, np.ndarray] = {}
                for col_name, col_filename in lbl_files_dict.items():
                    col_path = input_path / col_filename
                    if col_path.exists():
                        per_col[col_name] = np.load(col_path)
                lbl_dict[split_name] = per_col
            elif lbl_file:
                lbl_path = input_path / lbl_file
                if lbl_path.exists():
                    lbl_dict[split_name] = np.load(lbl_path)

            idx_file = split_info.get("row_indices_file")
            if idx_file:
                idx_path = input_path / idx_file
                if idx_path.exists():
                    idx_dict[split_name] = np.load(idx_path)

        return {
            "embeddings": emb_dict,
            "labels": lbl_dict,
            "row_indices": idx_dict,
            "metadata": metadata,
        }

    else:
        # V1 fallback: use existing load_row_embeddings and wrap
        v1_result = load_row_embeddings(input_dir)

        emb_dict = {}
        lbl_dict = {}
        if "train_embeddings" in v1_result:
            emb_dict["train"] = v1_result["train_embeddings"]
        if "test_embeddings" in v1_result:
            emb_dict["test"] = v1_result["test_embeddings"]
        if "train_labels" in v1_result:
            lbl_dict["train"] = v1_result["train_labels"]
        if "test_labels" in v1_result:
            lbl_dict["test"] = v1_result["test_labels"]

        return {
            "embeddings": emb_dict,
            "labels": lbl_dict,
            "row_indices": {},
            "metadata": metadata if metadata else (
                v1_result["metadata"].to_dict() if "metadata" in v1_result else {}
            ),
        }
