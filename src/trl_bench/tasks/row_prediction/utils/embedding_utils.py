"""
Generic Embedding Utilities
Works with any embeddings - not specific to any model

Uses unified JSON metadata format (metadata.json).
"""

import numpy as np
import json
import os
from typing import Optional, Tuple, Dict, Any, List


def save_embeddings(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    task: str = 'auto',
) -> None:
    """
    Save embeddings and labels to disk (GENERIC function).

    Args:
        train_embeddings: Training embeddings array (N_train x D)
        train_labels: Training labels array (N_train,)
        test_embeddings: Test embeddings array (N_test x D)
        test_labels: Test labels array (N_test,)
        output_dir: Directory to save embeddings
        metadata: Optional metadata dict to save
        verbose: Print progress messages
        task: Task type ('auto', 'classification', or 'regression')

    Output:
        - output_dir/train_embeddings.npy
        - output_dir/train_labels.npy
        - output_dir/test_embeddings.npy
        - output_dir/test_labels.npy
        - output_dir/metadata.json
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\nSaving embeddings to {output_dir}/...")

    # Save embeddings
    np.save(os.path.join(output_dir, "train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(output_dir, "test_embeddings.npy"), test_embeddings)
    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)

    # Create/update metadata
    if metadata is None:
        metadata = {}

    # Auto-detect task type if not specified
    if task == 'auto':
        unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
        is_float = train_labels.dtype in [np.float32, np.float64]
        num_unique = len(unique_labels)
        total_samples = len(train_labels) + len(test_labels)
        uniqueness_ratio = num_unique / total_samples

        if is_float and uniqueness_ratio > 0.1:
            detected_task = 'regression'
        elif num_unique > 20:
            detected_task = 'regression'
        else:
            detected_task = 'classification'
    else:
        detected_task = task

    # Common metadata
    metadata.update({
        'embedding_dim': train_embeddings.shape[1],
        'train_samples': len(train_embeddings),
        'test_samples': len(test_embeddings),
        'task': detected_task
    })

    # Task-specific metadata
    if detected_task == 'classification':
        metadata.update({
            'num_classes': len(np.unique(train_labels)),
            'label_classes': sorted(np.unique(np.concatenate([train_labels, test_labels])).tolist())
        })
    else:  # regression
        all_labels = np.concatenate([train_labels, test_labels])
        metadata.update({
            'label_min': float(all_labels.min()),
            'label_max': float(all_labels.max()),
            'label_mean': float(all_labels.mean()),
            'label_std': float(all_labels.std())
        })

    # Save metadata as JSON
    metadata['version'] = metadata.get('version', '1.0')
    metadata['format'] = metadata.get('format', 'unified_row_embedding')
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"   Saved train_embeddings.npy: {train_embeddings.shape}")
        print(f"   Saved train_labels.npy: {train_labels.shape}")
        print(f"   Saved test_embeddings.npy: {test_embeddings.shape}")
        print(f"   Saved test_labels.npy: {test_labels.shape}")
        print(f"   Saved metadata.json")


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


def load_embeddings(
    embedding_dir: str,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray],
           np.ndarray, Optional[np.ndarray],
           Dict[str, Any]]:
    """
    Load embeddings and labels from disk (GENERIC function).

    Loads metadata from metadata.json (unified format).
    Auto-detects v2.0 split-aware format and extracts train/val/test splits.

    Args:
        embedding_dir: Directory where embeddings were saved
        verbose: Print progress messages

    Returns:
        Tuple of (train_embeddings, train_labels,
                  val_embeddings, val_labels,
                  test_embeddings, test_labels,
                  metadata).
        val_embeddings and val_labels are None when no val split exists.
    """
    if verbose:
        print(f"\nLoading embeddings from {embedding_dir}/...")

    # Check for v2.0 split-aware format
    metadata_json = os.path.join(embedding_dir, "metadata.json")
    metadata = {}
    metadata_format = None

    if os.path.exists(metadata_json):
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)

        version = metadata.get('version', '1.0')
        splits_map = metadata.get('splits', {})

        if _version_gte(version, '2.0') and splits_map:
            # V2 format: load from splits map
            metadata_format = 'unified_v2'

            extra_splits = [s for s in splits_map if s not in ('train', 'val', 'test')]
            if extra_splits and verbose:
                print(f"   Note: Extra splits found but not returned: {extra_splits}")

            train_embeddings = _load_split_array(embedding_dir, splits_map, 'train', 'embeddings_file')
            test_embeddings = _load_split_array(embedding_dir, splits_map, 'test', 'embeddings_file')
            train_labels = _load_split_array(embedding_dir, splits_map, 'train', 'labels_file')
            test_labels = _load_split_array(embedding_dir, splits_map, 'test', 'labels_file')

            # Load val split (None when absent — v1 data, "all"-only, etc.)
            val_embeddings = _load_split_array(embedding_dir, splits_map, 'val', 'embeddings_file')
            val_labels = _load_split_array(embedding_dir, splits_map, 'val', 'labels_file')

            # Fallback: if only "all" split exists, use it for both train and test
            if train_embeddings is None and test_embeddings is None and 'all' in splits_map:
                if verbose:
                    print(f"   Warning: Only 'all' split found, using it as both train and test")
                all_emb = _load_split_array(embedding_dir, splits_map, 'all', 'embeddings_file')
                all_lbl = _load_split_array(embedding_dir, splits_map, 'all', 'labels_file')
                if all_emb is not None:
                    train_embeddings = all_emb
                    test_embeddings = all_emb
                    train_labels = all_lbl
                    test_labels = all_lbl

            available = sorted(splits_map.keys())
            if train_embeddings is None:
                raise FileNotFoundError(
                    f"No 'train' (or 'all') split found in v2.0 metadata at {embedding_dir}. "
                    f"Available splits: {available}. "
                    f"For arbitrary split names, use load_split_embeddings() from "
                    f"utils.unified_embedding_format instead."
                )
            if test_embeddings is None:
                raise FileNotFoundError(
                    f"No 'test' (or 'all') split found in v2.0 metadata at {embedding_dir}. "
                    f"Available splits: {available}. "
                    f"For arbitrary split names, use load_split_embeddings() from "
                    f"utils.unified_embedding_format instead."
                )

            # Ensure essential fields exist
            if 'embedding_dim' not in metadata:
                metadata['embedding_dim'] = train_embeddings.shape[1]
            metadata['train_samples'] = len(train_embeddings)
            metadata['test_samples'] = len(test_embeddings)
            if val_embeddings is not None:
                metadata['val_samples'] = len(val_embeddings)

            if verbose:
                print(f"   Loaded train_embeddings: {train_embeddings.shape}")
                if val_embeddings is not None:
                    print(f"   Loaded val_embeddings: {val_embeddings.shape}")
                print(f"   Loaded test_embeddings: {test_embeddings.shape}")
                print(f"   Embedding dimension: {metadata['embedding_dim']}")
                print(f"   Metadata format: {metadata_format}")
                _print_label_info(train_labels, test_labels, metadata, verbose)

            return (train_embeddings, train_labels,
                    val_embeddings, val_labels,
                    test_embeddings, test_labels,
                    metadata)

        metadata_format = 'unified'
    else:
        metadata_format = 'inferred'

    # V1 / legacy: load fixed filenames (no val split)
    train_embeddings = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    test_embeddings = np.load(os.path.join(embedding_dir, "test_embeddings.npy"))
    val_embeddings = None
    val_labels = None

    train_labels_path = os.path.join(embedding_dir, "train_labels.npy")
    test_labels_path = os.path.join(embedding_dir, "test_labels.npy")

    train_labels = np.load(train_labels_path) if os.path.exists(train_labels_path) else None
    test_labels = np.load(test_labels_path) if os.path.exists(test_labels_path) else None

    if not metadata:
        if verbose:
            print(f"   Warning: No metadata.json found, inferring from arrays")
        metadata = {
            'embedding_dim': train_embeddings.shape[1],
            'train_samples': len(train_embeddings),
            'test_samples': len(test_embeddings),
        }

    # Ensure essential fields exist
    if 'embedding_dim' not in metadata:
        metadata['embedding_dim'] = train_embeddings.shape[1]
    if 'train_samples' not in metadata:
        metadata['train_samples'] = len(train_embeddings)
    if 'test_samples' not in metadata:
        metadata['test_samples'] = len(test_embeddings)

    if verbose:
        print(f"   Loaded train_embeddings: {train_embeddings.shape}")
        print(f"   Loaded test_embeddings: {test_embeddings.shape}")
        print(f"   Embedding dimension: {metadata['embedding_dim']}")
        print(f"   Metadata format: {metadata_format}")
        _print_label_info(train_labels, test_labels, metadata, verbose)

    return (train_embeddings, train_labels,
            val_embeddings, val_labels,
            test_embeddings, test_labels,
            metadata)


def _load_split_array(
    embedding_dir: str,
    splits_map: Dict[str, Any],
    split_name: str,
    file_key: str,
) -> Optional[np.ndarray]:
    """Load a single .npy array from the v2 splits map."""
    if split_name not in splits_map:
        return None
    filename = splits_map[split_name].get(file_key)
    if not filename:
        return None
    filepath = os.path.join(embedding_dir, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    return None


def _print_label_info(
    train_labels: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    metadata: Dict[str, Any],
    verbose: bool,
) -> None:
    """Print task type and label info."""
    if not verbose:
        return
    task = metadata.get('task', 'classification')
    if task == 'classification':
        print(f"   Task: Classification")
        print(f"   Number of classes: {metadata.get('num_classes', 'N/A')}")
    else:
        print(f"   Task: Regression")
        label_min = metadata.get('label_min', 'N/A')
        label_max = metadata.get('label_max', 'N/A')
        if label_min != 'N/A' and label_max != 'N/A':
            print(f"   Label range: [{label_min:.4f}, {label_max:.4f}]")
    if train_labels is not None:
        print(f"   Train labels loaded: {train_labels.shape}")
    else:
        print(f"   Train labels: Not found")
    if test_labels is not None:
        print(f"   Test labels loaded: {test_labels.shape}")
    else:
        print(f"   Test labels: Not found")


def verify_embeddings(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray
) -> bool:
    """
    Verify embeddings have correct shapes and types.

    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels
        test_embeddings: Test embeddings
        test_labels: Test labels

    Returns:
        True if all checks pass

    Raises:
        ValueError: If verification fails
    """
    # Check dimensions
    if train_embeddings.ndim != 2:
        raise ValueError(f"train_embeddings must be 2D, got shape {train_embeddings.shape}")
    if test_embeddings.ndim != 2:
        raise ValueError(f"test_embeddings must be 2D, got shape {test_embeddings.shape}")
    if train_labels is not None and train_labels.ndim != 1:
        raise ValueError(f"train_labels must be 1D, got shape {train_labels.shape}")
    if test_labels is not None and test_labels.ndim != 1:
        raise ValueError(f"test_labels must be 1D, got shape {test_labels.shape}")

    # Check embedding dimensions match
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError(f"Embedding dimensions don't match: train={train_embeddings.shape[1]}, test={test_embeddings.shape[1]}")

    # Check sample counts match (only if labels are provided)
    if train_labels is not None:
        if train_embeddings.shape[0] != train_labels.shape[0]:
            raise ValueError(f"Train sample counts don't match: embeddings={train_embeddings.shape[0]}, labels={train_labels.shape[0]}")
    if test_labels is not None:
        if test_embeddings.shape[0] != test_labels.shape[0]:
            raise ValueError(f"Test sample counts don't match: embeddings={test_embeddings.shape[0]}, labels={test_labels.shape[0]}")

    return True


def get_metadata_info(embedding_dir: str) -> Dict[str, Any]:
    """
    Quick function to read just the metadata without loading embeddings.

    Args:
        embedding_dir: Directory containing embeddings

    Returns:
        Metadata dictionary
    """
    metadata_json = os.path.join(embedding_dir, "metadata.json")

    if os.path.exists(metadata_json):
        with open(metadata_json, 'r') as f:
            return json.load(f)
    else:
        return {}


def get_available_labels(
    embedding_dir: str,
) -> Tuple[List[str], Dict[str, str]]:
    """Return (label_columns, label_task_types) from metadata.

    Handles all metadata vintages:
    - New multi-label: ``label_columns`` (list) + ``label_task_types`` (dict)
    - Legacy v2 single: ``label_columns`` with one entry
    - Legacy v1: ``label_column`` (singular string)

    Returns:
        Tuple of (label_columns list, label_task_types dict).
    """
    metadata = get_metadata_info(embedding_dir)

    # New format: label_columns is a list
    label_columns = metadata.get('label_columns')
    if label_columns and isinstance(label_columns, list):
        label_task_types = metadata.get('label_task_types', {})
        return label_columns, label_task_types

    # Legacy singular key
    label_col = metadata.get('label_column')
    if label_col:
        return [label_col], {}

    # V1 fallback: check has_labels flag
    if metadata.get('has_labels') or metadata.get('has_label'):
        lc = metadata.get('label_column')
        if lc:
            return [lc], {}

    return [], {}


def load_embeddings_for_label(
    embedding_dir: str,
    label_column: str,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray],
           np.ndarray, Optional[np.ndarray],
           Dict[str, Any]]:
    """Load embeddings with labels for a specific label column.

    Returns a 7-tuple matching ``load_embeddings()`` but loads the
    specified label column's arrays from the ``labels_files`` dict in
    the v2 split-aware metadata.

    Graceful fallback for single-label / v1 dirs: if ``labels_files``
    is absent but ``labels_file`` exists and the requested column matches
    the first (or only) label column, the single file is loaded instead.

    Args:
        embedding_dir: Directory containing embeddings.
        label_column: Name of the label column to load.
        verbose: Print progress messages.

    Returns:
        Tuple of (train_embeddings, train_labels,
                  val_embeddings, val_labels,
                  test_embeddings, test_labels,
                  metadata).
        val_embeddings and val_labels are None when no val split exists.

    Raises:
        ValueError: If the requested label_column is not found.
    """
    metadata = get_metadata_info(embedding_dir)
    version = metadata.get('version', '1.0')
    splits_map = metadata.get('splits', {})

    # Determine the first label column from metadata (for fallback matching)
    meta_label_columns = metadata.get('label_columns', [])
    if not meta_label_columns:
        lc = metadata.get('label_column')
        if lc:
            meta_label_columns = [lc]

    def _load_label_for_split(split_name: str) -> Optional[np.ndarray]:
        """Try to load label array for a specific split and column."""
        if split_name not in splits_map:
            return None

        split_info = splits_map[split_name]

        # New multi-label: labels_files dict
        labels_files = split_info.get('labels_files')
        if labels_files and label_column in labels_files:
            fpath = os.path.join(embedding_dir, labels_files[label_column])
            if os.path.exists(fpath):
                return np.load(fpath)

        # Fallback: single labels_file if requested column is the first label
        labels_file = split_info.get('labels_file')
        if labels_file and meta_label_columns and label_column == meta_label_columns[0]:
            fpath = os.path.join(embedding_dir, labels_file)
            if os.path.exists(fpath):
                return np.load(fpath)

        return None

    if _version_gte(version, '2.0') and splits_map:
        # V2 format
        train_embeddings = _load_split_array(embedding_dir, splits_map, 'train', 'embeddings_file')
        test_embeddings = _load_split_array(embedding_dir, splits_map, 'test', 'embeddings_file')
        val_embeddings = _load_split_array(embedding_dir, splits_map, 'val', 'embeddings_file')

        # Fallback: if only "all" split exists, use it for both train and test
        if train_embeddings is None and test_embeddings is None and 'all' in splits_map:
            if verbose:
                print(f"   Warning: Only 'all' split found, using it as both train and test")
            all_emb = _load_split_array(embedding_dir, splits_map, 'all', 'embeddings_file')
            if all_emb is not None:
                train_embeddings = all_emb
                test_embeddings = all_emb

        if train_embeddings is None or test_embeddings is None:
            available = sorted(splits_map.keys())
            raise FileNotFoundError(
                f"No 'train' or 'test' (or 'all') split found in v2.0 metadata at {embedding_dir}. "
                f"Available splits: {available}."
            )

        train_labels = _load_label_for_split('train')
        test_labels = _load_label_for_split('test')
        val_labels = _load_label_for_split('val')

        # Fallback: load labels from "all" split if train/test labels missing
        if train_labels is None and test_labels is None and 'all' in splits_map:
            all_labels = _load_label_for_split('all')
            if all_labels is not None:
                train_labels = all_labels
                test_labels = all_labels

    else:
        # V1 fallback (no val split)
        train_embeddings = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
        test_embeddings = np.load(os.path.join(embedding_dir, "test_embeddings.npy"))
        val_embeddings = None
        val_labels = None

        # For v1, only one label file exists per split
        if meta_label_columns and label_column == meta_label_columns[0]:
            train_path = os.path.join(embedding_dir, "train_labels.npy")
            test_path = os.path.join(embedding_dir, "test_labels.npy")
            train_labels = np.load(train_path) if os.path.exists(train_path) else None
            test_labels = np.load(test_path) if os.path.exists(test_path) else None
        else:
            train_labels = None
            test_labels = None

    if train_labels is None and test_labels is None:
        raise ValueError(
            f"Label column '{label_column}' not found in embedding directory {embedding_dir}. "
            f"Available label columns: {meta_label_columns}"
        )

    # Ensure metadata has essential fields
    if 'embedding_dim' not in metadata:
        metadata['embedding_dim'] = train_embeddings.shape[1]
    metadata['train_samples'] = len(train_embeddings)
    metadata['test_samples'] = len(test_embeddings)
    if val_embeddings is not None:
        metadata['val_samples'] = len(val_embeddings)

    if verbose:
        print(f"\nLoaded embeddings for label '{label_column}' from {embedding_dir}/")
        print(f"   Train: {train_embeddings.shape}, Test: {test_embeddings.shape}")
        if val_embeddings is not None:
            print(f"   Val: {val_embeddings.shape}")
        if train_labels is not None:
            print(f"   Train labels: {train_labels.shape}")
        if val_labels is not None:
            print(f"   Val labels: {val_labels.shape}")
        if test_labels is not None:
            print(f"   Test labels: {test_labels.shape}")

    return (train_embeddings, train_labels,
            val_embeddings, val_labels,
            test_embeddings, test_labels,
            metadata)
