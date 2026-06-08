"""
Validation utilities for embedding results.

Provides functions to validate embedding format and detect format type.
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np

from .schema import (
    TableEmbeddingResult,
    EmbeddingBatch,
    UNIFIED_TABLE_VERSION,
    UNIFIED_BATCH_VERSION,
)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def is_unified_format(data: Any) -> Tuple[bool, str]:
    """
    Check if data is in unified format.

    Args:
        data: Data to check

    Returns:
        Tuple of (is_unified, format_type)
        format_type is one of: 'unified_table_embedding', 'unified_batch_embedding',
                               'legacy_list', 'legacy_dict', 'tuple_list', 'unknown'
    """
    if isinstance(data, (TableEmbeddingResult, EmbeddingBatch)):
        return True, data.format

    if isinstance(data, dict):
        fmt = data.get('format', '')
        if fmt in ('unified_table_embedding', 'unified_batch_embedding'):
            return True, fmt
        if 'column_embeddings' in data or 'column_embedding' in data:
            return False, 'legacy_dict'
        return False, 'unknown'

    if isinstance(data, list):
        if not data:
            return False, 'unknown'

        first = data[0]

        # Check for unified batch format in list wrapper
        if isinstance(first, dict):
            if first.get('format') == 'unified_table_embedding':
                return True, 'unified_list'

            # Check for column_embeddings (plural) - unified
            if 'column_embeddings' in first and 'version' in first:
                return True, 'unified_list'

            # Check for column_embedding (singular) - legacy
            if 'column_embedding' in first:
                return False, 'legacy_list'

            # Dict list without clear format marker
            if 'column_embeddings' in first:
                return False, 'legacy_list'

            return False, 'legacy_list'

        # Tuple format: [(table, col, emb), ...]
        if isinstance(first, tuple) and len(first) == 3:
            return False, 'tuple_list'

    return False, 'unknown'


def validate_embedding_result(
    data: Union[Dict, TableEmbeddingResult],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate a single table embedding result.

    Args:
        data: Embedding result to validate
        strict: If True, require all fields. If False, only check present fields.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Convert to dict if dataclass
    if isinstance(data, TableEmbeddingResult):
        data = data.to_dict()

    if not isinstance(data, dict):
        return False, [f"Expected dict or TableEmbeddingResult, got {type(data)}"]

    # Required fields in strict mode
    if strict:
        required = ['table_id', 'model_name', 'embedding_dim', 'column_embeddings']
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

    # Check column_embeddings format
    col_emb = data.get('column_embeddings') or data.get('column_embedding')
    if col_emb is not None:
        if not isinstance(col_emb, dict):
            errors.append(f"column_embeddings should be dict, got {type(col_emb)}")
        else:
            # Check each embedding
            embedding_dim = data.get('embedding_dim', 0)
            for k, v in col_emb.items():
                # Key should be int or int-like string
                try:
                    int(k)
                except (ValueError, TypeError):
                    errors.append(f"Invalid column index key: {k}")

                # Value should be array-like
                if isinstance(v, np.ndarray):
                    if v.ndim != 1:
                        errors.append(f"Column {k} embedding should be 1D, got shape {v.shape}")
                    elif embedding_dim and v.shape[0] != embedding_dim:
                        errors.append(f"Column {k} has dim {v.shape[0]}, expected {embedding_dim}")
                elif isinstance(v, list):
                    if embedding_dim and len(v) != embedding_dim:
                        errors.append(f"Column {k} has dim {len(v)}, expected {embedding_dim}")
                else:
                    errors.append(f"Column {k} embedding has invalid type: {type(v)}")

    # Check table_embedding field (v2.0 format: dict, v1.0 format: array)
    table_emb = data.get('table_embedding')
    if table_emb is not None:
        if isinstance(table_emb, dict):
            # v2.0 format: dict with cls_embedding, table_embedding, column_mean
            expected_keys = {'cls_embedding', 'table_embedding', 'column_mean', 'token_mean'}
            # Legacy-only; remove after asset refresh
            legacy_keys = {'column_sum'}
            actual_keys = set(table_emb.keys())
            if not actual_keys.issubset(expected_keys | legacy_keys):
                unexpected = actual_keys - expected_keys - legacy_keys
                errors.append(f"table_embedding dict has unexpected keys: {unexpected}")
            # Validate each sub-field
            for sub_key in ['cls_embedding', 'table_embedding', 'column_mean', 'token_mean']:
                sub_value = table_emb.get(sub_key)
                if sub_value is not None:
                    if isinstance(sub_value, np.ndarray):
                        if sub_value.ndim != 1:
                            errors.append(f"table_embedding.{sub_key} should be 1D, got shape {sub_value.shape}")
                    elif isinstance(sub_value, list):
                        pass  # Lists are acceptable
                    else:
                        errors.append(f"table_embedding.{sub_key} has invalid type: {type(sub_value)}")
        elif isinstance(table_emb, np.ndarray):
            # v1.0 format: array
            if table_emb.ndim != 1:
                errors.append(f"table_embedding should be 1D, got shape {table_emb.shape}")
        elif isinstance(table_emb, list):
            pass  # Lists are acceptable (v1.0 serialized)
        else:
            errors.append(f"table_embedding has invalid type: {type(table_emb)}")

    # Check cls_embedding at top level (v1.0 format only)
    cls_emb = data.get('cls_embedding')
    if cls_emb is not None:
        if isinstance(cls_emb, np.ndarray):
            if cls_emb.ndim != 1:
                errors.append(f"cls_embedding should be 1D, got shape {cls_emb.shape}")
        elif isinstance(cls_emb, list):
            pass  # Lists are acceptable
        else:
            errors.append(f"cls_embedding has invalid type: {type(cls_emb)}")

    # Check context_embedding (can be 2D)
    ctx_emb = data.get('context_embedding')
    if ctx_emb is not None:
        if isinstance(ctx_emb, np.ndarray):
            if ctx_emb.ndim not in (1, 2):
                errors.append(f"context_embedding should be 1D or 2D, got shape {ctx_emb.shape}")

    # Check column_names
    col_names = data.get('column_names')
    if col_names is not None and not isinstance(col_names, list):
        errors.append(f"column_names should be list, got {type(col_names)}")

    return len(errors) == 0, errors


def validate_embedding_batch(
    data: Union[Dict, EmbeddingBatch, List],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate an embedding batch.

    Args:
        data: Batch data to validate
        strict: If True, require all fields

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Convert to list of items
    if isinstance(data, EmbeddingBatch):
        items = [r.to_dict() for r in data.results]
    elif isinstance(data, dict):
        fmt = data.get('format', '')
        if fmt == 'unified_batch_embedding':
            items = data.get('results', [])
        else:
            # Single item
            items = [data]
    elif isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, TableEmbeddingResult):
                items.append(item.to_dict())
            elif isinstance(item, dict):
                items.append(item)
            else:
                errors.append(f"Invalid item type in list: {type(item)}")
    else:
        return False, [f"Expected dict, list, or EmbeddingBatch, got {type(data)}"]

    # Validate each item
    for i, item in enumerate(items):
        is_valid, item_errors = validate_embedding_result(item, strict=strict)
        for err in item_errors:
            errors.append(f"Item {i}: {err}")

    return len(errors) == 0, errors


def check_embedding_compatibility(
    data1: Union[Dict, List],
    data2: Union[Dict, List]
) -> Tuple[bool, List[str]]:
    """
    Check if two embedding datasets are compatible (same dimension, etc.).

    Useful for checking query/datalake compatibility in join search.

    Args:
        data1: First embedding dataset
        data2: Second embedding dataset

    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []

    def get_dim(data):
        """Extract embedding dimension from data."""
        if isinstance(data, dict):
            return data.get('embedding_dim', 0)
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                col_emb = first.get('column_embeddings') or first.get('column_embedding', {})
                if col_emb:
                    first_emb = next(iter(col_emb.values()))
                    if isinstance(first_emb, np.ndarray):
                        return first_emb.shape[0]
                    elif isinstance(first_emb, list):
                        return len(first_emb)
            elif isinstance(first, tuple) and len(first) == 3:
                emb = first[2]
                if isinstance(emb, np.ndarray):
                    return emb.shape[0]
        return 0

    dim1 = get_dim(data1)
    dim2 = get_dim(data2)

    if dim1 and dim2 and dim1 != dim2:
        issues.append(f"Embedding dimensions differ: {dim1} vs {dim2}")

    return len(issues) == 0, issues


def get_format_summary(data: Any) -> Dict[str, Any]:
    """
    Get a summary of the embedding data format.

    Args:
        data: Embedding data

    Returns:
        Dict with format information
    """
    is_unified, fmt_type = is_unified_format(data)

    summary = {
        'is_unified_format': is_unified,
        'format_type': fmt_type,
        'num_tables': 0,
        'embedding_dim': 0,
        'model_name': '',
        'has_table_embedding': False,
        'has_cls_embedding': False,
    }

    if isinstance(data, EmbeddingBatch):
        summary['num_tables'] = len(data.results)
        summary['embedding_dim'] = data.embedding_dim
        summary['model_name'] = data.model_name
        if data.results:
            first = data.results[0]
            summary['has_table_embedding'] = first.table_embedding is not None
            summary['has_cls_embedding'] = first.cls_embedding is not None

    elif isinstance(data, dict):
        if data.get('format') == 'unified_batch_embedding':
            results = data.get('results', [])
            summary['num_tables'] = len(results)
            summary['embedding_dim'] = data.get('embedding_dim', 0)
            summary['model_name'] = data.get('model_name', '')
        else:
            summary['num_tables'] = 1
            summary['embedding_dim'] = data.get('embedding_dim', 0)
            summary['model_name'] = data.get('model_name', '')
            summary['has_table_embedding'] = data.get('table_embedding') is not None
            summary['has_cls_embedding'] = data.get('cls_embedding') is not None

    elif isinstance(data, list):
        summary['num_tables'] = len(data)
        if data:
            first = data[0]
            if isinstance(first, dict):
                col_emb = first.get('column_embeddings') or first.get('column_embedding', {})
                if col_emb:
                    first_emb = next(iter(col_emb.values()))
                    if isinstance(first_emb, np.ndarray):
                        summary['embedding_dim'] = first_emb.shape[0]
                    elif isinstance(first_emb, list):
                        summary['embedding_dim'] = len(first_emb)
                summary['model_name'] = first.get('model_name', '')
                summary['has_table_embedding'] = first.get('table_embedding') is not None
                summary['has_cls_embedding'] = first.get('cls_embedding') is not None
            elif isinstance(first, tuple) and len(first) == 3:
                emb = first[2]
                if isinstance(emb, np.ndarray):
                    summary['embedding_dim'] = emb.shape[0]

    return summary
