"""
Format adapters for converting between unified format and task-specific formats.

These adapters enable downstream tasks to extract table-level embeddings
from the unified v2.0 format.

Format (v2.0):
    table_embedding is a dict with keys: cls_embedding, table_embedding,
    column_mean, token_mean
"""

from typing import List, Tuple, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np

from .schema import TableEmbeddingResult, EmbeddingBatch, TableLevelEmbedding


VALID_VARIANTS = {'cls_embedding', 'table_embedding', 'column_mean', 'token_mean'}


def get_table_level_embedding(
    item: Dict[str, Any],
    variant: str = 'column_mean',
) -> Optional[np.ndarray]:
    """
    Extract a table-level embedding from v2.0 format.

    Returns exactly the requested variant or None — no silent fallback.

    Args:
        item: Embedding dict (v2.0 format)
        variant: Which variant to extract. Options:
            - 'column_mean': Mean-pooled column embeddings (default, most common)
            - 'cls_embedding': CLS token embedding (if represents table)
            - 'table_embedding': Native table embedding (if model supports)
            - 'token_mean': Mean of all non-padding token hidden states

    Returns:
        np.ndarray of shape (embedding_dim,) or None if not available

    Raises:
        ValueError: If variant is not a recognised variant name
        TypeError: If table_embedding is not a dict (and not None) — indicates v1.0 format
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Valid: {sorted(VALID_VARIANTS)}"
        )

    table_emb = item.get('table_embedding')

    if table_emb is None:
        return None

    if not isinstance(table_emb, dict):
        raise TypeError(
            f"table_embedding must be a dict (v2.0 format), got {type(table_emb).__name__}. "
            f"v1.0 format (raw array) is no longer supported — regenerate embeddings."
        )

    value = table_emb.get(variant)
    if value is not None:
        return np.array(value, dtype=np.float32) if isinstance(value, list) else value
    return None


def extract_table_embeddings_batch(
    data: List[Dict[str, Any]],
    variant: str = 'column_mean',
) -> Dict[str, np.ndarray]:
    """
    Extract table-level embeddings from a batch of embedding dicts.

    Convenience function for downstream tasks that need table embeddings
    from a list of embedding results.

    Args:
        data: List of embedding dicts (v2.0 format)
        variant: Which variant to extract (see get_table_level_embedding)

    Returns:
        Dict mapping table_id to embedding array
    """
    result = {}
    for item in data:
        table_id = item.get('table_id') or item.get('table') or item.get('table_name', '')
        if not table_id:
            continue

        # Extract from path if needed
        if '/' in table_id or '\\' in table_id:
            table_id = Path(table_id).stem
        if table_id.endswith('.csv'):
            table_id = table_id[:-4]

        emb = get_table_level_embedding(item, variant=variant)
        if emb is not None:
            result[table_id] = emb

    return result


def to_join_search_format(
    data: Union[List[Dict], EmbeddingBatch, List[TableEmbeddingResult]],
    embedding_key: str = 'column_embeddings'
) -> List[Tuple[str, int, np.ndarray]]:
    """
    Convert unified format to join search tuple format.

    Join search expects: [(table_name, col_idx, embedding), ...]

    This is the format used by downstream_tasks/join_search/run_search.py

    Args:
        data: Unified format embeddings (list of dicts or EmbeddingBatch)
        embedding_key: Key for column embeddings ('column_embeddings' or 'column_embedding')

    Returns:
        List of tuples: (table_name, column_index, embedding_array)
    """
    result = []

    # Normalize input to list of items
    if isinstance(data, EmbeddingBatch):
        items = [r.to_dict() for r in data.results]
    elif isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, TableEmbeddingResult):
                items.append(item.to_dict())
            elif isinstance(item, dict):
                items.append(item)
            else:
                raise TypeError(f"Unsupported item type: {type(item)}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    for item in items:
        # Get table identifier
        table_id = item.get('table_id') or item.get('table') or item.get('table_name', '')

        # Handle 'column_embedding' (singular) or 'column_embeddings' (plural)
        col_emb = item.get('column_embeddings') or item.get('column_embedding', {})

        if not col_emb:
            continue

        # Convert each column to tuple
        for col_idx, embedding in col_emb.items():
            col_idx = int(col_idx) if isinstance(col_idx, str) else col_idx

            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)

            result.append((table_id, col_idx, embedding))

    return result


def to_starmie_format(
    data: Union[List[Dict], EmbeddingBatch, List[TableEmbeddingResult]]
) -> List[Tuple[str, np.ndarray]]:
    """
    Convert unified format to Starmie's format.

    Starmie expects: [(table_name, column_embeddings_array), ...]
    where column_embeddings_array has shape (num_columns, embedding_dim)

    Args:
        data: Unified format embeddings

    Returns:
        List of tuples: (table_name, column_embeddings_array)
    """
    result = []

    # Normalize input
    if isinstance(data, EmbeddingBatch):
        items = [r.to_dict() for r in data.results]
    elif isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, TableEmbeddingResult):
                items.append(item.to_dict())
            elif isinstance(item, dict):
                items.append(item)
            else:
                raise TypeError(f"Unsupported item type: {type(item)}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    for item in items:
        table_id = item.get('table_id') or item.get('table') or item.get('table_name', '')

        col_emb = item.get('column_embeddings') or item.get('column_embedding', {})

        if not col_emb:
            continue

        # Sort by column index and stack
        sorted_cols = sorted(col_emb.items(), key=lambda x: int(x[0]) if isinstance(x[0], str) else x[0])
        embeddings = []

        for _, emb in sorted_cols:
            if isinstance(emb, list):
                embeddings.append(np.array(emb, dtype=np.float32))
            elif isinstance(emb, np.ndarray):
                embeddings.append(emb.astype(np.float32))

        if embeddings:
            col_array = np.stack(embeddings, axis=0)
            result.append((table_id, col_array))

    return result


def from_legacy_format(
    data: Union[List[Tuple], List[Dict], Dict],
    model_name: str = '',
    embedding_dim: int = 0
) -> List[Dict]:
    """
    Convert legacy formats to unified dict format.

    Supports:
    - Tuple format: [(table, col, emb), ...]
    - Legacy dict format: [{'table': ..., 'column_embedding': ...}, ...]
    - TabSketchFM dict format: {table_name: {'column_embeddings': ...}, ...}

    Args:
        data: Legacy format data
        model_name: Model name to add to results
        embedding_dim: Embedding dimension (auto-detected if not provided)

    Returns:
        List of dicts in unified format
    """
    result = []

    if isinstance(data, dict) and not isinstance(data, list):
        # Dict with table_name keys (TabSketchFM batch format)
        for table_name, table_data in data.items():
            if isinstance(table_data, dict):
                # Get column embeddings
                col_emb = table_data.get('column_embeddings') or table_data.get('col_embeddings', {})

                result.append({
                    'table_id': table_name,
                    'table': table_data.get('source_path') or table_name,
                    'model_name': model_name or table_data.get('model_name', ''),
                    'embedding_dim': embedding_dim or table_data.get('embedding_dim', 0),
                    'column_embeddings': col_emb,
                    'table_embedding': table_data.get('table_embedding'),
                    'cls_embedding': table_data.get('cls_embedding'),
                    'column_names': table_data.get('column_names', []),
                })

    elif isinstance(data, list):
        if data and isinstance(data[0], tuple) and len(data[0]) == 3:
            # Tuple format: [(table, col, emb), ...]
            # Group by table
            tables = {}
            for table, col, emb in data:
                if table not in tables:
                    tables[table] = {}
                tables[table][col] = emb

            for table, cols in tables.items():
                # Auto-detect embedding dim
                if not embedding_dim and cols:
                    first_emb = next(iter(cols.values()))
                    if isinstance(first_emb, np.ndarray):
                        embedding_dim = first_emb.shape[0]
                    elif isinstance(first_emb, list):
                        embedding_dim = len(first_emb)

                result.append({
                    'table_id': Path(table).stem if '/' in table or '\\' in table else table,
                    'table': table,
                    'model_name': model_name,
                    'embedding_dim': embedding_dim,
                    'column_embeddings': cols,
                    'table_embedding': None,
                    'cls_embedding': None,
                    'column_names': [],
                })

        elif data and isinstance(data[0], dict):
            # List of dicts - normalize keys
            for item in data:
                col_emb = item.get('column_embeddings') or item.get('column_embedding', {})

                # Auto-detect embedding dim
                if not embedding_dim and col_emb:
                    first_emb = next(iter(col_emb.values()))
                    if isinstance(first_emb, np.ndarray):
                        embedding_dim = first_emb.shape[0]
                    elif isinstance(first_emb, list):
                        embedding_dim = len(first_emb)

                table = item.get('table') or item.get('table_id') or item.get('table_name', '')

                result.append({
                    'table_id': item.get('table_id') or (Path(table).stem if table and ('/' in table or '\\' in table) else table),
                    'table': table,
                    'model_name': model_name or item.get('model_name', ''),
                    'embedding_dim': embedding_dim or item.get('embedding_dim', 0),
                    'column_embeddings': col_emb,  # Use standard key (plural)
                    'table_embedding': item.get('table_embedding'),
                    'cls_embedding': item.get('cls_embedding'),
                    'column_names': item.get('column_names', []),
                })

    return result


def normalize_embeddings(
    data: Union[List[Dict], EmbeddingBatch],
    key_from: str = 'column_embedding',
    key_to: str = 'column_embeddings'
) -> List[Dict]:
    """
    Normalize embedding key names in place.

    Converts 'column_embedding' (singular) to 'column_embeddings' (plural).

    Args:
        data: Embedding data
        key_from: Source key to rename
        key_to: Target key name

    Returns:
        List of dicts with normalized keys
    """
    if isinstance(data, EmbeddingBatch):
        items = [r.to_dict() for r in data.results]
    elif isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, TableEmbeddingResult):
                items.append(item.to_dict())
            elif isinstance(item, dict):
                items.append(item.copy())  # Copy to avoid mutation
            else:
                raise TypeError(f"Unsupported item type: {type(item)}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    for item in items:
        if key_from in item and key_to not in item:
            item[key_to] = item.pop(key_from)

    return items


def unified_to_legacy_list(
    data: Union[List[Dict], EmbeddingBatch],
    include_table_path: bool = True
) -> List[Dict]:
    """
    Convert unified format to legacy list-of-dicts format.

    This is useful for backward compatibility with code expecting the old format.

    Args:
        data: Unified format data
        include_table_path: If True, include 'table' key with path

    Returns:
        List of dicts in legacy format
    """
    if isinstance(data, EmbeddingBatch):
        items = data.results
    elif isinstance(data, list):
        items = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    result = []
    for item in items:
        if isinstance(item, TableEmbeddingResult):
            d = item.to_dict()
        elif isinstance(item, dict):
            d = item
        else:
            continue

        legacy = {
            'table': d.get('source_path') or d.get('table') or d.get('table_id', ''),
            'column_embeddings': d.get('column_embeddings', {}),  # Use plural
            'table_embedding': d.get('table_embedding'),
            'cls_embedding': d.get('cls_embedding'),
        }

        if not include_table_path:
            legacy.pop('table', None)

        result.append(legacy)

    return result
