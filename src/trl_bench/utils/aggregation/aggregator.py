"""
Embedding aggregation functions.

This module provides reusable functions to aggregate embeddings from
lower-level representations (e.g., column embeddings) to higher-level
representations (e.g., table embeddings).
"""

from enum import Enum
from typing import Dict, List, Optional, Union
import numpy as np


class AggregationMethod(Enum):
    """Supported aggregation methods."""
    MEAN = 'mean'
    SUM = 'sum'
    MAX = 'max'
    MIN = 'min'


# List of supported aggregation method names
SUPPORTED_AGGREGATIONS = [method.value for method in AggregationMethod]


def aggregate_embeddings(
    embeddings: Union[Dict[int, np.ndarray], List[np.ndarray], np.ndarray],
    method: Union[str, AggregationMethod] = AggregationMethod.MEAN,
) -> Optional[np.ndarray]:
    """
    Aggregate multiple embeddings into a single embedding.

    This function takes a collection of embeddings (e.g., column embeddings)
    and aggregates them into a single embedding (e.g., table embedding) using
    the specified aggregation method.

    Args:
        embeddings: Collection of embeddings to aggregate. Can be:
            - Dict[int, np.ndarray]: Column index to embedding mapping
            - List[np.ndarray]: List of embedding arrays
            - np.ndarray: 2D array of shape (num_embeddings, embedding_dim)
        method: Aggregation method. One of:
            - 'mean' or AggregationMethod.MEAN: Element-wise mean (default)
            - 'sum' or AggregationMethod.SUM: Element-wise sum
            - 'max' or AggregationMethod.MAX: Element-wise maximum
            - 'min' or AggregationMethod.MIN: Element-wise minimum

    Returns:
        np.ndarray: Aggregated embedding of shape (embedding_dim,), or None if
            the input is empty or invalid.

    Examples:
        >>> col_embs = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        >>> aggregate_embeddings(col_embs, 'mean')
        array([2., 3.])

        >>> aggregate_embeddings(col_embs, 'sum')
        array([4., 6.])
    """
    # Convert method string to enum if needed
    if isinstance(method, str):
        try:
            method = AggregationMethod(method.lower())
        except ValueError:
            raise ValueError(
                f"Unknown aggregation method: {method}. "
                f"Supported methods: {SUPPORTED_AGGREGATIONS}"
            )

    # Convert input to a list of arrays
    if embeddings is None:
        return None

    if isinstance(embeddings, dict):
        if len(embeddings) == 0:
            return None
        # Sort by key to ensure deterministic ordering
        embedding_list = [embeddings[k] for k in sorted(embeddings.keys())]
    elif isinstance(embeddings, np.ndarray):
        if embeddings.ndim == 1:
            # Single embedding, return as-is
            return embeddings.astype(np.float32)
        elif embeddings.ndim == 2:
            if embeddings.shape[0] == 0:
                return None
            embedding_list = list(embeddings)
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {embeddings.shape}")
    elif isinstance(embeddings, list):
        if len(embeddings) == 0:
            return None
        embedding_list = embeddings
    else:
        raise TypeError(
            f"Expected dict, list, or np.ndarray, got {type(embeddings)}"
        )

    # Stack embeddings into 2D array
    try:
        stacked = np.stack(embedding_list, axis=0)
    except ValueError as e:
        raise ValueError(f"Cannot stack embeddings: {e}")

    # Apply aggregation
    if method == AggregationMethod.MEAN:
        result = np.mean(stacked, axis=0)
    elif method == AggregationMethod.SUM:
        result = np.sum(stacked, axis=0)
    elif method == AggregationMethod.MAX:
        result = np.max(stacked, axis=0)
    elif method == AggregationMethod.MIN:
        result = np.min(stacked, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

    return result.astype(np.float32)


