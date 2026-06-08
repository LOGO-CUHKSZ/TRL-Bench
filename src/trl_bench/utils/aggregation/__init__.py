"""
Embedding aggregation utilities.

This module provides functions to aggregate lower-level embeddings into
higher-level representations (e.g., column embeddings → table embedding).
"""

from .aggregator import (
    aggregate_embeddings,
    AggregationMethod,
    SUPPORTED_AGGREGATIONS,
)

__all__ = [
    'aggregate_embeddings',
    'AggregationMethod',
    'SUPPORTED_AGGREGATIONS',
]
