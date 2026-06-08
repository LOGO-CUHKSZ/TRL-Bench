"""
Unified Embedding Format Utilities

This module provides standardized utilities for working with table embedding outputs
across different models in the TRL benchmark.

The unified format standardizes:
- Key naming: Always 'column_embeddings' (plural)
- Batch structure: Consistent dict format
- Optional fields: 'table_embedding', 'cls_embedding', 'context_embedding'
- Metadata: 'version', 'format', 'model_name', 'embedding_dim'

Usage:
    from trl_bench.utils.unified_embedding_format import (
        TableEmbeddingResult,
        EmbeddingBatch,
        save_embeddings,
        load_embeddings,
        to_join_search_format,
        to_starmie_format,
        validate_embedding_result,
    )

    # Create a result
    result = TableEmbeddingResult(
        table_id='my_table',
        model_name='doduo',
        embedding_dim=768,
        column_embeddings={0: np.array([...]), 1: np.array([...])}
    )

    # Save
    save_embeddings(result, 'output.pkl')

    # Load and convert to legacy format
    loaded = load_embeddings('output.pkl')
    tuples = to_join_search_format(loaded)
"""

from .schema import (
    TableEmbeddingResult,
    TableLevelEmbedding,
    EmbeddingBatch,
    RowEmbeddingMetadata,
    RowEmbeddingMetadataV2,
    SplitInfo,
    UNIFIED_TABLE_VERSION,
    UNIFIED_BATCH_VERSION,
    UNIFIED_ROW_VERSION,
    UNIFIED_ROW_VERSION_V2,
)

from .io import (
    save_embeddings,
    load_embeddings,
    save_row_embeddings,
    load_row_embeddings,
    save_split_embeddings,
    load_split_embeddings,
    encode_label_column,
)

from .adapters import (
    to_join_search_format,
    to_starmie_format,
    from_legacy_format,
    normalize_embeddings,
    get_table_level_embedding,
    extract_table_embeddings_batch,
)

from .validators import (
    validate_embedding_result,
    validate_embedding_batch,
    is_unified_format,
)

__all__ = [
    # Schema
    'TableEmbeddingResult',
    'TableLevelEmbedding',
    'EmbeddingBatch',
    'RowEmbeddingMetadata',
    'RowEmbeddingMetadataV2',
    'SplitInfo',
    'UNIFIED_TABLE_VERSION',
    'UNIFIED_BATCH_VERSION',
    'UNIFIED_ROW_VERSION',
    'UNIFIED_ROW_VERSION_V2',
    # IO
    'save_embeddings',
    'load_embeddings',
    'save_row_embeddings',
    'load_row_embeddings',
    'save_split_embeddings',
    'load_split_embeddings',
    'encode_label_column',
    # Adapters
    'to_join_search_format',
    'to_starmie_format',
    'from_legacy_format',
    'normalize_embeddings',
    'get_table_level_embedding',
    'extract_table_embeddings_batch',
    # Validators
    'validate_embedding_result',
    'validate_embedding_batch',
    'is_unified_format',
]

__version__ = '2.0'
