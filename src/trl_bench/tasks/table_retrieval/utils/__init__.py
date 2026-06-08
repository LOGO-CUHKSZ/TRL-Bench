"""Utility functions for table retrieval."""

from .data_utils import (
    EmbeddingRetrievalDataset,
    load_table_embeddings,
    build_csv_to_table_id_mapping,
    load_query_embeddings,
    load_training_data,
    create_dataloader,
    load_tables,
    load_questions,
    load_id2table_mapping,
    save_id2table_mapping,
    build_id2table_mapping,
)
from .faiss_utils import build_index, search_index
from .metrics import compute_recall_at_k, compute_mrr, print_metrics

__all__ = [
    # Data utilities
    'EmbeddingRetrievalDataset',
    'load_table_embeddings',
    'load_query_embeddings',
    'load_training_data',
    'create_dataloader',
    'load_tables',
    'load_questions',
    'load_id2table_mapping',
    'save_id2table_mapping',
    'build_id2table_mapping',
    # FAISS utilities
    'build_index',
    'search_index',
    # Metrics
    'compute_recall_at_k',
    'compute_mrr',
    'print_metrics',
]
