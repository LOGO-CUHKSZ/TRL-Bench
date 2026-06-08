"""
TAPAS: Table Parser for Table Understanding and Question Answering

TAPAS is a BERT-based model for answering questions about tabular data,
introduced in "TAPAS: Weakly Supervised Table Parsing via Pre-training"
(Herzig et al., ACL 2020).

Paper: https://arxiv.org/abs/2004.02349

This module provides:
- TAPASEmbedder: Generate table and column embeddings from CSV files
- TAPASQuestionAnswering: Answer natural language questions about tables

Example usage:
    from models.tapas import TAPASEmbedder, TAPASQuestionAnswering

    # Generate embeddings
    embedder = TAPASEmbedder()
    embeddings = embedder.encode_csv('table.csv')

    # Answer questions
    qa = TAPASQuestionAnswering()
    answer = qa.answer('table.csv', 'What is the total revenue?')
"""

from .generate_column_embeddings import (
    TAPASEmbedder,
    get_column_embeddings,
)

from .tapas_qa import (
    TAPASQuestionAnswering,
)

__all__ = [
    'TAPASEmbedder',
    'TAPASQuestionAnswering',
    'get_column_embeddings',
]

__version__ = '1.0.0'
