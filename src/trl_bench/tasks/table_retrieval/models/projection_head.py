"""
Projection head for mapping embeddings to shared retrieval space.

Re-exports from the unified heads module for backward compatibility.
"""

from trl_bench.utils.downstream.heads import (
    ACTIVATIONS,
    AdapterLayer,
    MLPHead as ProjectionHead,
    DualProjectionHead,
)

__all__ = ['ACTIVATIONS', 'AdapterLayer', 'ProjectionHead', 'DualProjectionHead']
