"""FAISS utilities for index building and searching."""

import numpy as np
import faiss
from typing import Tuple


def build_index(embeddings: np.ndarray, use_gpu: bool = True,
                normalize: bool = True) -> faiss.Index:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: np.ndarray of shape (N, D)
        use_gpu: Whether to use GPU acceleration
        normalize: Whether to L2-normalize embeddings for cosine similarity

    Returns:
        faiss.Index
    """
    embeddings = embeddings.astype(np.float32)
    d = embeddings.shape[1]

    # L2 normalize for cosine similarity (inner product on unit vectors = cosine)
    if normalize:
        faiss.normalize_L2(embeddings)

    # Create index (inner product = cosine similarity after normalization)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    print(f"Built FAISS index with {index.ntotal} vectors of dimension {d}")

    # Move to GPU if available and requested
    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"Moving index to GPU (found {faiss.get_num_gpus()} GPUs)")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # Shard across multiple GPUs
        co.useFloat16 = True  # Use FP16 to reduce memory
        index = faiss.index_cpu_to_all_gpus(index, co=co)
        print("Index moved to GPU with sharding")

    return index


def search_index(index: faiss.Index, queries: np.ndarray, k: int = 100,
                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for top-k nearest neighbors.

    Args:
        index: FAISS index
        queries: np.ndarray of shape (Q, D) - query embeddings
        k: Number of nearest neighbors to retrieve
        normalize: Whether to L2-normalize queries

    Returns:
        distances: np.ndarray of shape (Q, k) - similarity scores
        indices: np.ndarray of shape (Q, k) - indices of retrieved items
    """
    queries = queries.astype(np.float32)

    # L2 normalize queries for cosine similarity
    if normalize:
        faiss.normalize_L2(queries)

    # Search
    distances, indices = index.search(queries, k)

    return distances, indices


def save_index(index: faiss.Index, path: str):
    """Save FAISS index to disk."""
    # If GPU index, convert to CPU first
    if hasattr(index, 'index'):  # GPU index wrapper
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path)
    print(f"Saved index to {path}")


def load_index(path: str, use_gpu: bool = True) -> faiss.Index:
    """Load FAISS index from disk."""
    index = faiss.read_index(path)
    print(f"Loaded index with {index.ntotal} vectors")

    if use_gpu and faiss.get_num_gpus() > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
        print("Index moved to GPU")

    return index
