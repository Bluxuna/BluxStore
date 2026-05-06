"""Distance and similarity helpers."""

from __future__ import annotations

import numpy as np


EPSILON: float = 1e-12


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize each row vector to unit L2 norm.

    Args:
        vectors: A 2D array with shape (n_vectors, dim).

    Returns:
        A 2D array of the same shape where each vector has unit L2 norm.

    Raises:
        ValueError: If `vectors` is not a 2D NumPy array.
    """
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("vectors must be a 2D NumPy array")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, EPSILON)


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query and many vectors.

    Args:
        query: A 1D array with shape (dim,).
        vectors: A 2D array with shape (n_vectors, dim).

    Returns:
        A 1D array with cosine similarity scores, shape (n_vectors,).

    Raises:
        ValueError: If the input shapes are invalid.
    """
    if query.ndim != 1:
        raise ValueError("query must be a 1D array")
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("query dimension must match vectors dimension")

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denom = np.maximum(query_norm * vector_norms, EPSILON)
    return (vectors @ query) / denom


def euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from one query to many vectors.

    Args:
        query: A 1D array with shape (dim,).
        vectors: A 2D array with shape (n_vectors, dim).

    Returns:
        A 1D array with Euclidean distances, shape (n_vectors,).

    Raises:
        ValueError: If the input shapes are invalid.
    """
    if query.ndim != 1:
        raise ValueError("query must be a 1D array")
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("query dimension must match vectors dimension")

    diff = vectors - query[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1))


def top_k_indices(scores: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Return top-k indices from a score array using partial sorting.

    Args:
        scores: A 1D array of similarity or distance values.
        k: Number of indices to return.
        largest: If True, higher scores are better. If False, lower scores are
            better.

    Returns:
        A 1D array containing `k` indices sorted by ranking quality.

    Raises:
        ValueError: If `scores` is not 1D or if `k` is not positive.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, scores.shape[0])
    if largest:
        candidate_idx = np.argpartition(scores, -k)[-k:]
        order = np.argsort(scores[candidate_idx])[::-1]
    else:
        candidate_idx = np.argpartition(scores, k - 1)[:k]
        order = np.argsort(scores[candidate_idx])

    return candidate_idx[order]
