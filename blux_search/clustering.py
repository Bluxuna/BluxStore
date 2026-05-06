"""NumPy-based clustering utilities for IVF indexing."""

from __future__ import annotations

import numpy as np


def _validate_kmeans_inputs(vectors: np.ndarray, n_clusters: int) -> None:
    """Validate K-Means inputs before clustering.

    Args:
        vectors: A 2D matrix with shape (n_samples, dim).
        n_clusters: Number of clusters.

    Raises:
        ValueError: If input data is invalid.
    """
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("vectors must be a 2D NumPy array")
    if vectors.shape[0] == 0:
        raise ValueError("vectors must not be empty")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > vectors.shape[0]:
        raise ValueError("n_clusters cannot exceed number of vectors")


def assign_to_centroids(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each vector to the closest centroid.

    Args:
        vectors: Matrix of points, shape (n_samples, dim).
        centroids: Matrix of centroids, shape (n_clusters, dim).

    Returns:
        Cluster assignment array of shape (n_samples,).

    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    if vectors.ndim != 2 or centroids.ndim != 2:
        raise ValueError("vectors and centroids must both be 2D")
    if vectors.shape[1] != centroids.shape[1]:
        raise ValueError("vectors and centroids dimensions must match")

    # Batched squared distances via ||x-c||^2 = ||x||^2 + ||c||^2 - 2x·c.
    v_norm = np.sum(vectors * vectors, axis=1, keepdims=True)
    c_norm = np.sum(centroids * centroids, axis=1)[None, :]
    distances = v_norm + c_norm - 2.0 * (vectors @ centroids.T)
    return np.argmin(distances, axis=1)


def kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run K-Means with NumPy.

    Args:
        vectors: Matrix of vectors with shape (n_samples, dim).
        n_clusters: Number of centroids to learn.
        max_iter: Maximum iteration count.
        tol: Stop if centroid shift is below this threshold.
        seed: Random seed for centroid initialization.

    Returns:
        A tuple containing:
            - centroids: Array with shape (n_clusters, dim).
            - assignments: Array with shape (n_samples,) where each entry is the
              cluster id for a vector.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_kmeans_inputs(vectors, n_clusters)

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(vectors.shape[0], size=n_clusters, replace=False)
    centroids = vectors[init_idx].copy()

    assignments = np.zeros(vectors.shape[0], dtype=np.int64)

    for _ in range(max_iter):
        assignments = assign_to_centroids(vectors, centroids)
        new_centroids = centroids.copy()

        for cluster_id in range(n_clusters):
            members = vectors[assignments == cluster_id]
            if members.shape[0] > 0:
                new_centroids[cluster_id] = np.mean(members, axis=0)
            else:
                # Re-seed empty clusters so they can recover next iteration.
                rand_idx = rng.integers(0, vectors.shape[0])
                new_centroids[cluster_id] = vectors[rand_idx]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    final_assignments = assign_to_centroids(vectors, centroids)
    return centroids, final_assignments
