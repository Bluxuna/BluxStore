"""Tests for distance and normalization helpers."""

from __future__ import annotations

import numpy as np

from blux_search.distance import cosine_similarity, euclidean_distance, l2_normalize


def test_l2_normalize_produces_unit_vectors() -> None:
    vectors = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, np.array([1.0, 1.0], dtype=np.float32), atol=1e-6)


def test_cosine_similarity_expected_order() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)

    scores = cosine_similarity(query, vectors)
    order = np.argsort(scores)[::-1]

    assert int(order[0]) == 0
    assert int(order[1]) == 2


def test_euclidean_distance_expected_order() -> None:
    query = np.array([1.0, 1.0], dtype=np.float32)
    vectors = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float32)

    dist = euclidean_distance(query, vectors)
    order = np.argsort(dist)

    assert int(order[0]) == 0
    assert int(order[1]) == 1
