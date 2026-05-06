"""Integration tests for BluxEngine behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blux_search.index import BluxEngine


def test_flat_search_returns_exact_nearest_neighbor_cosine() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    metadata = [{"id": 0}, {"id": 1}, {"id": 2}]

    engine = BluxEngine(metric="cosine", normalize=True)
    engine.add(vectors, metadata)

    query = np.array([1.0, 0.0], dtype=np.float32)
    result = engine.search_flat(query, k=1)

    assert result[0].idx == 0


def test_flat_search_returns_exact_nearest_neighbor_euclidean() -> None:
    vectors = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    engine = BluxEngine(metric="euclidean", normalize=False)
    engine.add(vectors, metadata=[{"id": 0}, {"id": 1}, {"id": 2}])

    query = np.array([2.1, 2.1], dtype=np.float32)
    result = engine.search_flat(query, k=1)

    assert result[0].idx == 1


def test_ivf_search_finds_relevant_result() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    metadata = [{"id": i} for i in range(vectors.shape[0])]

    engine = BluxEngine(metric="cosine", normalize=True)
    engine.add(vectors, metadata)
    engine.build_ivf(n_clusters=2, max_iter=50, seed=7)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = engine.search_ivf(query, k=2, n_probe=1)

    assert len(results) > 0
    assert results[0].idx in {0, 1}


def test_persistence_round_trip(tmp_path: Path) -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = [{"id": 10, "text": "x"}, {"id": 11, "text": "y"}]

    engine = BluxEngine(metric="cosine", normalize=True)
    engine.add(vectors, metadata)
    engine.build_ivf(n_clusters=2, max_iter=10)

    store_dir = tmp_path / "store"
    engine.save(store_dir)

    loaded = BluxEngine.load(store_dir)
    query = np.array([1.0, 0.0], dtype=np.float32)
    result = loaded.search_flat(query, k=1)

    assert loaded.vectors is not None
    assert loaded.vectors.shape == vectors.shape
    assert result[0].metadata["id"] == 10
