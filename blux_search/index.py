"""Core indexing engine for Blux Search."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from blux_search.clustering import assign_to_centroids, kmeans
from blux_search.distance import (
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
    top_k_indices,
)


Metric = str
_SUPPORTED_METRICS = {"cosine", "euclidean"}


@dataclass(frozen=True)
class SearchResult:
    """Search hit returned by the engine.

    Attributes:
        idx: Index of the matched vector in the base matrix.
        score: Similarity score (for cosine) or negative distance proxy.
        metadata: User-provided metadata for the matched vector.
    """

    idx: int
    score: float
    metadata: dict[str, Any]


class BluxEngine:
    """Vector search engine supporting flat and IVF search."""

    def __init__(
        self,
        metric: Metric = "cosine",
        normalize: bool = True,
    ) -> None:
        """Create an engine instance.

        Args:
            metric: Either "cosine" or "euclidean".
            normalize: Whether to L2-normalize vectors on ingest and query.

        Raises:
            ValueError: If metric is unsupported.
        """
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Expected one of {_SUPPORTED_METRICS}."
            )

        self.metric = metric
        self.normalize = normalize

        self.vectors: np.ndarray | None = None
        self.metadata: list[dict[str, Any]] = []

        self.ivf_centroids: np.ndarray | None = None
        self.ivf_assignments: np.ndarray | None = None
        self.ivf_lists: dict[int, np.ndarray] = {}

    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None) -> None:
        """Add vectors and optional per-vector metadata."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")

        vectors_fp32 = vectors.astype(np.float32, copy=True)
        if self.normalize:
            vectors_fp32 = l2_normalize(vectors_fp32)

        if metadata is None:
            metadata = [{"id": i} for i in range(vectors_fp32.shape[0])]

        if len(metadata) != vectors_fp32.shape[0]:
            raise ValueError("metadata length must match number of vectors")

        if self.vectors is None:
            self.vectors = vectors_fp32
            self.metadata = list(metadata)
        else:
            if vectors_fp32.shape[1] != self.vectors.shape[1]:
                raise ValueError("new vectors dimension must match existing index")
            offset = self.vectors.shape[0]
            self.vectors = np.vstack([self.vectors, vectors_fp32])

            # Keep deterministic ids when the caller does not provide one.
            patched_meta: list[dict[str, Any]] = []
            for i, meta in enumerate(metadata):
                item = dict(meta)
                item.setdefault("id", offset + i)
                patched_meta.append(item)
            self.metadata.extend(patched_meta)

    def build_ivf(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        """Build IVF centroids and posting lists.

        Args:
            n_clusters: Number of IVF lists (coarse clusters).
            max_iter: K-Means max iterations.
            tol: K-Means centroid shift tolerance.
            seed: Random seed.

        Raises:
            ValueError: If vectors are not added before building IVF.
        """
        if self.vectors is None:
            raise ValueError("add vectors before building IVF")

        centroids, assignments = kmeans(
            self.vectors,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        self.ivf_centroids = centroids
        self.ivf_assignments = assignments

        self.ivf_lists = {
            cluster_id: np.where(assignments == cluster_id)[0]
            for cluster_id in range(n_clusters)
        }

    def _score(self, query: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, bool]:
        """Score candidates using the configured metric.

        Args:
            query: Query vector, shape (dim,).
            candidates: Candidate matrix, shape (n_candidates, dim).

        Returns:
            Tuple of scores and sorting direction flag where True means larger is
            better.
        """
        if self.metric == "cosine":
            return cosine_similarity(query, candidates), True

        # Distance is lower-is-better; negate it so ranking stays descending.
        distances = euclidean_distance(query, candidates)
        return -distances, True

    def search_flat(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Run exact search over the full index.

        Args:
            query: Query embedding with shape (dim,).
            k: Number of top results.

        Returns:
            List of top-k SearchResult objects sorted by relevance.

        Raises:
            ValueError: If index is empty or query shape is invalid.
        """
        if self.vectors is None:
            raise ValueError("index is empty; add vectors first")
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1]:
            raise ValueError("query must be 1D with matching vector dimension")

        query_vec = query.astype(np.float32, copy=True)
        if self.normalize:
            query_vec = l2_normalize(query_vec[None, :])[0]

        scores, largest = self._score(query_vec, self.vectors)
        top_idx = top_k_indices(scores, k=k, largest=largest)
        return [
            SearchResult(
                idx=int(idx),
                score=float(scores[idx]),
                metadata=self.metadata[int(idx)],
            )
            for idx in top_idx
        ]

    def search_ivf(self, query: np.ndarray, k: int = 5, n_probe: int = 2) -> list[SearchResult]:
        """Run approximate search using IVF candidate pruning.

        Args:
            query: Query embedding with shape (dim,).
            k: Number of top results.
            n_probe: Number of nearest IVF clusters to inspect.

        Returns:
            Top-k SearchResult list based on probed candidates.

        Raises:
            ValueError: If IVF is not built or inputs are invalid.
        """
        if self.vectors is None:
            raise ValueError("index is empty; add vectors first")
        if self.ivf_centroids is None:
            raise ValueError("IVF not built; call build_ivf first")
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1]:
            raise ValueError("query must be 1D with matching vector dimension")

        query_vec = query.astype(np.float32, copy=True)
        if self.normalize:
            query_vec = l2_normalize(query_vec[None, :])[0]

        centroid_dist = euclidean_distance(query_vec, self.ivf_centroids)
        probe_idx = top_k_indices(-centroid_dist, k=n_probe, largest=True)

        candidate_ids: list[int] = []
        for cid in probe_idx:
            candidate_ids.extend(self.ivf_lists.get(int(cid), np.array([], dtype=int)).tolist())

        if not candidate_ids:
            return []

        unique_ids = np.unique(np.array(candidate_ids, dtype=np.int64))
        candidate_vectors = self.vectors[unique_ids]
        scores, largest = self._score(query_vec, candidate_vectors)
        local_top = top_k_indices(scores, k=min(k, unique_ids.shape[0]), largest=largest)

        return [
            SearchResult(
                idx=int(unique_ids[i]),
                score=float(scores[i]),
                metadata=self.metadata[int(unique_ids[i])],
            )
            for i in local_top
        ]

    def save(self, directory: str | Path) -> None:
        """Save vectors, metadata, and IVF state to disk.

        Args:
            directory: Target path for serialized engine files.

        Raises:
            ValueError: If index has not been initialized.
        """
        if self.vectors is None:
            raise ValueError("cannot save empty index")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "vectors.npy", self.vectors)

        if self.ivf_centroids is not None:
            np.save(path / "ivf_centroids.npy", self.ivf_centroids)
        if self.ivf_assignments is not None:
            np.save(path / "ivf_assignments.npy", self.ivf_assignments)

        meta = {
            "metric": self.metric,
            "normalize": self.normalize,
            "metadata": self.metadata,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "BluxEngine":
        """Load an engine from disk.

        Args:
            directory: Path containing persisted engine files.

        Returns:
            Reconstructed BluxEngine instance.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        path = Path(directory)
        meta_path = path / "metadata.json"
        vectors_path = path / "vectors.npy"

        if not meta_path.exists() or not vectors_path.exists():
            raise FileNotFoundError("metadata.json and vectors.npy are required")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        engine = cls(metric=meta["metric"], normalize=meta["normalize"])
        engine.vectors = np.load(vectors_path)
        engine.metadata = meta["metadata"]

        centroids_path = path / "ivf_centroids.npy"
        assignments_path = path / "ivf_assignments.npy"
        if centroids_path.exists() and assignments_path.exists():
            engine.ivf_centroids = np.load(centroids_path)
            engine.ivf_assignments = np.load(assignments_path)

            n_clusters = engine.ivf_centroids.shape[0]
            engine.ivf_lists = {
                cluster_id: np.where(engine.ivf_assignments == cluster_id)[0]
                for cluster_id in range(n_clusters)
            }

        return engine

    def encode_texts(
        self,
        texts: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> np.ndarray:
        """Encode text with Sentence-Transformers.

        Args:
            texts: Input text list.
            model_name: Sentence-Transformers model identifier.

        Returns:
            Embedding matrix with shape (len(texts), dim).

        Raises:
            ValueError: If input text list is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return embeddings.astype(np.float32)

    def assign_queries_to_clusters(self, queries: np.ndarray) -> np.ndarray:
        """Assign query vectors to the existing IVF centroids.

        Args:
            queries: Query matrix with shape (n_queries, dim).

        Returns:
            Cluster id array with shape (n_queries,).

        Raises:
            ValueError: If IVF is not built.
        """
        if self.ivf_centroids is None:
            raise ValueError("IVF not built")
        return assign_to_centroids(queries, self.ivf_centroids)
