# Blux Search

Blux Search is a production-oriented vector search engine built from scratch in Python.
It demonstrates both exact retrieval and approximate retrieval, with a design that mirrors
real systems while staying simple enough to study and extend.

Core principles:
- No external vector databases (no FAISS, Pinecone, Chroma, etc.).
- NumPy is used for all linear algebra, scoring, and clustering.
- Sentence-Transformers is used only to generate embeddings.

## Why This Project Exists

Many teams use managed vector databases without understanding what happens under the hood.
Blux Search provides a transparent reference implementation of the core building blocks:
- Flat (brute-force) search for correctness and baseline recall.
- IVF (Inverted File Index) for practical speedups using candidate pruning.
- Disk persistence for index durability.

This lets you reason about ranking quality, latency, memory layout, and ANN tradeoffs
before scaling to distributed systems.

## Repository Structure

```text
.
├── blux_search
│   ├── __init__.py
│   ├── clustering.py     # NumPy K-Means + centroid assignment
│   ├── distance.py       # Cosine / Euclidean math primitives
│   └── index.py          # BluxEngine (Flat + IVF + persistence)
├── tests
│   ├── test_distance.py
│   └── test_engine.py
├── main.py               # Demo + SQLite corpus/embedding CRUD cache
├── blux_store/           # Local persisted index + sqlite db
└── README.md
```

## Architecture Overview

### 1) Embedding Stage

Raw text is converted to dense vectors using Sentence-Transformers. This is the only model
component in the system.

Pipeline:
1. Input text list
2. SentenceTransformer model inference
3. Float32 embedding matrix `(N, D)`
4. Optional L2 normalization

### 2) Core Search Engine (`BluxEngine`)

`BluxEngine` stores vectors + metadata and exposes:
- `search_flat`: exact brute-force nearest-neighbor search
- `build_ivf`: coarse partitioning of vector space via K-Means
- `search_ivf`: approximate search over selected IVF lists
- `save` / `load`: persistence using `.npy` + JSON metadata

### 3) Flat Index (Exact Search)

Flat search compares the query vector against every indexed vector.
- Time complexity per query: `O(N * D)`
- Recall: 100% (exact)
- Latency: grows linearly with dataset size

Use this when correctness is critical or dataset size is modest.

### 4) IVF Index (Approximate Search)

IVF (Inverted File Index) reduces search cost by narrowing candidates:
1. Run K-Means over all vectors to learn `K` centroids.
2. Assign each vector to nearest centroid (its posting list).
3. At query time, find nearest `n_probe` centroids.
4. Score only vectors inside those selected lists.

Approximate query complexity:
- Coarse stage: `O(K * D)`
- Fine stage: `O(C * D)` where `C` is candidate count from probed lists
- Total: `O((K + C) * D)` instead of `O(N * D)`

When `C << N`, latency drops substantially.

## Architecture & Math (Detailed)

### Cosine Similarity

Cosine similarity measures angle similarity:

```math
\text{cos}(\theta) = \frac{q \cdot x}{\|q\|_2 \|x\|_2}
```

Interpretation:
- `1.0`: same direction
- `0.0`: orthogonal
- `-1.0`: opposite direction

With L2-normalized vectors (`||q|| = ||x|| = 1`), cosine simplifies to dot product:

```math
\text{cos}(\theta) = q \cdot x
```

This is why normalization is powerful: it turns cosine scoring into a fast matrix-vector multiply.

### Euclidean Distance

Euclidean distance measures geometric closeness:

```math
d(q, x) = \sqrt{\sum_i (q_i - x_i)^2}
```

Interpretation:
- Smaller is better.
- Sensitive to vector magnitude.

In implementation, Euclidean is converted to a score by negating distances so ranking can use
"higher is better" semantics consistently.

### K-Means for IVF

K-Means objective:

```math
\min_{\{c_j\}} \sum_{i=1}^{N} \|x_i - c_{a_i}\|_2^2
```

Where:
- `c_j`: centroid of cluster `j`
- `a_i`: cluster assignment for vector `x_i`

Algorithm loop:
1. Initialize centroids.
2. Assignment step: assign each point to nearest centroid.
3. Update step: recompute each centroid as mean of assigned points.
4. Repeat until centroid shift < tolerance.

In Blux Search, this is implemented using NumPy only.

### Why Flat to IVF is an Important Transition

Flat search gives perfect recall but poor scalability (`O(N)`).
IVF introduces controlled approximation: it searches fewer vectors while preserving relevance in most cases.

You move from:
- "scan everything"

to:
- "scan likely regions first"

This is the same foundational idea behind modern ANN systems.

## Persistence Layer

Blux Search simulates database disk storage with explicit file artifacts:
- `vectors.npy`: dense vector matrix
- `ivf_centroids.npy`: coarse cluster centers (if IVF built)
- `ivf_assignments.npy`: cluster id per vector (if IVF built)
- `metadata.json`: metric config, normalize flag, vector metadata

Benefits:
- Fast reload without recomputing embeddings or IVF training
- Deterministic reproducibility across restarts
- Easy observability of index internals

### SQLite Corpus + Embedding Cache

`main.py` also keeps a local SQLite database at `blux_store/embeddings.sqlite3` with:
- `corpus_data(text PRIMARY KEY)`: source/user corpus records
- `corpus_cache(model_name, text, embedding)`: model-scoped embedding cache

This avoids re-embedding the same text across script runs. Missing texts are embedded once and inserted into cache.
The cache is keyed by `(model_name, text)`, so model changes do not contaminate existing vectors.

CRUD helpers in `main.py`:
- `create_corpus_rows`
- `read_corpus_rows`
- `update_corpus_row` (invalidates stale cache rows)
- `delete_corpus_row` (cache rows cascade via FK)

## Simulated Benchmarking

These values are simulated but reflect realistic trends for ANN systems.
Environment assumption: CPU-only, 384-dim vectors, Python + NumPy.

| Index Size (N) | Flat Search p95 (ms) | IVF Search p95 (ms) (`K=256`, `n_probe=8`) | Recall@10 (IVF) |
|---:|---:|---:|---:|
| 10,000 | 9.8 | 2.1 | 0.99 |
| 100,000 | 93.4 | 8.7 | 0.97 |
| 1,000,000 | 945.0 | 41.5 | 0.94 |

Takeaway:
- Flat grows nearly linearly with `N`.
- IVF keeps latency growth much slower by pruning candidates.
- Recall can be tuned by adjusting `n_probe` (higher probe => better recall, higher latency).

## Installation

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## Usage

### Run the Demo

```bash
python main.py
```

What it does:
1. Inserts/reads corpus records in SQLite.
2. Reuses cached embeddings and embeds only missing texts.
3. Builds a Flat index and IVF index.
4. Executes a semantic query.
5. Prints exact vs approximate results.
6. Saves and reloads the index from disk.

### Minimal Programmatic Example

```python
import numpy as np
from blux_search import BluxEngine

engine = BluxEngine(metric="cosine", normalize=True)

vectors = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.9, 0.1],
], dtype=np.float32)
metadata = [{"id": 0}, {"id": 1}, {"id": 2}]

engine.add(vectors, metadata)
engine.build_ivf(n_clusters=2)

query = np.array([1.0, 0.0], dtype=np.float32)
exact = engine.search_flat(query, k=2)
approx = engine.search_ivf(query, k=2, n_probe=1)

print(exact)
print(approx)
```

## Testing

```bash
pytest -q
```

Test coverage validates:
- L2 normalization and distance math correctness
- Exact nearest-neighbor behavior for cosine and Euclidean metrics
- IVF result sanity on clustered vectors
- Persistence round-trip integrity

## Design Tradeoffs

- Flat index:
  - Pros: exact, simple, deterministic
  - Cons: linear latency with dataset size

- IVF index:
  - Pros: much faster on large datasets
  - Cons: approximate; quality depends on `K`, `n_probe`, and data distribution

- Normalization:
  - Pros: enables fast cosine via dot product
  - Cons: may remove useful magnitude information for some tasks

## Production Extension Ideas

- Quantization for memory reduction
- HNSW-like graph routing layer on top of IVF
- Batched query execution APIs
- Incremental centroid maintenance
- Structured filtering + hybrid lexical/vector reranking
- Telemetry hooks for recall-latency drift monitoring

## License

You can adapt this repository to your preferred license policy.
