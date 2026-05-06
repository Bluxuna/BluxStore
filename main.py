"""Demo script for Blux Search."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np

from blux_search import BluxEngine


def init_db(db_path: Path) -> None:
    """Initialize SQLite schema for corpus data and embedding cache."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_data (
                text TEXT PRIMARY KEY
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_cache (
                model_name TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (model_name, text),
                FOREIGN KEY (text) REFERENCES corpus_data(text) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_cache_model ON corpus_cache(model_name)"
        )
        conn.commit()


def create_corpus_rows(db_path: Path, texts: list[str]) -> int:
    """Insert new corpus texts; ignores duplicates."""
    if not texts:
        return 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.executemany(
            "INSERT OR IGNORE INTO corpus_data (text) VALUES (?)",
            [(text,) for text in texts],
        )
        conn.commit()
        return cur.rowcount if cur.rowcount is not None else 0


def read_corpus_rows(db_path: Path) -> list[str]:
    """Read all corpus texts in deterministic order."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT text FROM corpus_data ORDER BY rowid").fetchall()
        return [row[0] for row in rows]


def update_corpus_row(db_path: Path, old_text: str, new_text: str) -> bool:
    """Update one corpus text and invalidate stale cached embeddings."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.execute(
            "UPDATE corpus_data SET text = ? WHERE text = ?",
            (new_text, old_text),
        )
        # Any embeddings tied to old/new text are potentially stale after content change.
        conn.execute("DELETE FROM corpus_cache WHERE text IN (?, ?)", (old_text, new_text))
        conn.commit()
        return (cur.rowcount or 0) > 0


def delete_corpus_row(db_path: Path, text: str) -> bool:
    """Delete one corpus text (cache rows cascade via FK)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.execute("DELETE FROM corpus_data WHERE text = ?", (text,))
        conn.commit()
        return (cur.rowcount or 0) > 0


def load_or_create_corpus_embeddings(
    engine: BluxEngine,
    corpus: list[str],
    model_name: str,
    db_path: Path,
) -> np.ndarray:
    """Load cached embeddings for corpus texts; embed only missing texts."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("SELECT 1 FROM corpus_data LIMIT 1")

        unique_texts = list(dict.fromkeys(corpus))
        placeholders = ",".join("?" for _ in unique_texts)

        cached: dict[str, np.ndarray] = {}
        if unique_texts:
            rows = conn.execute(
                f"""
                SELECT text, embedding
                FROM corpus_cache
                WHERE model_name = ? AND text IN ({placeholders})
                """,
                [model_name, *unique_texts],
            ).fetchall()
            for text, emb_blob in rows:
                cached[text] = np.frombuffer(emb_blob, dtype=np.float32)

        missing_texts = [text for text in unique_texts if text not in cached]
        if missing_texts:
            missing_embeddings = engine.encode_texts(missing_texts).astype(np.float32, copy=False)
            conn.executemany(
                "INSERT OR REPLACE INTO corpus_cache (model_name, text, embedding) VALUES (?, ?, ?)",
                [
                    (model_name, text, emb.tobytes())
                    for text, emb in zip(missing_texts, missing_embeddings, strict=True)
                ],
            )
            for text, emb in zip(missing_texts, missing_embeddings, strict=True):
                cached[text] = emb

        conn.commit()
        return np.vstack([cached[text] for text in corpus]).astype(np.float32, copy=False)


def main() -> None:
    """Run a small end-to-end demo."""
    corpus = [
        "Distributed systems design patterns",
        "Transformer architectures for NLP",
        "Vector databases and indexing strategies",
        "Approximate nearest neighbor search basics",
        "Microservices observability and tracing",
        "Cosine similarity and embedding spaces",
        "მწვანე ხეები და ტყეები",
    ]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    db_path = Path("./blux_store/embeddings.sqlite3")
    init_db(db_path)
    create_corpus_rows(db_path, corpus)
    corpus = read_corpus_rows(db_path)
    engine = BluxEngine(metric="cosine", normalize=True)
    corpus_embeddings = load_or_create_corpus_embeddings(engine, corpus, model_name, db_path)

    metadata = [{"id": idx, "text": text} for idx, text in enumerate(corpus)]
    engine.add(corpus_embeddings, metadata=metadata)
    engine.build_ivf(n_clusters=2, max_iter=50)

    query_text = "რა არის ბუნებაში კაი?"
    query_embedding = engine.encode_texts([query_text])[0]

    print("\\nFlat search (exact):")
    for rank, result in enumerate(engine.search_flat(query_embedding, k=3), start=1):
        print(f"{rank}. idx={result.idx}, score={result.score:.4f}, text={result.metadata['text']}")

    print("\\nIVF search (approx):")
    for rank, result in enumerate(engine.search_ivf(query_embedding, k=3, n_probe=1), start=1):
        print(f"{rank}. idx={result.idx}, score={result.score:.4f}, text={result.metadata['text']}")

    engine.save("./blux_store")
    restored = BluxEngine.load("./blux_store")
    restored_results = restored.search_flat(query_embedding, k=1)

    print("\\nReload check:")
    print(f"Top result after reload: idx={restored_results[0].idx}, score={restored_results[0].score:.4f}")


if __name__ == "__main__":
    main()
