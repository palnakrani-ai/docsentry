"""Disk-cached embeddings for the ablation.

Four configurations over two chunkings and 150 questions would re-embed several
hundred texts on every run, against a free tier that allows 100 embedding
requests per minute. Without a cache the ablation takes tens of minutes and is
mostly spent re-deriving vectors that have not changed.

The cache key is the SHA-256 of the text plus the model name, so editing a
document or switching models invalidates only what actually changed. Cached
vectors are content-addressed rather than position-addressed, which means a
re-chunk that produces identical text costs nothing.

Requests are paced to the same limit the ingest respects.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

CACHE_DIR = Path(__file__).resolve().parent / ".embedding_cache"
BATCH_SIZE = 50
COOLDOWN_SECONDS = 40
MAX_RETRIES = 5


def _key(text: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()
    return digest


def _cache_path(key: str) -> Path:
    # Two-level fan-out keeps any one directory small enough to list quickly.
    return CACHE_DIR / key[:2] / f"{key}.json"


def _read(key: str) -> list[float] | None:
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # a corrupt entry is just a cache miss


def _write(key: str, vector: list[float]) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vector), encoding="utf-8")


def _model_name(embeddings) -> str:
    return getattr(embeddings, "model", embeddings.__class__.__name__)


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text


def _embed_documents_paced(texts: list[str], embeddings) -> list[list[float]]:
    """Embed uncached texts in batches that stay under the rate limit."""
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out.extend(embeddings.embed_documents(batch))
                break
            except Exception as exc:
                if not _is_rate_limited(exc) or attempt == MAX_RETRIES:
                    raise
                print(
                    f"[embed-cache] rate limited, waiting {COOLDOWN_SECONDS}s",
                    file=sys.stderr,
                )
                time.sleep(COOLDOWN_SECONDS)
        if start + BATCH_SIZE < len(texts):
            time.sleep(COOLDOWN_SECONDS)
    return out


def embed_texts(texts: list[str], embeddings) -> list[list[float]]:
    """Vectors for these texts, embedding only what is not already cached."""
    model = _model_name(embeddings)
    keys = [_key(t, model) for t in texts]
    cached = [_read(k) for k in keys]

    missing_idx = [i for i, vec in enumerate(cached) if vec is None]
    if missing_idx:
        print(
            f"[embed-cache] {len(cached) - len(missing_idx)} hit, "
            f"{len(missing_idx)} to embed",
            file=sys.stderr,
        )
        fresh = _embed_documents_paced([texts[i] for i in missing_idx], embeddings)
        for i, vector in zip(missing_idx, fresh):
            _write(keys[i], vector)
            cached[i] = vector

    return [vec for vec in cached if vec is not None]


def embed_chunks(chunks: list[Document], embeddings) -> list[list[float]]:
    return embed_texts([c.page_content for c in chunks], embeddings)


def embed_query(question: str, embeddings) -> list[float]:
    """Queries are cached too: the same 150 run against four configurations."""
    return embed_queries([question], embeddings)[0]


def embed_queries(questions: list[str], embeddings) -> list[list[float]]:
    """Vectors for a batch of queries, paced and cached.

    Queries go through the same rate-limited path as documents. An earlier
    version called embed_query directly with no pacing, which was fine at the
    smoke-test size and then hit the per-minute limit the moment the full
    question set ran.
    """
    model = _model_name(embeddings)
    keys = [_key(f"query::{q}", model) for q in questions]
    cached = [_read(k) for k in keys]

    missing_idx = [i for i, vec in enumerate(cached) if vec is None]
    if missing_idx:
        print(
            f"[embed-cache] queries: {len(cached) - len(missing_idx)} hit, "
            f"{len(missing_idx)} to embed",
            file=sys.stderr,
        )
        fresh = _embed_documents_paced([questions[i] for i in missing_idx], embeddings)
        for i, vector in zip(missing_idx, fresh):
            _write(keys[i], vector)
            cached[i] = vector

    return [vec for vec in cached if vec is not None]
