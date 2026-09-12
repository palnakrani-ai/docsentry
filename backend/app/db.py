"""Postgres access: the connection, the audit trail, and collection queries.

Two things live in this database. The vectors, written and read by
langchain-postgres, and an `events` row per answered question. Keeping them
together is the point: a dashboard can ask how often the system refused and
what the retrieval scores looked like when it did, without joining across two
stores that were never designed to be joined.

Chroma exposed a `.get()` that returned every record's metadata, which the
health and sources routes used for their counts. PGVector has no equivalent,
so those counts are plain SQL here against the two tables langchain-postgres
creates (`langchain_pg_collection` and `langchain_pg_embedding`).
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

# Local dev default points at docker-compose.dev.yml. Supabase is the same
# variable with a different value, which is the whole reason it is a variable.
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://docsentry:docsentry@localhost:5433/docsentry"
)

COLLECTION_NAME = "docsentry"

_engine: Engine | None = None


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    """One pooled engine per process.

    pool_pre_ping is set because the hosted case is a managed Postgres behind
    a pooler that drops idle connections without telling anyone. Without it the
    first request after a quiet period fails on a stale connection, which looks
    exactly like an outage and is not one.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            database_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
    return _engine


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def record_event(
    *,
    question: str,
    answered: bool,
    refusal_reason: str | None,
    citations: list[dict[str, Any]],
    flags: list[str],
    top_score: float | None,
    latency_ms: int,
) -> None:
    """Write one row describing an answered question.

    Never raises. This is called off the response path and a failure to record
    what happened must not change what happened. The request has already been
    answered by the time this runs; losing an audit row is a worse outcome than
    nothing only in the sense that it is silent, which is why it logs.
    """
    import json
    import logging

    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO events (
                        question, answered, refusal_reason,
                        citations, flags, top_score, latency_ms
                    ) VALUES (
                        :question, :answered, :refusal_reason,
                        CAST(:citations AS jsonb), :flags, :top_score, :latency_ms
                    )
                    """
                ),
                {
                    "question": question,
                    "answered": answered,
                    "refusal_reason": refusal_reason,
                    "citations": json.dumps(citations),
                    "flags": flags,
                    "top_score": top_score,
                    "latency_ms": latency_ms,
                },
            )
    except Exception:
        logging.getLogger(__name__).warning(
            "failed to record event; the request itself was unaffected",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Collection queries, replacing Chroma's .get()
# ---------------------------------------------------------------------------


def collection_cmetadata(collection_name: str = COLLECTION_NAME) -> dict[str, Any]:
    """Collection-level metadata, which is where the embedding model is recorded.

    Returns an empty dict when the collection does not exist yet, so a caller
    can tell "not ingested" apart from "ingested with unknown settings" by
    checking the keys rather than catching an exception. That includes the case
    where langchain-postgres has not created its tables at all, which is what a
    database that has had the migration but never an ingest looks like.
    """
    try:
        return _collection_cmetadata(collection_name)
    except ProgrammingError:
        return {}


def _collection_cmetadata(collection_name: str) -> dict[str, Any]:
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT cmetadata FROM langchain_pg_collection WHERE name = :name"),
            {"name": collection_name},
        ).first()
    if row is None or row[0] is None:
        return {}
    return dict(row[0])


def collection_stats(collection_name: str = COLLECTION_NAME) -> tuple[int, int]:
    """Return (document_count, chunk_count) for the health and sources routes.

    An index that has never been built is reported as zero chunks rather than
    raised as an error. langchain-postgres creates its tables on first write, so
    querying them before any ingest raises UndefinedTable, and letting that
    propagate makes the health check say "database: error" when the database is
    perfectly fine and the fix is to run the ingest. Naming the wrong dependency
    is worse than naming none: it sends whoever is on call to the wrong place.
    """
    try:
        return _collection_stats(collection_name)
    except ProgrammingError:
        return 0, 0


def _collection_stats(collection_name: str) -> tuple[int, int]:
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT COUNT(DISTINCT e.cmetadata ->> 'source'), COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :name
                """
            ),
            {"name": collection_name},
        ).first()
    if row is None:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def collection_metadatas(collection_name: str = COLLECTION_NAME) -> list[dict]:
    """Per-chunk metadata for every chunk in the collection.

    Empty when the index has never been built, for the same reason as above.
    """
    try:
        return _collection_metadatas(collection_name)
    except ProgrammingError:
        return []


def _collection_metadatas(collection_name: str) -> list[dict]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :name
                """
            ),
            {"name": collection_name},
        ).all()
    return [dict(r[0]) for r in rows if r[0]]
