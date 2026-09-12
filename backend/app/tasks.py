"""Background tasks.

Only one task matters: rebuilding the vector index. It used to happen at Docker
build time, which made changing a document a redeploy.

The task deliberately does not swallow its own failures. A rebuild that fails
silently leaves the collection in whatever state it reached, and the API would
keep answering from a partial index with a healthy-looking health check. That is
the exact failure this project already learned about the hard way in LeadTriage:
a monitor that watches a symptom the outage does not produce.
"""

from __future__ import annotations

import logging

from .celery_app import celery_app
from .db import collection_stats

log = logging.getLogger(__name__)


@celery_app.task(
    name="app.tasks.reindex",
    bind=True,
    # Embedding calls are rate limited and the ingest already retries inside
    # itself for quota errors. These retries are for the coarser failures:
    # Postgres unreachable, the model 404ing, a container restarting mid-run.
    autoretry_for=(Exception,),
    retry_backoff=60,
    retry_backoff_max=900,
    retry_jitter=True,
    max_retries=3,
)
def reindex(self) -> dict[str, int]:
    """Rebuild the whole vector collection from docs/ and report what landed.

    `ingest.main` drops and recreates the collection, so this is idempotent and
    every vector in the result came from a single embedding model.
    """
    from .ingest import main as ingest_main

    log.info("reindex starting (attempt %s)", self.request.retries + 1)
    ingest_main()

    documents, chunks = collection_stats()
    log.info("reindex complete: %s chunks from %s documents", chunks, documents)

    if chunks == 0:
        # An empty collection means the rebuild produced nothing, and the API
        # would answer every question with a refusal while looking healthy.
        # Better to fail the task and retry than to publish silence.
        raise RuntimeError("reindex produced an empty collection")

    return {"documents": documents, "chunks": chunks}
