"""Celery application and schedule.

What this exists for, stated plainly: the vector index needs to be rebuildable
without redeploying the image. That is one task. A broker, a worker and a
scheduler to run one task is a lot of moving parts, and the honest reason for
them is that rebuilding the index was previously a `docker build`, not that the
request path needs a queue. DocSentry answers one synchronous question in about
two seconds and nothing about that changes here.

Keeping the API and the worker in the same image means they cannot disagree
about how a document is chunked or which embedding model built the index.
"""

import os

from celery import Celery
from celery.schedules import crontab

# Redis is reachable as `redis` inside the Compose network. The default is the
# Compose hostname rather than localhost, because the container case is the one
# that must work without configuration; a developer running a worker on the host
# can set the variable.
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)

# `include` rather than autodiscover_tasks plus an import at the bottom of this
# module. The tasks are declared on this app object directly (see tasks.py), so
# any process that imports app.tasks gets the configured app rather than
# Celery's unconfigured default. With @shared_task the API could dispatch
# through a default app with no broker or result backend, and the failure showed
# up only when something asked for a result.
celery_app = Celery(
    "docsentry",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # An ingest walks the whole corpus and is paced around the embedding rate
    # limit, so it can legitimately run for many minutes. The default time limit
    # would kill it partway and leave a half-written collection.
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60,
    # One ingest at a time. Two concurrent rebuilds would race on the same
    # collection and compete for the same embedding quota, which is how the
    # latency report got poisoned during M2.
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_app.conf.beat_schedule = {
    "reindex-weekly": {
        "task": "app.tasks.reindex",
        # Sunday 03:00 UTC. The corpus is a fixed set of policy documents that
        # changes when someone edits them, so this is a safety net for drift
        # rather than a pipeline. It also keeps the Supabase project from
        # idling out, which pauses free-tier projects after about a week.
        "schedule": crontab(hour=3, minute=0, day_of_week=0),
    },
}

