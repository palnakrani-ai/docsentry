"""Sentry wiring.

Inert unless `SENTRY_DSN` is set, in the same way LangSmith tracing is inert
unless `LANGSMITH_TRACING` is exactly "true". A demo that anyone can clone and
run must not require an error-tracking account.

`send_default_pii` stays off. Every request to this service carries a user's
question in the body, and questions are exactly the kind of thing that turns out
to contain something private. The `events` table already records them, in a
database with row level security on it, which is a deliberate place to keep them.
An error tracker is not.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def init_sentry() -> bool:
    """Start Sentry if a DSN is configured. Returns whether it started.

    Never raises. Error reporting failing to start is not a reason for the
    service to fail to start.
    """
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
            release=os.environ.get("SENTRY_RELEASE"),
            # Sampled rather than 1.0: this runs on one small box and tracing
            # every request costs latency on the path being measured.
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            send_default_pii=False,
            before_send=_scrub,
        )
        return True
    except Exception:
        log.warning("Sentry failed to initialise; continuing without it", exc_info=True)
        return False


def _scrub(event: dict, hint: dict) -> dict:
    """Drop request bodies before an event leaves the process.

    The body of a /api/chat request is the user's question. Sentry captures
    request data by default for unhandled exceptions, and the one field this
    service must never ship to a third party is the one it is built around.
    """
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("data", None)
        request.pop("cookies", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for key in list(headers):
                if key.lower() in {"authorization", "cookie", "x-api-key"}:
                    headers.pop(key)
    return event
