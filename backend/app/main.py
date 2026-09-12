"""DocSentry FastAPI app: chat, health, sources, static frontend."""

import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import db, guardrails, rag
from .ingest import BASE_DIR
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceInfo,
    SourcesResponse,
)

STATIC_DIR = BASE_DIR / "static"

RATE_LIMIT = 20  # requests
RATE_WINDOW = 60.0  # seconds

app = FastAPI(title="DocSentry", version="1.0.0")

# The frontend is served by this same app, so same-origin requests need no
# CORS at all. The localhost entries below only support Vite dev mode.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# In-memory per-IP rate limiter (hand-rolled sliding window, no redis)
# ---------------------------------------------------------------------------

_hits: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    now = time.monotonic()
    window = _hits[_client_ip(request)]
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded: 20 requests per minute. Try again shortly.",
        )
    window.append(now)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest, request: Request, background: BackgroundTasks
) -> ChatResponse:
    check_rate_limit(request)
    started = time.perf_counter()

    cleaned, error = guardrails.validate_question(body.question)
    if error:
        raise HTTPException(status_code=400, detail=error)

    flags = guardrails.detect_injection(cleaned)

    try:
        result = rag.answer_question(cleaned)
    except HTTPException:
        raise
    except Exception as exc:  # index missing, Postgres down, Gemini outage, etc.
        raise HTTPException(
            status_code=503, detail=f"Answer generation failed: {exc}"
        ) from exc

    for flag in result.flags:
        if flag not in flags:
            flags.append(flag)

    latency_ms = int((time.perf_counter() - started) * 1000)

    # The audit row is queued rather than written here. It runs after the
    # response is sent, so a slow or unreachable database costs the user
    # nothing, and db.record_event swallows its own failures for the same
    # reason. A request that was answered correctly must not turn into an error
    # because the record of it could not be saved.
    background.add_task(
        db.record_event,
        question=cleaned,
        answered=not result.refused,
        refusal_reason=_refusal_reason(result, flags),
        citations=[c.model_dump() for c in result.citations],
        flags=flags,
        top_score=result.top_score,
        latency_ms=latency_ms,
    )

    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        refused=result.refused,
        flags=flags,
        latencyMs=latency_ms,
    )


def _refusal_reason(result: rag.RagResult, flags: list[str]) -> str | None:
    """Why this request refused, in one word, or None if it answered.

    Ordered most specific first. model_unavailable matters most because it is
    the one reason that is an outage rather than a decision, and the M2
    evaluation work turned on being able to tell those apart after the fact.
    """
    if not result.refused:
        return None
    for flag in (guardrails.FLAG_MODEL_UNAVAILABLE, guardrails.FLAG_OFF_TOPIC):
        if flag in flags:
            return flag
    return "ungrounded"


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    documents = 0
    chunks = 0
    try:
        documents, chunks = rag.collection_stats()
    except Exception:
        pass  # index not built yet; report zeros rather than crash
    return HealthResponse(status="ok", documents=documents, chunks=chunks)


@app.get("/api/sources", response_model=SourcesResponse)
def sources() -> SourcesResponse:
    infos: list[SourceInfo] = []
    try:
        by_source: dict[str, dict] = {}
        for meta in rag.collection_metadatas():
            source = meta.get("source", "unknown")
            entry = by_source.setdefault(
                source, {"title": meta.get("title", source), "sections": set()}
            )
            entry["sections"].add(meta.get("section", ""))
        infos = [
            SourceInfo(source=src, title=data["title"], sections=len(data["sections"]))
            for src, data in sorted(by_source.items())
        ]
    except Exception:
        pass  # index not built yet
    return SourcesResponse(sources=infos)


# ---------------------------------------------------------------------------
# Static frontend (built by the Docker frontend stage into backend/static)
# ---------------------------------------------------------------------------


class SpaStaticFiles(StaticFiles):
    """Serve the built SPA; unknown paths fall back to index.html so
    client-side routing works on refresh/deep links."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return FileResponse(Path(self.directory) / "index.html")
            raise


if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
    app.mount("/", SpaStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    def frontend_placeholder() -> PlainTextResponse:
        return PlainTextResponse(
            "DocSentry API is running, but the frontend has not been built. "
            "Build the frontend into backend/static or use the API directly "
            "at /api/chat, /api/health, /api/sources."
        )
