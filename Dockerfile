# ---------------------------------------------------------------------------
# Stage 1: build the React frontend
# ---------------------------------------------------------------------------
FROM node:20 AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime
#
# One image, three roles. The api, the Celery worker and Beat all run from this
# same image with different commands, so they cannot disagree about how a
# document is chunked or which embedding model built the index. Compose picks
# the role.
#
# The index is no longer built here. It used to be: `RUN python -m app.ingest`
# baked a Chroma directory into the image, which made changing a document a
# rebuild and a redeploy, and meant a GEMINI_API_KEY had to be present at build
# time as a BuildKit secret. The index now lives in Postgres and is rebuilt by
# the reindex task. See INFRA-MIGRATION.md.
# ---------------------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

# curl is here for the container healthcheck and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend code and docs land at /app/app and /app/docs, so the package's
# relative paths (docs/, static/) resolve under /app.
COPY backend/app ./app
COPY backend/docs ./docs
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini

# Frontend build output served by FastAPI.
COPY --from=frontend /frontend/dist ./static

# Nothing here runs as root. The app writes nothing to disk now that the index
# is in Postgres, so it does not need to.
# state/ exists and is owned by the app user so that a named volume mounted
# over it inherits that ownership instead of arriving root-owned. Celery Beat
# writes its schedule there.
RUN useradd --create-home --uid 10001 docsentry     && mkdir -p /home/docsentry/state     && chown -R docsentry:docsentry /app /home/docsentry
USER docsentry

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
