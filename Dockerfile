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
# Stage 2: Python runtime with a pre-built Chroma index
# ---------------------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend code and docs land at /app/app and /app/docs, so the package's
# relative paths (docs/, chroma_index/, static/) all resolve under /app.
COPY backend/app ./app
COPY backend/docs ./docs

# Frontend build output served by FastAPI.
COPY --from=frontend /frontend/dist ./static

# Build the vector index at image build time so the Space boots ready.
# On Hugging Face Spaces, add GEMINI_API_KEY as a BUILD-TIME secret
# (Settings -> Variables and secrets -> "Secret", exposed to the builder);
# it is used here once and is not baked into the final image layers as env.
# Hugging Face Spaces exposes secrets as BuildKit secret mounts; plain
# docker builds can keep passing --build-arg GEMINI_API_KEY instead.
ARG GEMINI_API_KEY
RUN --mount=type=secret,id=GEMINI_API_KEY,mode=0444,required=false \
    sh -c 'if [ -f /run/secrets/GEMINI_API_KEY ]; then export GEMINI_API_KEY="$(cat /run/secrets/GEMINI_API_KEY)"; fi; python -m app.ingest'

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
