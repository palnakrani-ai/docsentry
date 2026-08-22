---
title: DocSentry
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# DocSentry

A RAG document Q&A demo with guardrails. It answers questions about a fictional outdoor gear retailer, Meridian Outfitters, strictly from the company's handbook and policy documents. Every answer carries citations, anything outside the docs gets a polite refusal, and prompt injection attempts are flagged and ignored. Ships as a single Docker image for Hugging Face Spaces.

Stack: FastAPI, LangChain 1.0 (LCEL chain, `langchain-google-genai`, `langchain-chroma`), Chroma embedded with the index built at Docker build time, Gemini for embeddings and answers, React + Vite frontend served as static files by FastAPI. Optional LangSmith tracing.

The chain is `retrieve -> grounding gate -> generate -> verify`. LangChain handles composition, retrieval and structured output. The two checks that decide whether an answer is allowed to reach the user stay in plain Python inside Runnables: an instruction telling a model to be careful is not the same thing as a guarantee.

## Guardrails

1. Input validation: 1 to 500 character questions, control characters stripped.
2. Injection detection before any LLM call: ignore-instructions phrasing, system prompt fishing, role-play requests, raw document exfiltration, jailbreak phrasings. Flagged questions are still answered from the docs only and the instructions are never obeyed.
3. The question is wrapped in `<user_question>` tags and the system prompt declares it untrusted data.
4. Grounding enforced in code: the model must return structured JSON with cited chunk ids and a confidence bit. No citations, low confidence, or weak retrieval scores all produce a refusal with `refused: true`. The bot never answers from general knowledge.
5. In-memory rate limiting, 20 requests per minute per IP.

## API

- `POST /api/chat` with `{ "question": "..." }` returns `{ answer, citations, refused, flags, latencyMs }`
- `GET /api/health` returns `{ status, documents, chunks }`
- `GET /api/sources` returns the indexed documents and their section counts

## Run locally

Backend (Python 3.11+):

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # PowerShell: $env:GEMINI_API_KEY="..."
python -m app.ingest                  # builds backend/chroma_index
uvicorn app.main:app --reload --port 7860
```

Frontend (optional, for the chat UI):

```bash
cd frontend
npm ci
npm run build                         # then copy dist/ to backend/static/
npm run dev                           # or run the Vite dev server instead
```

Without a built frontend, the root URL serves a plain status message and the API remains fully usable.

Env vars: `GEMINI_API_KEY` (required), `GEMINI_MODEL` (default `gemini-flash-lite-latest`), `GEMINI_EMBED_MODEL` (default `gemini-embedding-001`, auto-falls back to `text-embedding-004` on 404), `PORT` (Spaces sets 7860).

Tracing is optional and off unless you set all three: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. With them set, every request shows up in LangSmith as a four-step trace with token counts, per-step latency and the retrieved chunks.

## Deploy to Hugging Face Spaces

1. Create a Space with the Docker SDK.
2. In Space Settings, under Variables and secrets, add `GEMINI_API_KEY` as a Secret. Spaces exposes secrets to the Docker build, which this image uses to embed the docs during the build so the Space boots with a ready index.
3. Push this repo to the Space. The Dockerfile builds the frontend, installs the backend, runs ingestion, and starts uvicorn on port 7860.
