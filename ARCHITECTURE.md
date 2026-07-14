# DocSentry — Document Q&A with Guardrails

Portfolio demo #2. A RAG chatbot over a fictional company handbook that answers with citations, refuses when the answer is not in the documents, and blocks prompt-injection attempts. Single Docker deployable for Hugging Face Spaces.

## Stack

- Python 3.11, FastAPI, LangChain (text splitting + retrieval plumbing)
- ChromaDB embedded, persisted to ./chroma_index, PRE-BUILT at Docker build time
- Gemini free tier: `gemini-flash-lite-latest` for answers (env GEMINI_MODEL overrides), `gemini-embedding-001` for embeddings (env GEMINI_EMBED_MODEL overrides; if unavailable fall back to `text-embedding-004`)
- React + Vite chat UI, built to static files, served by FastAPI
- Docker multi-stage: node build of frontend, python runtime, ingest run during build so the Space boots with a ready index

## Layout

```
docsentry/
  backend/
    app/main.py         FastAPI: /api/chat, /api/health, /api/sources, static mount
    app/rag.py          retrieval + Gemini answer generation
    app/guardrails.py   input validation + injection detection + grounding enforcement
    app/ingest.py       chunk + embed docs/ into chroma_index (runnable standalone)
    app/schemas.py      pydantic models
    docs/               fictional "Meridian Outfitters" company handbook (markdown)
    requirements.txt
  frontend/             Vite + React + TypeScript chat UI
  Dockerfile
  README.md
```

## API contract (frozen — both sides build to this)

POST /api/chat
  request  { "question": string (1..500 chars) }
  response {
    "answer": string,
    "citations": [ { "source": string, "section": string, "snippet": string } ],
    "refused": boolean,          // true when answer not grounded in docs
    "flags": string[],           // e.g. ["prompt_injection_suspected", "off_topic"]
    "latencyMs": number
  }

GET /api/health  -> { "status": "ok", "documents": number, "chunks": number }
GET /api/sources -> { "sources": [ { "source": string, "title": string, "sections": number } ] }

## Guardrails (the selling point)

1. Input validation: length caps, strip control chars, reject empty/oversized.
2. Injection detection: pattern layer (ignore-instructions phrasing, role-play requests, system-prompt fishing) BEFORE any LLM call; suspected injections get flagged, question still answered from docs only, never obeyed.
3. Question wrapped in <user_question> delimiters; system prompt states it is untrusted data.
4. Grounding enforcement: LLM must return structured JSON { answer, citedChunkIds, confident }. If no chunks cited, confidence low, or retrieval scores below threshold: respond with a polite refusal ("not covered in the documentation") and refused=true. Never answer from general knowledge.
5. Rate limiting: simple in-memory limiter per IP (slowapi or hand-rolled), 20 req/min.

## Demo content

Fictional company "Meridian Outfitters" (outdoor gear retailer): employee handbook (leave, remote work, expenses), product FAQ (returns, warranty, shipping), refund policy, security policy. 4-6 markdown files, realistic, specific numbers (14-day returns, 2-year warranty) so answers are verifiably grounded.

## Env

GEMINI_API_KEY (required), GEMINI_MODEL, GEMINI_EMBED_MODEL, PORT (HF Spaces sets 7860).

## Deploy

Hugging Face Spaces, Docker SDK. Dockerfile EXPOSE 7860, uvicorn on 0.0.0.0:7860. Index built during image build so no cold ingest.
