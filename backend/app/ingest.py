"""Ingest docs/*.md into a persistent Chroma index via LangChain.

Splits each markdown file per "## " section, then recursively into ~800-char
chunks with overlap, embeds with Gemini through LangChain's embedding
interface, and stores everything in a Chroma collection with
{source, section, title} metadata.

Runnable standalone:  python -m app.ingest
Idempotent: the collection is dropped and recreated on every run.
"""

import os
import re
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths resolve relative to this package so they work both locally
# (backend/ as cwd) and inside the Docker image (/app).
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_index"

COLLECTION_NAME = "docsentry"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Gemini's free embedding tier allows 100 requests per minute, and each chunk
# costs one request. 50 chunks per batch with a 40 second cooldown holds the
# rate at roughly 75 per minute, leaving headroom for a retry inside the window.
EMBED_BATCH_SIZE = 50
EMBED_COOLDOWN_SECONDS = 40
EMBED_MAX_RETRIES = 5

PRIMARY_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
FALLBACK_EMBED_MODEL = "text-embedding-004"


def _api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return api_key


def _qualified(model: str) -> str:
    """langchain-google-genai expects the 'models/' prefix."""
    return model if model.startswith("models/") else f"models/{model}"


def build_embeddings(model: str = PRIMARY_EMBED_MODEL) -> GoogleGenerativeAIEmbeddings:
    """Embeddings for both indexing and querying.

    task_type is deliberately left unset: the class then uses
    retrieval_document for embed_documents and retrieval_query for
    embed_query, which is the asymmetry this corpus wants.
    """
    return GoogleGenerativeAIEmbeddings(
        model=_qualified(model),
        google_api_key=_api_key(),
    )


def _is_model_missing(exc: Exception) -> bool:
    """True when Gemini rejected the embedding model as unavailable (404)."""
    if getattr(exc, "code", None) == 404:
        return True
    return "404" in str(exc) or "not found" in str(exc).lower()


def _split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a markdown document into (title, [(section_heading, body), ...]).

    The title comes from the first "# " heading. Each "## " heading starts a
    new section; any preamble before the first "## " becomes an "Overview"
    section.
    """
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []

    preamble = parts[0]
    preamble = re.sub(r"^#\s+.+$", "", preamble, flags=re.MULTILINE).strip()
    if preamble:
        sections.append(("Overview", preamble))

    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            sections.append((heading, body))
    return title, sections


def build_chunks() -> tuple[list[str], list[Document]]:
    """Return (ids, documents) for all markdown files in docs/."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    ids: list[str] = []
    documents: list[Document] = []

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in {DOCS_DIR}")

    for path in md_files:
        source = path.name
        title, sections = _split_sections(path.read_text(encoding="utf-8"))
        for section_heading, body in sections:
            # Keep the heading with the text so embeddings carry context.
            section_text = f"## {section_heading}\n\n{body}"
            for j, chunk in enumerate(splitter.split_text(section_text)):
                slug = re.sub(r"[^a-z0-9]+", "-", section_heading.lower()).strip("-")
                ids.append(f"{path.stem}::{slug}::{j}")
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "section": section_heading,
                            "title": title,
                        },
                    )
                )
    return ids, documents


def _is_rate_limited(exc: Exception) -> bool:
    """True when Gemini rejected the call for exceeding a quota (429)."""
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text


def _retry_after(exc: Exception) -> int:
    """Seconds Gemini asked us to wait, from its retry_delay field."""
    match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(exc))
    return int(match.group(1)) if match else EMBED_COOLDOWN_SECONDS


def _embed_batch(store: Chroma, batch: list[Document], batch_ids: list[str]) -> None:
    """Add one batch, retrying on quota errors with the delay Gemini asks for."""
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            store.add_documents(documents=batch, ids=batch_ids)
            return
        except Exception as exc:
            if not _is_rate_limited(exc) or attempt == EMBED_MAX_RETRIES:
                raise
            wait = _retry_after(exc) + 5
            print(
                f"[ingest] rate limited, waiting {wait}s "
                f"(attempt {attempt}/{EMBED_MAX_RETRIES - 1})",
                file=sys.stderr,
            )
            time.sleep(wait)


def _write_index(ids: list[str], documents: list[Document], model: str) -> Chroma:
    """Embed and write in paced batches.

    The free embedding tier allows 100 requests per minute and each chunk is
    one request, so a corpus of any size fails if written in a single call.
    Batches are written at a rate that stays under the limit, and a batch that
    is rate limited anyway waits for the delay Gemini names and retries.
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=build_embeddings(model),
        collection_metadata={"hnsw:space": "cosine", "embed_model": model},
    )

    total = len(documents)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        batch_ids = ids[start : start + EMBED_BATCH_SIZE]
        _embed_batch(store, batch, batch_ids)
        done = start + len(batch)
        print(f"[ingest] embedded {done}/{total}")
        if done < total:
            time.sleep(EMBED_COOLDOWN_SECONDS)

    return store


def main() -> None:
    ids, documents = build_chunks()
    n_docs = len({d.metadata["source"] for d in documents})
    print(f"[ingest] {n_docs} documents -> {len(documents)} chunks")

    # Drop the old collection so re-running is idempotent and every vector in
    # the index comes from a single embedding model.
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
        ).delete_collection()
    except Exception:
        pass  # first run, nothing to delete

    model = PRIMARY_EMBED_MODEL
    try:
        store = _write_index(ids, documents, model)
    except Exception as exc:
        if not _is_model_missing(exc) or model == FALLBACK_EMBED_MODEL:
            raise
        print(
            f"[ingest] model {model} unavailable (404); "
            f"retrying with {FALLBACK_EMBED_MODEL}",
            file=sys.stderr,
        )
        model = FALLBACK_EMBED_MODEL
        store = _write_index(ids, documents, model)

    print(f"[ingest] embedded with {model}")
    written = len(store.get(include=[])["ids"])
    print(f"[ingest] wrote {written} chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
