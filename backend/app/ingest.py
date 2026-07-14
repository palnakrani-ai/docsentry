"""Ingest docs/*.md into a persistent Chroma index.

Splits each markdown file per "## " section, then recursively into ~800-char
chunks with overlap, embeds via Gemini, and stores everything in a Chroma
collection with {source, section, title} metadata.

Runnable standalone:  python -m app.ingest
Idempotent: the collection is dropped and recreated on every run.
"""

import os
import re
import sys
from pathlib import Path

import chromadb
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths resolve relative to this package so they work both locally
# (backend/ as cwd) and inside the Docker image (/app).
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_index"

COLLECTION_NAME = "docsentry"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_BATCH_SIZE = 50

PRIMARY_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
FALLBACK_EMBED_MODEL = "text-embedding-004"


def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def _embed_batch(
    client: genai.Client, model: str, texts: list[str], task_type: str
) -> list[list[float]]:
    result = client.models.embed_content(
        model=model,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return [list(e.values) for e in result.embeddings]


def embed_texts(
    client: genai.Client, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> tuple[list[list[float]], str]:
    """Embed texts, falling back to FALLBACK_EMBED_MODEL if the primary 404s.

    Returns (embeddings, model_used).
    """
    model = PRIMARY_EMBED_MODEL
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        try:
            embeddings.extend(_embed_batch(client, model, batch, task_type))
        except ClientError as exc:
            if getattr(exc, "code", None) == 404 and model != FALLBACK_EMBED_MODEL:
                print(
                    f"[ingest] model {model} unavailable (404); "
                    f"retrying with {FALLBACK_EMBED_MODEL}",
                    file=sys.stderr,
                )
                model = FALLBACK_EMBED_MODEL
                # Restart from scratch so all vectors come from one model.
                return embed_texts_with_model(client, texts, model, task_type)
            raise
    return embeddings, model


def embed_texts_with_model(
    client: genai.Client, texts: list[str], model: str, task_type: str
) -> tuple[list[list[float]], str]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        embeddings.extend(_embed_batch(client, model, batch, task_type))
    return embeddings, model


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


def build_chunks() -> tuple[list[str], list[str], list[dict]]:
    """Return (ids, documents, metadatas) for all markdown files in docs/."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

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
                documents.append(chunk)
                metadatas.append(
                    {"source": source, "section": section_heading, "title": title}
                )
    return ids, documents, metadatas


def main() -> None:
    ids, documents, metadatas = build_chunks()
    n_docs = len({m["source"] for m in metadatas})
    print(f"[ingest] {n_docs} documents -> {len(documents)} chunks")

    client = _gemini_client()
    embeddings, model_used = embed_texts(client, documents)
    print(f"[ingest] embedded with {model_used}")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run, nothing to delete
    collection = chroma.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embed_model": model_used},
    )
    collection.add(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )
    print(f"[ingest] wrote {collection.count()} chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
