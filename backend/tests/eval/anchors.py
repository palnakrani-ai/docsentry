"""Resolve dataset labels to chunks, whatever the current chunking is.

Labels cite an exact sentence from a source document rather than a chunk id.
Chunk ids encode the chunking strategy (`loyalty-program::tiers::0`), so a
label written against them dies the moment CHUNK_SIZE changes -- which the
ablation deliberately does four times. Anchoring to text survives that, and
lets the same 150 labels score every retrieval configuration.

A retrieved chunk supports a label when it contains one of the label's anchors.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace so a chunk boundary inside a sentence still matches.

    The splitter breaks on newlines, so an anchor spanning a line break appears
    with different whitespace in the document than in the chunk. Normalizing
    both sides makes the comparison about words rather than layout.
    """
    return _WS.sub(" ", text).strip()


@lru_cache(maxsize=None)
def document_text(source_doc: str) -> str:
    """Normalized full text of one corpus document."""
    path = DOCS_DIR / source_doc
    if not path.is_file():
        raise FileNotFoundError(f"corpus document not found: {source_doc}")
    return normalize(path.read_text(encoding="utf-8"))


def anchor_in_document(anchor: str, source_docs: list[str]) -> bool:
    """True when the anchor appears verbatim in any of the named documents."""
    needle = normalize(anchor)
    return any(needle in document_text(doc) for doc in source_docs)


def chunk_supports(chunk_text: str, anchors: list[str]) -> bool:
    """True when a retrieved chunk carries at least one of the label's anchors.

    A chunk holding only part of an anchor does not count. Partial credit here
    would inflate context recall precisely where retrieval is weakest, which is
    the measurement the ablation exists to make.
    """
    haystack = normalize(chunk_text)
    return any(normalize(a) in haystack for a in anchors)


def supporting_chunk_count(chunks: list[str], anchors: list[str]) -> int:
    """How many of the retrieved chunks support the label."""
    return sum(1 for chunk in chunks if chunk_supports(chunk, anchors))


def anchors_found(chunks: list[str], anchors: list[str]) -> list[str]:
    """Which anchors are present somewhere in the retrieved set."""
    haystacks = [normalize(c) for c in chunks]
    return [a for a in anchors if any(normalize(a) in h for h in haystacks)]
