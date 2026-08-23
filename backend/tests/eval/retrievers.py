"""The four retrieval configurations the ablation compares.

Each returns the text of the top k chunks for a question. They share one
embedding model and one corpus so the only variable is retrieval strategy.

Configurations:
  naive     fixed-size chunks, dense vector search       (the baseline)
  semantic  section-aware chunks, dense vector search    (what DocSentry ships)
  hybrid    semantic chunks, BM25 + dense, reciprocal rank fusion
  reranked  hybrid, then a cross-encoder-free rerank on term overlap

The reranker is deliberately not a hosted cross-encoder. Adding a paid API call
per query would make the ablation expensive to reproduce, and the milestone
requires every published number to be reproducible by the demo command. A
lexical reranker is weaker than a cross-encoder and the table says so.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, _split_sections

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def naive_chunks() -> list[Document]:
    """Fixed-size splitting over the raw file, ignoring document structure.

    This is the baseline a first-pass RAG build produces: no heading awareness,
    so a chunk can begin mid-sentence in one policy and end inside the next.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": path.name}))
    return docs


@lru_cache(maxsize=1)
def semantic_chunks() -> list[Document]:
    """Section-aware splitting: the strategy the shipped ingest uses.

    Each "## " section is split separately and the heading is prepended, so a
    chunk never spans two policies and always carries its own context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        _title, sections = _split_sections(path.read_text(encoding="utf-8"))
        for heading, body in sections:
            section_text = f"## {heading}\n\n{body}"
            for chunk in splitter.split_text(section_text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": path.name, "section": heading},
                    )
                )
    return docs


# ---------------------------------------------------------------------------
# BM25, implemented locally
# ---------------------------------------------------------------------------

@dataclass
class BM25:
    """Standard BM25 over a fixed corpus.

    Implemented here rather than pulled in as a dependency: it is thirty lines,
    it removes a package from the reproduction path, and the ablation needs the
    scores to be deterministic across machines.
    """

    corpus_tokens: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.n = len(self.corpus_tokens)
        self.doc_len = [len(t) for t in self.corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.freqs = [Counter(t) for t in self.corpus_tokens]

        df: Counter = Counter()
        for tokens in self.corpus_tokens:
            df.update(set(tokens))
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_tokens = _tokens(query)
        out = [0.0] * self.n
        for i, freq in enumerate(self.freqs):
            length = self.doc_len[i] or 1
            total = 0.0
            for term in q_tokens:
                if term not in freq:
                    continue
                tf = freq[term]
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            out[i] = total
        return out


@lru_cache(maxsize=1)
def _semantic_bm25() -> tuple[BM25, list[Document]]:
    docs = semantic_chunks()
    return BM25([_tokens(d.page_content) for d in docs]), docs


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

def retrieve_dense(question: str, chunks: list[Document], k: int, embeddings) -> list[str]:
    """Cosine similarity against pre-embedded chunks."""
    from tests.eval.embedding_cache import embed_chunks, embed_query

    matrix = embed_chunks(chunks, embeddings)
    q = embed_query(question, embeddings)
    sims = [sum(a * b for a, b in zip(q, vec)) for vec in matrix]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [chunks[i].page_content for i in order]


def retrieve_naive(question: str, k: int, embeddings) -> list[str]:
    return retrieve_dense(question, naive_chunks(), k, embeddings)


def retrieve_semantic(question: str, k: int, embeddings) -> list[str]:
    return retrieve_dense(question, semantic_chunks(), k, embeddings)


def retrieve_hybrid(question: str, k: int, embeddings) -> list[str]:
    """Reciprocal rank fusion of BM25 and dense rankings.

    RRF rather than score averaging because BM25 scores and cosine similarities
    are on different scales, and normalising them introduces a tuning knob that
    would need its own justification.
    """
    from tests.eval.embedding_cache import embed_chunks, embed_query

    bm25, docs = _semantic_bm25()
    lexical = bm25.scores(question)
    lex_order = sorted(range(len(docs)), key=lambda i: lexical[i], reverse=True)

    matrix = embed_chunks(docs, embeddings)
    q = embed_query(question, embeddings)
    dense = [sum(a * b for a, b in zip(q, vec)) for vec in matrix]
    dense_order = sorted(range(len(docs)), key=lambda i: dense[i], reverse=True)

    C = 60  # the constant from the original RRF paper
    fused: dict[int, float] = {}
    for rank, idx in enumerate(lex_order[:50], start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (C + rank)
    for rank, idx in enumerate(dense_order[:50], start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (C + rank)

    best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
    return [docs[i].page_content for i in best]


def retrieve_reranked(question: str, k: int, embeddings) -> list[str]:
    """Hybrid over a wider net, then reordered by query term coverage.

    Fetches 3k candidates and keeps k. The rerank favours chunks containing more
    distinct query terms, which is a crude proxy for a cross-encoder and is
    reported as such.
    """
    candidates = retrieve_hybrid(question, k * 3, embeddings)
    q_terms = set(_tokens(question))
    if not q_terms:
        return candidates[:k]

    def coverage(text: str) -> float:
        present = q_terms & set(_tokens(text))
        return len(present) / len(q_terms)

    return sorted(candidates, key=coverage, reverse=True)[:k]


RETRIEVERS = {
    "naive": retrieve_naive,
    "semantic": retrieve_semantic,
    "hybrid": retrieve_hybrid,
    "reranked": retrieve_reranked,
}
