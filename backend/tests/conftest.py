"""Shared fixtures.

Nothing in this suite calls Gemini, Chroma or the network. The chain still
needs an API key present to construct, so a dummy is supplied when the
environment has none.
"""

import os

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
# Never ship traces from a test run to a real LangSmith project.
os.environ["LANGSMITH_TRACING"] = "false"

from app import rag  # noqa: E402


def make_chunk(label: str, score: float, section: str = "Return Window"):
    return rag.RetrievedChunk(
        label=label,
        text=f"body of {label}",
        source="returns-refunds.md",
        section=section,
        score=score,
    )


@pytest.fixture
def chunks():
    """Two retrieved chunks, both comfortably above the grounding threshold."""
    return [make_chunk("chunk-1", 0.90), make_chunk("chunk-2", 0.80)]


@pytest.fixture
def fake_store(monkeypatch):
    """Replace the vector store so retrieval returns fixed scores, no network.

    rag._retrieve calls get_store() at request time rather than capturing it
    when the chain is built, so patching here reaches the live chain.
    """

    def _install(score: float, n: int = 1):
        class _Doc:
            def __init__(self):
                self.page_content = "body"
                self.metadata = {
                    "source": "returns-refunds.md",
                    "section": "Return Window",
                }

        class _Store:
            @staticmethod
            def similarity_search_with_relevance_scores(query, k):
                return [(_Doc(), score) for _ in range(n)]

        monkeypatch.setattr(rag, "get_store", lambda: _Store())

    return _install
