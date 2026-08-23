"""Fixtures and run-size control for the evaluation suite.

The suite is split by cost. Retrieval and dataset checks are deterministic and
free, so they always run and can gate CI. Generation checks call an LLM judge
and are sampled by default, because a full 150-question RAGAS run is roughly
600 judge calls against a free tier capped at a few dozen requests a minute.

    pytest tests/eval -q                     # sampled, the default
    EVAL_SAMPLE=0 pytest tests/eval -q       # every question, slow and expensive
    EVAL_SAMPLE=40 pytest tests/eval -q      # a specific sample size
"""

from __future__ import annotations

import os
import random

import pytest
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SAMPLE = 12
SAMPLE_SEED = 20260823  # fixed so a sampled run is reproducible


def _sample_size() -> int:
    raw = os.environ.get("EVAL_SAMPLE", str(DEFAULT_SAMPLE))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_SAMPLE


def sample(records: list[dict]) -> list[dict]:
    """A reproducible subset, or everything when EVAL_SAMPLE=0."""
    size = _sample_size()
    if size == 0 or size >= len(records):
        return records
    return random.Random(SAMPLE_SEED).sample(records, size)


def requires_llm() -> None:
    """Skip rather than fail when no key is configured.

    A contributor without a key should still be able to run the deterministic
    half of the suite and get a meaningful signal from it.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set; skipping LLM-judged checks")


@pytest.fixture(scope="session")
def dataset() -> list[dict]:
    from tests.eval.validate_dataset import load_dataset

    return load_dataset()


@pytest.fixture(scope="session")
def answerable(dataset: list[dict]) -> list[dict]:
    return [r for r in dataset if r["expected_behavior"] == "answer"]


@pytest.fixture(scope="session")
def unanswerable(dataset: list[dict]) -> list[dict]:
    return [r for r in dataset if r["expected_behavior"] == "refuse"]


def eval_embed_model() -> str:
    """Embedding model the evaluation uses.

    Separate from the model the shipped index was built with, because the free
    embedding quota is per model per day. When one model's daily budget is spent
    the evaluation can still run on another, and the model is recorded alongside
    every published number so the figures stay interpretable. Every retrieval
    configuration in a given run uses the same model, so the ablation remains a
    controlled comparison of chunking and ranking rather than of embeddings.
    """
    from app.ingest import PRIMARY_EMBED_MODEL

    return os.environ.get("EVAL_EMBED_MODEL", PRIMARY_EMBED_MODEL)


@pytest.fixture(scope="session")
def embeddings():
    from app.ingest import build_embeddings

    return build_embeddings(eval_embed_model())
