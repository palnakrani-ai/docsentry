"""Deterministic evaluation gates: dataset integrity and retrieval quality.

Nothing here calls an LLM. These run free, in seconds, and give the same answer
on every machine, which is what makes them safe to fail a build on.

Thresholds are set below the measured baseline rather than at it. A gate set at
the current number turns every incidental fluctuation into a red build, and a
suite that cries wolf gets bypassed.
"""

from __future__ import annotations

import pytest

from tests.eval.metrics import score_retrieval, summarize
from tests.eval.retrievers import RETRIEVERS, naive_chunks, semantic_chunks
from tests.eval.validate_dataset import load_dataset, validate

TOP_K = 5

# Floors, measured on the committed corpus and labels, set with headroom.
MIN_HIT_RATE = 0.85
MIN_RECALL = 0.80
MIN_MRR = 0.65

TARGET_TOTAL = 150


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------

def test_dataset_is_structurally_sound():
    """Every label parses, is unique, and cites text that exists in the corpus.

    This runs first because every other number depends on it. A label whose
    anchor has drifted out of the corpus scores against nothing while the suite
    still reports a figure, which is worse than having no label at all.
    """
    problems = validate(load_dataset())
    assert problems == [], "dataset problems:\n" + "\n".join(f"  - {p}" for p in problems)


def test_dataset_has_the_committed_number_of_labels(dataset):
    assert len(dataset) == TARGET_TOTAL


def test_dataset_covers_every_label_type(dataset):
    """All four question types are represented, and refusals are a real share.

    A dataset of only answerable factual lookups measures retrieval and nothing
    else. The refusal cases are what make it a test of grounding.
    """
    types = {r["type"] for r in dataset}
    assert types == {"factual", "multi_hop", "unanswerable", "ambiguous"}

    refusals = [r for r in dataset if r["expected_behavior"] == "refuse"]
    assert len(refusals) >= 25, "too few refusal cases to measure over-answering"


def test_refusal_cases_cite_nothing(unanswerable):
    """A refusal case with a citation is a contradiction in the label itself."""
    for record in unanswerable:
        assert not record["source_docs"], f"{record['id']} cites docs but expects refusal"
        assert not record["source_anchors"], f"{record['id']} cites anchors but expects refusal"


def test_corpus_is_large_enough_for_the_ablation_to_mean_anything():
    """Guards the reason the corpus was expanded in the first place.

    At 26 chunks against k=5 every strategy retrieved a fifth of the corpus and
    the four configurations were indistinguishable. If the corpus ever shrinks
    back toward that, the ablation stops measuring retrieval and this fails.
    """
    chunks = len(semantic_chunks())
    assert chunks >= 150, f"only {chunks} chunks; ablation loses discriminating power"
    assert TOP_K / chunks < 0.05, "top k is too large a share of the corpus"


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def semantic_scores(answerable, embeddings):
    """Score the shipped configuration once and reuse across assertions."""
    retriever = RETRIEVERS["semantic"]
    return [
        score_retrieval(r["id"], retriever(r["question"], TOP_K, embeddings), r["source_anchors"])
        for r in answerable
    ]


def test_shipped_retrieval_meets_its_floor(semantic_scores):
    summary = summarize(semantic_scores)
    assert summary.hit_rate >= MIN_HIT_RATE, f"hit rate {summary.hit_rate:.3f}"
    assert summary.recall >= MIN_RECALL, f"context recall {summary.recall:.3f}"
    assert summary.mrr >= MIN_MRR, f"MRR {summary.mrr:.3f}"


def test_section_aware_chunking_beats_naive(answerable, embeddings):
    """The shipped strategy has to earn its complexity.

    Section-aware chunking costs a splitting pass and heading bookkeeping. If it
    does not retrieve better than fixed-size splitting on this corpus, the
    simpler thing should ship instead, and this test would say so.
    """
    subset = answerable[:40]  # enough to separate the two, cheap enough to run every time
    results = {}
    for name in ("naive", "semantic"):
        retriever = RETRIEVERS[name]
        scores = [
            score_retrieval(r["id"], retriever(r["question"], TOP_K, embeddings), r["source_anchors"])
            for r in subset
        ]
        results[name] = summarize(scores)

    # Compared on MRR rather than recall. The full ablation puts recall at 0.943
    # against 0.935, a margin too thin to assert on without the test flapping.
    # The real difference is where the supporting chunk lands in the ranking:
    # 0.890 against 0.833. Section-aware chunks keep a policy intact, so the
    # right chunk surfaces higher rather than merely appearing somewhere in k.
    assert results["semantic"].mrr > results["naive"].mrr, (
        f"semantic MRR {results['semantic'].mrr:.3f} did not beat "
        f"naive {results['naive'].mrr:.3f}; the extra complexity is not paying for itself"
    )


def test_multi_hop_questions_retrieve_across_documents(answerable, embeddings):
    """Multi-hop labels cite two documents, and retrieval should reach both.

    Scored separately from the headline number because a suite dominated by
    single-document lookups can post a strong average while failing every
    question that needs two sources.
    """
    multi = [r for r in answerable if r["type"] == "multi_hop"]
    assert multi, "no multi-hop questions to score"

    retriever = RETRIEVERS["semantic"]
    scores = [
        score_retrieval(r["id"], retriever(r["question"], TOP_K, embeddings), r["source_anchors"])
        for r in multi
    ]
    summary = summarize(scores)
    assert summary.hit_rate >= 0.80, f"multi-hop hit rate {summary.hit_rate:.3f}"
