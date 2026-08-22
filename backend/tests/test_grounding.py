"""Grounding enforcement: the checks that decide whether an answer ships.

These sit in plain Python inside Runnables rather than in the prompt, which
is the whole point of the design and therefore the part worth testing.
"""

import pytest

from app import rag
from app.guardrails import FLAG_OFF_TOPIC, GROUNDING_SCORE_THRESHOLD

from .conftest import make_chunk


# ---------------------------------------------------------------------------
# The gate: is anything retrieved close enough to answer from
# ---------------------------------------------------------------------------


def test_threshold_is_documented_value():
    """ARCHITECTURE.md and the README both quote 0.45."""
    assert GROUNDING_SCORE_THRESHOLD == 0.45


def test_empty_retrieval_is_ungrounded():
    assert rag._is_ungrounded({"chunks": []})


def test_all_weak_chunks_are_ungrounded():
    chunks = [make_chunk("chunk-1", 0.44), make_chunk("chunk-2", 0.10)]
    assert rag._is_ungrounded({"chunks": chunks})


def test_one_strong_chunk_is_enough():
    """The best score decides, not the average, so weak extras do not sink it."""
    chunks = [make_chunk("chunk-1", 0.46), make_chunk("chunk-2", 0.10)]
    assert not rag._is_ungrounded({"chunks": chunks})


@pytest.mark.parametrize(
    "score,ungrounded",
    [(0.0, True), (0.44, True), (0.449, True), (0.45, False), (1.0, False)],
)
def test_threshold_boundary(score, ungrounded):
    assert rag._is_ungrounded({"chunks": [make_chunk("chunk-1", score)]}) is ungrounded


# ---------------------------------------------------------------------------
# The verifier: is what the model returned allowed to reach the user
# ---------------------------------------------------------------------------


def test_cited_and_confident_answer_ships(chunks):
    result = rag._verify_and_build(
        {
            "chunks": chunks,
            "payload": rag.AnswerPayload(
                answer="14 days.", citedChunkIds=["chunk-1"], confident=True
            ),
        }
    )
    assert result.refused is False
    assert result.answer == "14 days."
    assert len(result.citations) == 1
    assert result.citations[0].source == "returns-refunds.md"


def test_no_citations_refuses(chunks):
    """An answer with nothing behind it is not an answer."""
    result = rag._verify_and_build(
        {
            "chunks": chunks,
            "payload": rag.AnswerPayload(
                answer="14 days.", citedChunkIds=[], confident=True
            ),
        }
    )
    assert result.refused
    assert result.answer == rag.REFUSAL_MESSAGE
    assert result.citations == []


def test_not_confident_refuses(chunks):
    result = rag._verify_and_build(
        {
            "chunks": chunks,
            "payload": rag.AnswerPayload(
                answer="Maybe 14 days.", citedChunkIds=["chunk-1"], confident=False
            ),
        }
    )
    assert result.refused


def test_citation_pointing_at_a_chunk_that_was_never_retrieved_refuses(chunks):
    """The model can invent a chunk id. Citations are checked, not trusted."""
    result = rag._verify_and_build(
        {
            "chunks": chunks,
            "payload": rag.AnswerPayload(
                answer="Invented.", citedChunkIds=["chunk-99"], confident=True
            ),
        }
    )
    assert result.refused


def test_partially_invented_citations_keep_only_the_real_ones(chunks):
    result = rag._verify_and_build(
        {
            "chunks": chunks,
            "payload": rag.AnswerPayload(
                answer="14 days.",
                citedChunkIds=["chunk-1", "chunk-99"],
                confident=True,
            ),
        }
    )
    assert result.refused is False
    assert len(result.citations) == 1


def test_none_payload_refuses(chunks):
    """Regression: with_structured_output returns None rather than raising
    when the model produces nothing parseable. Dereferencing it surfaced as a
    503 instead of a refusal."""
    result = rag._verify_and_build({"chunks": chunks, "payload": None})
    assert result.refused
    assert result.answer == rag.REFUSAL_MESSAGE


def test_empty_answer_string_falls_back_to_the_refusal_text(chunks):
    result = rag._verify_and_build(
        {
            "chunks": chunks,
            "payload": rag.AnswerPayload(
                answer="   ", citedChunkIds=["chunk-1"], confident=True
            ),
        }
    )
    assert result.answer == rag.REFUSAL_MESSAGE


# ---------------------------------------------------------------------------
# The chain end to end, with retrieval stubbed and the model never reached
# ---------------------------------------------------------------------------


def test_off_topic_question_refuses_without_calling_the_model(fake_store):
    """An off-topic question should cost a retrieval and nothing else."""
    fake_store(score=0.2)
    result = rag.get_chain().invoke({"question": "What is the capital of France?"})
    assert result.refused
    assert FLAG_OFF_TOPIC in result.flags
    assert result.citations == []


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_question_stays_inside_the_untrusted_envelope(chunks):
    formatted = rag._format_context({"question": "How long do returns take?", "chunks": chunks})
    body = rag._prompt.format_messages(**formatted)[-1].content
    assert "<user_question>\nHow long do returns take?\n</user_question>" in body


def test_braces_in_a_question_are_not_treated_as_template_variables(chunks):
    """A question containing {curly} braces must not be re-parsed as a
    template, or a user could inject template syntax."""
    formatted = rag._format_context({"question": "what is {this} and {{that}}", "chunks": chunks})
    body = rag._prompt.format_messages(**formatted)[-1].content
    assert "{this}" in body
    assert "{{that}}" in body


def test_chunk_labels_and_sources_reach_the_prompt(chunks):
    formatted = rag._format_context({"question": "q", "chunks": chunks})
    body = rag._prompt.format_messages(**formatted)[-1].content
    assert "[chunk-1]" in body
    assert "[chunk-2]" in body
    assert "returns-refunds.md" in body


def test_system_prompt_is_the_first_message(chunks):
    formatted = rag._format_context({"question": "q", "chunks": chunks})
    messages = rag._prompt.format_messages(**formatted)
    assert messages[0].type == "system"
    assert "DocSentry" in messages[0].content
