"""Retrieval metrics computed from the dataset anchors, with no LLM involved.

These are separated from the RAGAS and DeepEval metrics deliberately. Context
recall and precision are decidable from the labels: an anchor is either in the
retrieved text or it is not. Paying an LLM judge to decide that would add cost,
latency, and variance to a question that has an exact answer, and would make
the ablation non-reproducible.

The LLM judge is reserved for what actually needs judgement, which is whether a
generated answer is faithful to its context and responsive to the question.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.eval.anchors import anchors_found, chunk_supports


@dataclass(frozen=True)
class RetrievalScore:
    """One question's retrieval outcome under one configuration."""

    question_id: str
    recall: float           # fraction of the label's anchors present anywhere in the top k
    precision: float        # fraction of retrieved chunks that carry an anchor
    hit: bool               # at least one anchor retrieved
    first_hit_rank: int | None  # 1-based rank of the first supporting chunk


def score_retrieval(
    question_id: str,
    retrieved_texts: list[str],
    anchors: list[str],
) -> RetrievalScore:
    """Score one retrieval against one label."""
    if not anchors:
        raise ValueError(f"{question_id}: cannot score retrieval without anchors")

    found = anchors_found(retrieved_texts, anchors)
    recall = len(found) / len(anchors)

    supporting = [chunk_supports(t, anchors) for t in retrieved_texts]
    precision = (sum(supporting) / len(supporting)) if supporting else 0.0

    first_hit = next((i + 1 for i, ok in enumerate(supporting) if ok), None)

    return RetrievalScore(
        question_id=question_id,
        recall=recall,
        precision=precision,
        hit=first_hit is not None,
        first_hit_rank=first_hit,
    )


@dataclass(frozen=True)
class RetrievalSummary:
    """Aggregate across a set of questions."""

    n: int
    recall: float
    precision: float
    hit_rate: float
    mrr: float

    def as_row(self, label: str) -> str:
        return (
            f"| {label} | {self.n} | {self.recall:.3f} | {self.precision:.3f} "
            f"| {self.hit_rate:.3f} | {self.mrr:.3f} |"
        )


def summarize(scores: list[RetrievalScore]) -> RetrievalSummary:
    """Mean recall, precision, hit rate, and mean reciprocal rank.

    MRR uses 0 for a miss rather than dropping the question, so a configuration
    cannot improve its score by failing to retrieve anything at all.
    """
    if not scores:
        return RetrievalSummary(n=0, recall=0.0, precision=0.0, hit_rate=0.0, mrr=0.0)

    n = len(scores)
    reciprocal = [
        (1.0 / s.first_hit_rank) if s.first_hit_rank else 0.0 for s in scores
    ]
    return RetrievalSummary(
        n=n,
        recall=sum(s.recall for s in scores) / n,
        precision=sum(s.precision for s in scores) / n,
        hit_rate=sum(1 for s in scores if s.hit) / n,
        mrr=sum(reciprocal) / n,
    )


# ---------------------------------------------------------------------------
# Refusal behaviour
# ---------------------------------------------------------------------------

def refusal_outcomes(records: list[dict], refused_flags: dict[str, bool]) -> dict:
    """Confusion matrix for refusal against expected_behavior.

    Both error types are reported separately because they are not equally bad
    here. Answering an unanswerable question invents policy a reader may act on.
    Refusing an answerable one is unhelpful but harmless.
    """
    should_refuse = {r["id"] for r in records if r["expected_behavior"] == "refuse"}

    true_pos = sum(1 for rid, refused in refused_flags.items() if refused and rid in should_refuse)
    false_pos = sum(1 for rid, refused in refused_flags.items() if refused and rid not in should_refuse)
    false_neg = sum(1 for rid, refused in refused_flags.items() if not refused and rid in should_refuse)
    true_neg = sum(1 for rid, refused in refused_flags.items() if not refused and rid not in should_refuse)

    total = len(refused_flags) or 1
    n_should = sum(1 for rid in refused_flags if rid in should_refuse) or 1
    n_should_not = sum(1 for rid in refused_flags if rid not in should_refuse) or 1

    return {
        "correct_refusals": true_pos,
        "missed_refusals": false_neg,      # answered something unanswerable
        "over_refusals": false_pos,        # refused something answerable
        "correct_answers": true_neg,
        "refusal_recall": true_pos / n_should,
        "over_refusal_rate": false_pos / n_should_not,
        "accuracy": (true_pos + true_neg) / total,
    }
