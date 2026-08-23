"""LLM-judged generation quality: RAGAS faithfulness and answer relevancy,
DeepEval hallucination, and refusal behaviour on the unanswerable set.

Sampled by default. A full 150-question run is roughly 600 judge calls against
a free tier capped at a few dozen requests a minute, so the default is a fixed
reproducible sample and the full run is opt-in with EVAL_SAMPLE=0.

Skipped entirely without an API key, so the deterministic half of the suite
stays runnable by anyone.
"""

from __future__ import annotations

import pytest

from tests.eval.conftest import requires_llm, sample
from tests.eval.metrics import refusal_outcomes

# Thresholds sit below measured behaviour, not at it. A gate pinned to the
# current number turns ordinary model variance into a red build.
MIN_FAITHFULNESS = 0.75
MIN_ANSWER_RELEVANCY = 0.70
MAX_HALLUCINATION = 0.35

# Answering a question the corpus cannot support is the failure this project
# exists to prevent, so it is gated hard. Refusing something answerable is
# unhelpful rather than dangerous, and is gated loosely.
MIN_REFUSAL_RECALL = 0.80
MAX_OVER_REFUSAL_RATE = 0.35


@pytest.fixture(scope="module")
def answered(answerable):
    """Run the real pipeline over a sample and keep the contexts it retrieved."""
    requires_llm()
    from app.rag import answer_question

    out = []
    for record in sample(answerable):
        result = answer_question(record["question"])
        out.append(
            {
                "record": record,
                "answer": result.answer,
                "refused": result.refused,
                # Full retrieved chunks, not the truncated citation snippets.
                "contexts": result.retrieved_contexts,
            }
        )
    return out


def test_ragas_faithfulness_and_relevancy(answered):
    """Faithfulness: is every claim in the answer supported by the context.

    Run only over responses that actually answered. Scoring a refusal for
    faithfulness measures the refusal template, and a suite that refuses
    everything would post a perfect score.
    """
    requires_llm()
    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    import os

    scored = [a for a in answered if not a["refused"] and a["contexts"]]
    if len(scored) < 3:
        pytest.skip(f"only {len(scored)} non-refused answers in the sample")

    dataset = Dataset.from_dict(
        {
            "question": [a["record"]["question"] for a in scored],
            "answer": [a["answer"] for a in scored],
            "contexts": [a["contexts"] for a in scored],
            "ground_truth": [a["record"]["ground_truth"] for a in scored],
        }
    )

    key = os.environ["GEMINI_API_KEY"]

    # Both have to be wrapped. Passing the raw LangChain objects works for
    # faithfulness, which only calls the LLM, and silently yields nan for answer
    # relevancy, which needs the embeddings. A nan that reaches an assertion is
    # a metric that failed while looking like it ran.
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=key)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=f"models/{os.environ.get('GEMINI_EMBED_MODEL', 'gemini-embedding-2')}",
            google_api_key=key,
        )
    )

    # answer_relevancy generates `strictness` candidate questions from the answer
    # and measures their similarity to the real one. It asks the model for that
    # many candidates in a single call, and gemini-flash-lite rejects n > 1 with
    # "Multiple candidates is not enabled for this model". With raise_exceptions
    # off that surfaced as an empty column and a nan mean, which is why the guard
    # below exists. strictness=1 keeps the metric working at the cost of a noisier
    # estimate from a single generated question.
    answer_relevancy.strictness = 1

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )

    scores = result.to_pandas()
    faith_col = scores["faithfulness"].dropna()
    rel_col = scores["answer_relevancy"].dropna()

    print(
        f"\nRAGAS over n={len(scored)}: "
        f"faithfulness={faith_col.mean():.3f} (n={len(faith_col)}) "
        f"relevancy={rel_col.mean():.3f} (n={len(rel_col)})"
    )

    # A metric that produced no values did not pass, it failed to run.
    # Averaging an empty column gives nan, and every nan comparison is false,
    # so without these the test fails with an unreadable "nan >= 0.7".
    assert len(faith_col) > 0, "faithfulness produced no scores; the judge did not run"
    assert len(rel_col) > 0, (
        "answer relevancy produced no scores. It needs embeddings, so this is "
        "usually the embeddings wrapper or an exhausted embedding quota."
    )

    assert float(faith_col.mean()) >= MIN_FAITHFULNESS, f"faithfulness {faith_col.mean():.3f}"
    assert float(rel_col.mean()) >= MIN_ANSWER_RELEVANCY, f"answer relevancy {rel_col.mean():.3f}"


def test_deepeval_hallucination(answered):
    """Hallucination: does the answer contradict or exceed its context.

    Overlaps with faithfulness deliberately. They disagree often enough that
    agreement between two independently implemented judges is worth more than
    either alone.
    """
    requires_llm()
    import os

    from deepeval.metrics import HallucinationMetric
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase

    scored = [a for a in answered if not a["refused"] and a["contexts"]]
    if len(scored) < 3:
        pytest.skip(f"only {len(scored)} non-refused answers in the sample")

    model = GeminiModel(
        model="gemini-flash-lite-latest",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    metric = HallucinationMetric(threshold=MAX_HALLUCINATION, model=model)

    scores = []
    for item in scored:
        case = LLMTestCase(
            input=item["record"]["question"],
            actual_output=item["answer"],
            context=item["contexts"],
        )
        metric.measure(case)
        scores.append(metric.score)

    mean_score = sum(scores) / len(scores)
    worst = max(scores)
    print(f"\nDeepEval hallucination over n={len(scores)}: mean={mean_score:.3f} worst={worst:.3f}")

    assert mean_score <= MAX_HALLUCINATION, f"mean hallucination {mean_score:.3f}"


def test_refuses_what_the_corpus_cannot_support(unanswerable, answerable):
    """The behaviour that distinguishes this system from a plain RAG demo.

    Both error directions are reported. Answering an unanswerable question
    invents policy a reader may act on. Refusing an answerable one is merely
    unhelpful, so it is allowed a much looser bound.
    """
    requires_llm()
    from app.rag import answer_question

    from app.guardrails import FLAG_MODEL_UNAVAILABLE

    probe = sample(unanswerable) + sample(answerable)
    results = {r["id"]: answer_question(r["question"]) for r in probe}

    # A refusal caused by the generation call failing is an outage, not a
    # grounding decision. Scoring it as a correct refusal would report a broken
    # run as perfect behaviour, so these are excluded and the run is failed if
    # too many of them occur to measure anything.
    unavailable = [rid for rid, res in results.items() if FLAG_MODEL_UNAVAILABLE in res.flags]
    if unavailable:
        print(f"\n{len(unavailable)}/{len(results)} calls failed: {unavailable[:5]}")
    assert len(unavailable) <= len(results) * 0.2, (
        f"{len(unavailable)} of {len(results)} generation calls failed "
        f"(usually an exhausted rate limit). Refusal cannot be measured on this "
        f"run, and scoring it would report an outage as correct behaviour."
    )

    scored = {rid: res for rid, res in results.items() if rid not in unavailable}
    probe = [r for r in probe if r["id"] in scored]
    flags = {rid: res.refused for rid, res in scored.items()}

    outcomes = refusal_outcomes(probe, flags)
    print(
        f"\nRefusal: recall={outcomes['refusal_recall']:.3f} "
        f"over-refusal={outcomes['over_refusal_rate']:.3f} "
        f"missed={outcomes['missed_refusals']} over={outcomes['over_refusals']}"
    )

    assert outcomes["refusal_recall"] >= MIN_REFUSAL_RECALL, (
        f"refusal recall {outcomes['refusal_recall']:.3f}: the system answered "
        f"{outcomes['missed_refusals']} question(s) the corpus cannot support"
    )
    assert outcomes["over_refusal_rate"] <= MAX_OVER_REFUSAL_RATE, (
        f"over-refusal rate {outcomes['over_refusal_rate']:.3f}"
    )
