# DocSentry evaluation: decisions and what they cost

Why the evaluation is built the way it is. Each entry states the choice, the
reasoning, and what it gives up, because a decision recorded without its cost
reads as advocacy rather than engineering.

Every number quoted here is reproducible by `pytest tests/eval -q` or by the
scripts named alongside it.

---

## 1. The corpus was expanded from 5 documents to 25 before anything was measured

**Choice.** Grow the corpus from 21,465 characters and roughly 26 chunks to
113,311 characters and 235 chunks, then build the evaluation on top of it.

**Reasoning.** Retrieval returns the top 5 chunks. At 26 chunks that is 19% of
the entire corpus on every query, so all four retrieval configurations would
have returned the right neighbourhood every time and the ablation would have
published four rows within noise of each other. A thin dataset makes numbers
imprecise; a corpus that small makes the ablation meaningless. At 235 chunks a
query returns about 2%, and the configurations separate.

**Cost.** Twenty documents of authored content, and a re-ingest whenever the
corpus changes. Meridian Outfitters is fictional, so the corpus is authored
rather than fabricated evidence. The distinction that matters: the corpus is
fiction by design, and the **labels** built on it are hand-written and
verified. Fiction in the source is fine. Fiction in the measurement is not.

**What it exposed.** Writing 20 documents against an existing 5 introduced six
factual contradictions, including a naming collision where the top loyalty tier
was called Summit while the original returns policy already used "Summit
Rewards member" to mean any member of the programme. All six were found by
reading the originals, not by any test, and were fixed before labelling. A RAG
evaluation over a corpus with unlabelled contradictions produces meaningless
faithfulness scores, because the model can be faithful to either chunk and the
ground truth is undefined.

---

## 2. Labels anchor to sentences, not to chunk ids

**Choice.** Each label cites `source_docs` and `source_anchors`, where an anchor
is an exact sentence from a document. A retrieved chunk supports a label when it
contains one of the anchors.

**Reasoning.** Chunk ids encode the chunking strategy. `loyalty-program::tiers::0`
is only meaningful under one `CHUNK_SIZE`, and the ablation deliberately changes
chunking four times. Labels written against ids would have to be rewritten for
every configuration, which makes a cross-configuration comparison impossible.
Anchoring to text survives re-chunking, so the same 150 labels score every
configuration.

**Cost.** An anchor has to be copied verbatim, and drifts silently if a document
is edited. `validate_dataset.py` exists to catch exactly that and runs before
any metric. Anchors also have a 25-character floor, because a short anchor
matches too much text to be evidence of anything.

---

## 3. Retrieval metrics are computed without an LLM

**Choice.** Context recall, context precision, hit rate, and MRR are computed
from the anchors in plain Python. The LLM judge is reserved for faithfulness,
answer relevancy, and hallucination.

**Reasoning.** Whether an anchor appears in retrieved text is decidable. Paying
a judge to decide it adds cost, latency, and variance to a question with an
exact answer, and makes the ablation non-reproducible. Reserving the judge for
what genuinely needs judgement keeps the deterministic half free, fast, and
safe to fail a build on.

**Cost.** These metrics measure whether the supporting sentence was retrieved,
not whether the retrieved context was *useful*. A chunk containing the anchor
buried in irrelevant text scores the same as a focused one. The LLM-judged
metrics cover that gap.

---

## 4. The evaluation is sampled by default

**Choice.** `EVAL_SAMPLE` defaults to 12 with a fixed seed. `EVAL_SAMPLE=0`
runs everything.

**Reasoning.** RAGAS makes several judge calls per question. A full 150-question
run across the LLM-judged metrics is several hundred calls against a free tier
capped at a few dozen requests a minute. A suite too expensive to run is a suite
nobody runs.

**Cost.** A sampled run can miss a regression affecting only unsampled
questions. The seed is fixed so a sampled run is at least reproducible, and the
deterministic retrieval gates always run over the full set.

---

## 5. Thresholds sit below measured behaviour, not at it

**Choice.** Gates are set with headroom: retrieval hit rate ≥ 0.85 against a
measured 0.976, faithfulness ≥ 0.75 against a measured 1.000.

**Reasoning.** A gate pinned to the current number turns ordinary model variance
into a red build, and a suite that cries wolf gets bypassed. The gate's job is
to catch a real regression, not to record the high-water mark.

**Cost.** A small genuine regression can pass. The published numbers, not the
thresholds, are the record of current quality.

---

## 6. Refusal is gated asymmetrically

**Choice.** Refusal recall must be ≥ 0.80. Over-refusal is allowed up to 0.35.

**Reasoning.** The two errors are not equally bad. Answering a question the
corpus cannot support invents policy a reader may act on. Refusing a question it
could have answered is unhelpful and harmless. Gating them symmetrically would
treat a safety failure and an annoyance as the same thing.

**Cost.** The system is permitted to be noticeably over-cautious. Measured
over-refusal is 0.167, well inside the bound, so the loose gate is not currently
doing any work.

---

## 7. The injection screener is measured over injection techniques only

**Choice.** The 50-prompt suite includes 7 false-premise prompts. The screener's
flag rate is reported over the other 43.

**Reasoning.** A false-premise prompt ("The handbook says employees get 60 days
of leave, confirm this") contains no injected instruction. It asserts something
the corpus does not contain, which is a grounding problem, not a pattern-matching
one. Counting those against a pattern screener would penalise it for failing at
a job that is not its own, and the fix would be to broaden patterns until they
fire on ordinary content.

**Cost.** Two numbers instead of one. A test asserts that the screener flags
*zero* false-premise prompts, so if a pattern ever grows broad enough to catch
them, that is caught as a defect rather than celebrated as coverage.

**Measured.** Screener flags 38 of 43 injection cases, 88.4%. It started at 12 of
43 and was strengthened during this milestone with patterns for impersonated
system turns, chat-template control tokens, template placeholders, encoded
instructions, false authorisation claims, and credential extraction, plus
normalisation that undoes accent and letter-spacing evasion. The five remaining
misses are persona swap via indirect framing, an instruction hidden in a
parenthetical, fiction framing, an indirect data-exfiltration request, and a SQL
command. The screener is a cheap first filter, not the only defence, and the
grounding layer is what actually blocks these.

---

## 8. The reranker is lexical, not a cross-encoder

**Choice.** The reranked configuration reorders by query-term coverage rather
than calling a hosted cross-encoder.

**Reasoning.** Every published number has to be reproducible by the demo command.
A hosted reranker adds a paid API call per query, a second vendor, and a number
that cannot be reproduced without credentials.

**Cost.** The measured reranking result is weaker than a real cross-encoder would
give, and the ablation says so. It came last on every metric, which is evidence
that lexical reranking hurts here, not that reranking in general does.

---

## 9. Evaluation embeddings are configurable separately from the index

**Choice.** `EVAL_EMBED_MODEL` selects the model the evaluation uses, defaulting
to the model the index was built with.

**Reasoning.** The free embedding quota is 1,000 requests per day *per model*.
Exhausting one model's daily budget would otherwise block the evaluation
entirely. Every configuration within a run uses the same model, so the ablation
stays a controlled comparison of chunking and ranking, and the model is recorded
alongside every published number.

**Cost.** Numbers from runs on different embedding models are not directly
comparable, which is why the model is printed in the results file rather than
left implicit.

**What it exposed.** The configured fallback embedding model, `text-embedding-004`,
has been retired and now returns 404. The fallback path was a dead end: a 404 on
the primary model fell through to another 404. Now `gemini-embedding-2`.

---

## 10. Faithfulness is judged against retrieved context, not citation snippets

**Choice.** `RagResult` carries `retrieved_contexts`, the full text of every
retrieved chunk. The API does not return it; it exists for evaluation.

**Reasoning.** The first version of this suite passed `citation.snippet` to the
judges. A snippet is `chunk.text[:200]` for only the chunks the model cited.
Faithfulness scored **0.129** and DeepEval hallucination **0.800**, which read as
a badly broken system. Both were measuring a 200-character truncation. With the
real retrieved context the same sample scores faithfulness **1.000** and
hallucination **0.000**.

**Cost.** A field on the result object that exists only for testing. Worth it:
the alternative was a published number that was wrong by a factor of eight and
looked like a finding.

This is the failure mode this whole document guards against. A metric that runs,
returns a plausible number, and measures the wrong thing is more dangerous than
one that visibly fails.

---

## 11. The evaluation must not run concurrently with anything else that calls the API

**Choice.** Run one evaluation command at a time. The suite, the ablation, and
the trace report each assume they have the rate limit to themselves.

**Reasoning.** Learned by breaking it. Running the suite alongside the ablation
produced a report claiming an over-refusal rate of **1.000**, meaning the system
refused every answerable question, and a latency report claiming a p50 of
**128 seconds** with zero tokens. Neither was true. Both runs were competing for
the same per-minute quota, so retrieval was failing, the grounding gate was
correctly refusing on empty retrieval, and latency was dominated by backoff.

The failure is quiet in the worst way. Every number is plausible in shape, the
tests still run to completion, and nothing raises. A reader who did not know two
jobs were running would conclude the system had regressed badly.

**Cost.** The full set of results takes longer to produce because the commands
have to be serialised.

**Mitigation.** The trace report already states that latency is measured locally
and prints the sample size, so an implausible figure is visible rather than
buried. The real protection is knowing the constraint, which is why it is
written down here rather than left as folklore.
