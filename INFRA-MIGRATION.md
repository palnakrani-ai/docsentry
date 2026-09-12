# DocSentry — Infra Migration Off Render

Plan to move DocSentry off Render onto an owned stack, matching Dave Ebbelaar's architecture (Datalumina): FastAPI, Redis, Celery, Celery Beat, Supabase Postgres, Docker Compose, Hetzner, Cloudflare, Caddy, Sentry, Grafana. DocSentry goes first: it's already Python/FastAPI/Docker, closest of the four demos to the target shape.

## Why this one first

- Solves the real problem: Render free tier cold starts run 42 to 50 seconds, and an always-on VPS removes that without needing a keep-warm workaround. Worth saying plainly that no keep-warm cron exists today. There is no `.github/workflows` in this repo, so the cold start is currently just eaten.
- Already shaped like Dave's stack: receive a request, run a deterministic check (the grounding gate in `app/guardrails.py`), do work, return a result
- Flagship demo for the Upwork catalog and the portfolio, so a real infra story here carries the most weight

## What changes

| Layer | Now | After | Note |
|---|---|---|---|
| Hosting | Render free tier (`render.yaml`, `plan: free`) | Hetzner VPS, CX22 class | always-on, removes the 42 to 50s cold start |
| Orchestration | one Docker image, multi-stage | Docker Compose: `api`, `worker`, `beat`, `redis` | |
| Web server | FastAPI + uvicorn on :7860 | unchanged | |
| Frontend | React/Vite build served static by FastAPI | unchanged | |
| Vector store | Chroma embedded in-process, singleton at `rag.py:107` | Supabase Postgres with pgvector | see "Chroma has to go" below |
| Index build | `python -m app.ingest` at Docker build time | Celery task, on demand plus a Beat schedule | rebuild without a redeploy |
| Broker | none | Redis | exists to serve the ingest task |
| Audit trail | none beyond logs | Supabase Postgres `events` table, written off the response path | question, refusal reason, citations, latency_ms, timestamp |
| Database | none | Supabase Postgres, one project holding both vectors and events | |
| Scheduler | none | Celery Beat | periodic re-embed when source docs change |
| TLS and reverse proxy | Render-managed | Caddy | automatic HTTPS |
| DNS | Render subdomain | Cloudflare | |
| Errors | whatever Render surfaces | Sentry, wired into FastAPI | |
| Latency visibility | one measured number in the README | Grafana reading from `events` | latency over time, refusal rate, injection-flag rate |
| Secrets | BuildKit build secret plus Render `sync: false` | runtime env vars in Compose | the Hugging Face Spaces block at `Dockerfile:26-33` goes away |
| LLM | Gemini via `langchain-google-genai` | unchanged | |

## Chroma has to go, and the original plan was wrong about that

The first draft of this plan said Chroma stays and there was no reason to move it. That holds only while the index is baked into the image at build time, which is what the `RUN python -m app.ingest` line in the Dockerfile does today.

The moment ingest becomes a Celery task, it breaks. `rag.py:107` keeps `_store` as a module-level singleton, lazily loaded once and cached for the life of the process. A worker rebuilding the index writes new data to disk while the api process keeps answering from the copy it loaded at startup. Nothing errors. The answers just quietly go stale, on a system whose entire claim is that code verifies things rather than trusting them.

Moving the vectors into Supabase Postgres with pgvector fixes this by removing the local copy altogether. The worker writes embeddings to Postgres, the api reads from Postgres, and there is one source of truth. It also collapses two datastores into one, so a query event and the chunks it retrieved can be joined in the same database that Grafana already reads.

## The 0.45 threshold was the risk, and it measured as unchanged

`rag.py` refuses to answer when the best match scores below 0.45, using Chroma's
`similarity_search_with_relevance_scores`. That number was tuned against Chroma's
scoring, pgvector works in cosine distance, and nothing in the suite asserts the
score scale, so it could have changed meaning without a test failing.

Measured across all 150 labelled questions with one shared query embedding per
question: maximum absolute difference in top score **0.000000**, top-1 source
agreement **150/150**, decisions changed at 0.45 **zero**. Both stores use the
same embeddings, the same cosine distance and the same `1 - distance` formula,
and at 235 chunks Chroma's approximate index is effectively exact. The threshold
carries over untouched. Recorded with the failed first attempt at measuring it in
`DECISIONS.md` section 13.

The same data corrected something this plan assumed. All 26 unanswerable
questions score above 0.45 and reach the model, on both stores. The threshold is
not what catches them; the post-generation citation check is. The threshold only
saves a model call on questions that are nowhere near the corpus.

## The events write stays off the response path

Writing an event row per request is step 2 in the build order below, described there as no behaviour change. A synchronous insert would not be a no-op. The request path sits at p95 around 2.3 seconds already, and a blocking write adds latency plus a new way for requests to fail when the database is unreachable. Use FastAPI `BackgroundTasks`, or push the row through Celery. Either way the user's answer does not wait on the audit trail.

## Why Celery at all, since an interviewer will ask

Worth being straight about this. DocSentry serves one synchronous request shape, question in and answer out, in about two seconds. There is no long-running work in the request path and nothing that needs a queue to absorb load. Redis, a worker and Beat are three moving parts serving essentially one task.

The honest justification is that it makes the index rebuildable without a redeploy, and that it matches the architecture the rest of the work is heading toward. That is a real reason. Pretending the workload demands a task queue is not, and claiming it would be the weakest sentence in the whole story.

## Why Supabase rather than Postgres in the Compose file

Postgres running next to the api on the same Hetzner box would be faster, since it removes a network hop per query, and it would fit the "owned stack" framing more neatly. Supabase wins on everything else: managed backups, a SQL interface without SSH, no `pg_dump` cron of your own to write and then forget about, and it is the database in the architecture being matched.

The one thing to watch is that Supabase pauses free-tier projects after about a week without activity, which would reintroduce a cold start of exactly the kind this migration exists to remove. The Beat schedule that re-embeds source docs also keeps the database touched often enough that it should never idle out. Confirm that assumption once Beat is running rather than assuming it.

## Build order

Steps 1 to 4 are done, against the local pgvector container in
`docker-compose.dev.yml` rather than Supabase, so the port was verified before a
hosted dependency entered it. Steps 5 onward are M2.6.

1. ~~Postgres with pgvector and the `events` table.~~ Done as a local container.
   `backend/sql/init/001_schema.sql` is the schema, and it applies to Supabase by
   hand from the SQL editor when the project exists.
2. ~~Wire `app/main.py` to write an event row per request, through
   `BackgroundTasks`.~~ Done. `db.record_event` also swallows its own failures,
   so a request that was answered correctly cannot fail because the record of it
   could not be saved.
3. ~~Port the vector store from Chroma to pgvector.~~ Done. `app/db.py` is new and
   holds the connection, the audit write, and the collection queries that replace
   Chroma's `.get()`, which PGVector has no equivalent for.
4. ~~Re-tune the 0.45 relevance threshold against the 150-label set.~~ Done, and
   it needed no change. See the section above.
5. Move `app/ingest.py` into a Celery task, callable on demand instead of only at Docker build time. Drop the build-time ingest and the Hugging Face Spaces secret block from the Dockerfile.
6. Docker Compose: api, worker, beat and redis on one box, pointed at the Supabase project
7. Hetzner VPS, Cloudflare DNS and Caddy in front, then cut over from Render
8. Sentry on the FastAPI app
9. Grafana dashboard reading from `events`: latency over time, refusal rate, injection-flag rate
10. Re-measure and republish every number, see below

## What this invalidates and has to be re-measured

M2's standing promise is that every published number is reproducible by the demo command. This migration changes the retrieval backend and the hosting, so several of them stop being true on the day of the cutover.

- Latency and cost. The README currently reports p50 2,073 ms and p95 2,361 ms end to end at around 860 tokens, and the genesis records carry an earlier clean run at p50 1,495 ms and p95 1,762 ms. Both were measured elsewhere. Re-run `python -m tests.eval.run_trace_report` after cutover and update the README, `EXPLAINED.md` and `tests/eval/results/trace_report.md`.
- The four-way ablation does **not** need re-running, which an earlier draft of this document got wrong. `tests/eval/retrievers.py` builds its own chunkings and computes cosine similarity in memory against a disk-backed embedding cache. It never touched Chroma and does not touch pgvector, so the store swap cannot move those numbers.
- Run these one at a time. They share a Gemini rate limit and a contended run measures the contention rather than the system, which is the whole point of `DECISIONS.md` section 11.

## The branch is not deployable, deliberately

The image still runs `python -m app.ingest` at build time, and that now needs a
database, so the Docker build fails on this branch. That is step 5's job to fix
and it belongs with the Compose work rather than here.

It also means this must not reach `main` while Render is still serving the live
demo from a baked Chroma index. Merging before a database exists takes the
flagship demo down. The order is: Supabase project first, then the Dockerfile and
Compose changes, then merge, then cut over.

## Not urgent

Background work, and not a blocker on anything in the job search or the freelance push. Pull time into this once applications and proposals are actually landing something, rather than instead of them.
