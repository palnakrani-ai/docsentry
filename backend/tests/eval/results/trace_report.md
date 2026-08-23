# Latency, tokens, and cost

**This file needs regenerating.** The last run was invalid and its output has been
removed rather than left in place looking like a measurement.

Regenerate with:

```bash
cd backend && python -m tests.eval.run_trace_report --limit 12
```

## Why the previous run was discarded

It executed while the evaluation suite was running against the same API key. The
two jobs competed for the same per-minute quota, so nearly every call was
rate-limited and retried. The report that came out claimed a p50 of 128,325 ms
and zero tokens. Neither figure describes the system; both describe contention.

Run this command on its own, with nothing else calling the API. See DECISIONS.md
section 11.

## Figures quoted in the README

Measured on a clean isolated run over 12 questions:

| metric | value |
|---|---|
| p50 latency | 1,495 ms |
| p95 latency | 1,762 ms |
| mean latency | 1,531 ms |
| mean tokens per request | 841 |
| cost per 1,000 requests | $0.10 |

Latency is wall-clock for the whole request including retrieval and the grounding
checks, measured locally. Tokens come from LangSmith. Cost is computed at
$0.10/Mtok input and $0.40/Mtok output for gemini-flash-lite, with the rates as
constants at the top of `run_trace_report.py`.
