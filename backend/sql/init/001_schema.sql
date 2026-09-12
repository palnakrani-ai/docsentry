-- Schema for the DocSentry data layer.
--
-- Runs automatically on first boot of the local dev container. Against Supabase
-- it is applied by hand once, from the SQL editor. It is written to be safe to
-- run more than once either way.

-- pgvector. langchain-postgres creates its own tables on first use, but the
-- extension has to exist before it can.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per answered question. This is the audit trail the application has
-- never had: today a refusal leaves nothing behind except a log line, so there
-- is no way to answer "how often does it refuse, and why" after the fact.
--
-- Written off the response path, so a failure here must never fail a request.
CREATE TABLE IF NOT EXISTS events (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    question        TEXT         NOT NULL,
    answered        BOOLEAN      NOT NULL,
    refusal_reason  TEXT,

    -- Which chunks the answer cited, as returned to the user. Empty for a
    -- refusal. JSONB rather than a join table: it is read for dashboards and
    -- for spot checks, never joined against.
    citations       JSONB        NOT NULL DEFAULT '[]'::jsonb,

    -- Flags the guardrails raised: off_topic, model_unavailable, injection.
    flags           TEXT[]       NOT NULL DEFAULT '{}',

    -- Best retrieval score for the question. Stored because the grounding
    -- threshold is tuned against exactly this number, and tuning it blind
    -- against a test set is how it drifts out of step with production.
    top_score       DOUBLE PRECISION,

    latency_ms      INTEGER      NOT NULL
);

-- Grafana reads this table by time window, which is the only access pattern
-- that matters.
CREATE INDEX IF NOT EXISTS events_created_at_idx ON events (created_at DESC);

-- Refusal rate over a window is the panel most worth having, and it scans only
-- the rows that refused.
CREATE INDEX IF NOT EXISTS events_refused_idx
    ON events (created_at DESC)
    WHERE answered = false;
