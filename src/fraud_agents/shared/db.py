"""The PostgreSQL database: the permanent record (audit log) of every event."""

import os

import psycopg

from fraud_agents.shared import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,           -- one row per event, never twice
    event_type    TEXT NOT NULL,
    txn_id        TEXT NOT NULL,              -- links every event about one payment
    producer      TEXT NOT NULL,              -- which agent or service made it
    occurred_at   TIMESTAMPTZ NOT NULL,       -- when the payment happened
    published_at  TIMESTAMPTZ NOT NULL,       -- when the event was sent
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    score         DOUBLE PRECISION,           -- filled for risk.scored and transaction.decided
    decision      TEXT,                       -- filled for transaction.decided
    body          JSONB NOT NULL              -- the whole form, exactly as it was sent
);
CREATE INDEX IF NOT EXISTS events_txn_idx ON events (txn_id);
CREATE INDEX IF NOT EXISTS events_published_idx ON events (published_at DESC);

CREATE TABLE IF NOT EXISTS llm_calls (      -- one row per call to a language model
    id             BIGSERIAL PRIMARY KEY,
    called_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent          TEXT,                     -- which agent asked
    txn_id         TEXT,                     -- which payment it was about
    provider       TEXT,                     -- ollama or anthropic
    model          TEXT,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    ok             BOOLEAN NOT NULL,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_txn_idx ON llm_calls (txn_id);
"""


def connect() -> psycopg.Connection:
    """Connect using the settings in .env (the same ones Docker Compose uses)."""
    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=config.require("POSTGRES_USER"),
        password=config.require("POSTGRES_PASSWORD"),
        dbname=config.require("POSTGRES_DB"),
        autocommit=True,
    )


def create_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
