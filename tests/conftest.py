"""Shared test setup. Run all tests from the project folder with:  pytest

The tests need no Docker, no Llama and no dataset. The few that need Redis use the
`redis_client` fixture: they skip themselves when Redis isn't running (on CI they fail
instead, so a broken Redis can't give a green run). They use Redis database 15, never the
live system's database 0, so tests can't feed fake payments into a running system."""

import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Test-only settings (set before any project code reads them).
os.environ["REDIS_URL"] = "redis://localhost:6379/15"   # a separate database for tests
for name in ("CARD_TOKEN_KEY", "WEBHOOK_SECRET", "SMS_API_KEY", "SMS_WEBHOOK_SECRET"):
    os.environ[name] = f"test-{name.lower()}"
for name, value in (("POSTGRES_USER", "fraud"), ("POSTGRES_PASSWORD", "test"), ("POSTGRES_DB", "fraud")):
    os.environ[name] = value


@pytest.fixture(autouse=True)
def project_folder(monkeypatch):
    from fraud_agents.shared import config
    monkeypatch.setattr(config, "load_env", lambda *a, **k: None)   # never read the real .env
    monkeypatch.chdir(ROOT)   # the code reads models/ relative to the project folder


@pytest.fixture
def redis_client():
    """A Redis connection, or skip the test if Redis isn't running."""
    import redis
    from fraud_agents.shared import bus
    r = bus.connect()
    try:
        r.ping()
    except redis.RedisError:
        if os.environ.get("CI"):
            pytest.fail("Redis should be running on CI")
        pytest.skip("Redis is not running (start it with: docker compose up -d)")
    r.flushdb()                # database 15 only: a clean start for every test
    yield r
    r.flushdb()


@pytest.fixture
def payment():
    """One real-looking payment form (from the test set's first fraud)."""
    from fraud_agents.shared.events import (TransactionReceived, TransactionReceivedPayload,
                                            new_event_id, now_utc)
    return TransactionReceived(
        event_id=new_event_id(), published_at=now_utc(), txn_id="0b242abb623afc578575680df30655b9",
        occurred_at=datetime(2020, 6, 21, 22, 6, 39, tzinfo=timezone.utc),
        payload=TransactionReceivedPayload(
            card_id="card-demo-1", amount=Decimal("843.91"), category="shopping_net",
            merchant="fraud_Kilback LLC", merchant_lat=40.1, merchant_long=-75.2,
            home_lat=40.0, home_long=-75.0, city_pop=5000))
