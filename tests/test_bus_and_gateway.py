"""The message bus (Redis Streams) and the payment gateway. These need Redis: they skip
themselves if it isn't running (start it with: docker compose up -d). CI starts one."""

import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from fraud_agents.shared import bus, signing
from fraud_agents.shared.events import (TransactionDecided, TransactionDecidedPayload, new_event_id,
                                        now_utc, parse_event)


@pytest.fixture
def stream(redis_client):
    name = f"events:test.{uuid.uuid4().hex[:8]}"
    yield name
    redis_client.delete(name)


def decided(decision="hold"):
    return TransactionDecided(event_id=new_event_id(), published_at=now_utc(), txn_id=f"t-{uuid.uuid4().hex[:6]}",
                              occurred_at=now_utc(), payload=TransactionDecidedPayload(
                                  decision=decision, rule_id="x", rules_version="t", score=0.95,
                                  based_on_event_id="s"))


def test_every_event_is_handled_once_even_if_published_twice(redis_client, stream):
    event = decided()
    for _ in range(2):    # e.g. a retry after a crash publishes the same form again
        redis_client.xadd(stream, {"event": event.model_dump_json()})
    seen = []
    bus.ensure_group(redis_client, stream, "g1")
    bus._one_round(redis_client, stream, "g1", seen.append, "c1", 100, None)   # reads both copies
    assert [e.event_id for e in seen] == [event.event_id]                    # the second is skipped
    assert redis_client.xpending(stream, "g1")["pending"] == 0                # and both are done


def test_two_groups_each_get_every_event(redis_client, stream):
    event = decided()
    redis_client.xadd(stream, {"event": event.model_dump_json()})
    a, b = [], []
    for group, seen in (("ga", a), ("gb", b)):
        bus.ensure_group(redis_client, stream, group)
        bus._one_round(redis_client, stream, group, seen.append, "c1", 100, None)
    assert a == b == [event]


# Why Redis Streams and not Redis's own "Pub/Sub" feature: a Pub/Sub message is lost if no
# subscriber is listening at that moment. A stream keeps it until each subscriber has it.
def test_a_subscriber_that_was_down_still_gets_the_events_it_missed(redis_client, stream):
    bus.ensure_group(redis_client, stream, "late")         # subscribed earlier, then stopped
    events = [decided() for _ in range(3)]
    for event in events:                                   # published while nobody was reading
        redis_client.xadd(stream, {"event": event.model_dump_json()})
    seen = []
    bus._one_round(redis_client, stream, "late", seen.append, "c1", 100, None)   # back up
    assert seen == events


def test_an_event_a_crashed_subscriber_never_finished_is_delivered_again(redis_client, stream):
    event = decided()
    redis_client.xadd(stream, {"event": event.model_dump_json()})
    bus.ensure_group(redis_client, stream, "g")
    redis_client.xreadgroup("g", "crashed", {stream: ">"}, count=1)   # read it, crashed before XACK
    seen = []
    bus._one_round(redis_client, stream, "g", seen.append, "c2", 100, None, 10, 0)   # reclaim now
    assert seen == [event]
    assert redis_client.xpending(stream, "g")["pending"] == 0          # finished this time


def test_a_failing_event_is_handed_over_then_set_aside(redis_client, stream):
    event = decided()
    redis_client.xadd(stream, {"event": event.model_dump_json()})
    handed_over = []
    dead_before = redis_client.xlen(bus.DEAD_LETTER_STREAM)

    def always_fails(e):
        raise RuntimeError("the AI is down (test)")

    for _ in range(bus.MAX_DELIVERIES + 2):   # retried, then given up on
        bus._one_round(redis_client, stream, "g2", always_fails, "c1", 50, None, 1, 10,
                       lambda e, why: handed_over.append((e.event_id, why)))
        time.sleep(0.05)
    assert [h[0] for h in handed_over] == [event.event_id]
    assert redis_client.xlen(bus.DEAD_LETTER_STREAM) == dead_before + 1
    assert redis_client.xpending(stream, "g2")["pending"] == 0


def test_if_handing_over_fails_the_event_is_kept_for_another_try(redis_client, stream):
    event = decided()
    redis_client.xadd(stream, {"event": event.model_dump_json()})

    def always_fails(e):
        raise RuntimeError("the AI is down (test)")

    def handover_fails(e, why):
        raise RuntimeError("Redis hiccup while escalating (test)")

    for _ in range(bus.MAX_DELIVERIES + 2):
        bus._one_round(redis_client, stream, "g3", always_fails, "c1", 50, None, 1, 10, handover_fails)
        time.sleep(0.05)
    assert redis_client.xlen(bus.DEAD_LETTER_STREAM) == 0          # not set aside...
    assert redis_client.xpending(stream, "g3")["pending"] == 1      # ...still waiting to be handed over


# ---------- the gateway ----------

def payment_body(**changes):
    body = {"trans_num": f"test-{uuid.uuid4().hex}", "trans_date_trans_time": "2020-06-21 12:14:25",
            "cc_num": "4613314721966", "merchant": "fraud_Kilback LLC", "category": "shopping_net",
            "amt": "843.91", "lat": 40.0, "long": -75.0, "city_pop": 5000, "merch_lat": 40.1,
            "merch_long": -75.2}
    body.update(changes)
    return json.dumps(body).encode()


def post(client, body, secret=None):
    import os
    ts = str(int(time.time()))
    sig = signing.sign(secret or os.environ["WEBHOOK_SECRET"], ts, body)
    return client.post("/webhooks/payments", content=body,
                       headers={"X-Timestamp": ts, "X-Signature": sig, "Content-Type": "application/json"})


@pytest.fixture
def gateway(redis_client):
    from fraud_agents.services.gateway.app import app
    with TestClient(app) as client:
        yield client


def test_a_signed_payment_becomes_an_event_with_a_card_token(gateway, redis_client):
    body = payment_body()
    response = post(gateway, body)
    assert response.status_code == 202, response.text
    txn = json.loads(body)["trans_num"]
    entries = redis_client.xrevrange(bus.stream_for("transaction.received"), count=50)
    event = next(parse_event(f["event"]) for _, f in entries if parse_event(f["event"]).txn_id == txn)
    assert event.payload.card_id.startswith("card-") and "4613314721966" not in event.model_dump_json()
    assert post(gateway, body).json()["status"] == "duplicate"


def test_a_wrong_signature_is_refused(gateway):
    assert post(gateway, payment_body(), secret="not-the-secret").status_code == 401


def test_the_answer_key_is_refused(gateway):
    assert post(gateway, payment_body(is_fraud=1)).status_code == 422
