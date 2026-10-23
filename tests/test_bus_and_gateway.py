"""The message bus (Redis Streams) and the payment gateway. These need Redis: they skip
themselves if it isn't running (start it with: docker compose up -d). CI starts one."""

import json
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from fraud_agents.shared import bus, signing
from fraud_agents.shared.events import (TransactionDecided, TransactionDecidedPayload,
                                        TransactionResolved, TransactionResolvedPayload, new_event_id,
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
def gateway(redis_client, monkeypatch):
    from fraud_agents.services.gateway import app as gateway_app
    monkeypatch.setattr(gateway_app, "DECISION_DEADLINE_MS", 2000)   # roomy for slow CI machines
    with TestClient(gateway_app.app) as client:
        yield client


def received(redis_client, txn: str):
    """The transaction.received event the gateway published for this payment."""
    for _ in range(100):
        for _, f in redis_client.xrevrange(bus.stream_for("transaction.received"), count=50):
            event = parse_event(f["event"])
            if event.txn_id == txn:
                return event
        time.sleep(0.02)
    raise AssertionError(f"no transaction.received for {txn}")


def fake_decision(redis_client, txn: str, decision: str | None = None, delay: float = 0) -> threading.Thread:
    """Plays the Risk Scoring + Decision agents: publishes a verdict once the payment arrives.
    decision=None runs the REAL Decision rules on a high score (0.99), so a confirmed retry
    is approved only because of the customer_confirmed_retry rule."""
    from fraud_agents.agents.decision.agent import decide, load_rules
    from fraud_agents.shared.events import RiskScored, RiskScoredPayload

    def run():
        event = received(redis_client, txn)
        time.sleep(delay)
        if decision is None:
            scored = RiskScored(event_id=new_event_id(), published_at=now_utc(), txn_id=txn,
                                occurred_at=event.occurred_at, payload=RiskScoredPayload(
                                    score=0.99, model_version="test",
                                    confirmed_retry_of=event.payload.confirmed_retry_of))
            bus.publish(redis_client, decide(scored, load_rules()))
            return
        bus.publish(redis_client, TransactionDecided(
            event_id=new_event_id(), published_at=now_utc(), txn_id=txn, occurred_at=event.occurred_at,
            payload=TransactionDecidedPayload(decision=decision, rule_id="test", rules_version="t",
                                              score=0.95, based_on_event_id="s")))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def status(client, txn: str, secret=None):
    import os
    ts = str(int(time.time()))
    sig = signing.sign(secret or os.environ["WEBHOOK_SECRET"], ts, f"GET /payments/{txn}".encode())
    return client.get(f"/payments/{txn}", headers={"X-Timestamp": ts, "X-Signature": sig})


def resolve(redis_client, txn: str, resolution: str):
    bus.publish(redis_client, TransactionResolved(
        event_id=new_event_id(), published_at=now_utc(), txn_id=txn, occurred_at=now_utc(),
        payload=TransactionResolvedPayload(resolution=resolution, rule_id="test",
                                           rules_version="t", based_on_event_id="v")))


def wait_for_outcome(client, txn: str) -> dict:
    for _ in range(100):                            # the gateway's listener picks it up
        body = status(client, txn).json()
        if body["outcome"]:
            return body
        time.sleep(0.05)
    raise AssertionError("the gateway never recorded the outcome")


def test_a_signed_payment_becomes_an_event_with_a_card_token(gateway, redis_client):
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    fake_decision(redis_client, txn, "approve")
    response = post(gateway, body)
    assert response.status_code == 200, response.text
    event = received(redis_client, txn)
    assert event.payload.card_id.startswith("card-") and "4613314721966" not in event.model_dump_json()


def test_the_payment_source_gets_an_answer(gateway, redis_client):
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    fake_decision(redis_client, txn, "approve")
    answer = post(gateway, body).json()
    assert (answer["status"], answer["reason"]) == ("approved", "low_risk")
    assert answer["decided_in_ms"] < 2000
    again = post(gateway, body).json()             # a resent webhook gets the SAME answer
    assert (again["status"], again["reason"]) == ("approved", "low_risk")


def test_two_copies_sent_at_the_same_time_get_one_answer_and_one_event(gateway, redis_client):
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    fake_decision(redis_client, txn, "hold", delay=0.2)
    answers = []
    copies = [threading.Thread(target=lambda: answers.append(post(gateway, body).json())) for _ in range(2)]
    for c in copies:
        c.start()
    for c in copies:
        c.join()
    assert [a["reason"] for a in answers] == ["verify_with_customer"] * 2
    events = [e for _, f in redis_client.xrange(bus.stream_for("transaction.received"))
              if (e := parse_event(f["event"])).txn_id == txn]
    assert len(events) == 1


def test_no_verdict_in_time_is_a_soft_decline_and_a_late_verdict_does_not_change_it(gateway, redis_client,
                                                                                   monkeypatch):
    from fraud_agents.services.gateway import app as gateway_app
    monkeypatch.setattr(gateway_app, "DECISION_DEADLINE_MS", 100)
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    answer = post(gateway, body).json()
    assert (answer["status"], answer["decline_type"], answer["reason"]) == \
        ("declined", "soft", "no_decision_in_time")
    fake_decision(redis_client, txn, "approve").join()              # the verdict comes too late
    time.sleep(0.3)
    after = status(gateway, txn).json()
    assert (after["answer"], after["status"]) == ("declined_soft", gateway_app.NO_VERDICT_TEXT)


def test_a_resend_after_a_crash_mid_wait_gets_a_soft_decline(gateway, redis_client, monkeypatch):
    from fraud_agents.services.gateway import app as gateway_app
    monkeypatch.setattr(gateway_app, "DECISION_DEADLINE_MS", 100)
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    # The first request claimed the payment 5 s ago, then the gateway died before answering.
    redis_client.set(f"gateway:seen:{txn}", time.time() - 5)
    answer = post(gateway, body).json()
    assert (answer["status"], answer["reason"]) == ("declined", "no_decision_in_time")


def test_the_gateway_does_not_replay_old_verdicts(redis_client, monkeypatch):
    from fraud_agents.services.gateway import app as gateway_app
    redis_client.hset("payment:old-txn", "answer", "approved:low_risk")
    bus.publish(redis_client, TransactionDecided(       # a verdict from before the gateway started
        event_id=new_event_id(), published_at=now_utc(), txn_id="old-txn", occurred_at=now_utc(),
        payload=TransactionDecidedPayload(decision="hold", rule_id="test", rules_version="t",
                                          score=0.95, based_on_event_id="s")))
    with TestClient(gateway_app.app):
        time.sleep(0.5)
    assert redis_client.hget("payment:old-txn", "decision") is None


def test_a_held_payment_is_soft_declined_and_its_confirmed_retry_approved_once(gateway, redis_client):
    first = payment_body()
    txn = json.loads(first)["trans_num"]
    fake_decision(redis_client, txn, "hold")
    answer = post(gateway, first).json()
    assert (answer["status"], answer["decline_type"], answer["reason"]) == \
        ("declined", "soft", "verify_with_customer")
    assert status(gateway, txn).json()["status"] == "declined for now: waiting for the customer to confirm"

    resolve(redis_client, txn, "released")             # the customer said YES
    assert wait_for_outcome(gateway, txn)["outcome"] == "retry_allowed"

    # A different amount is a different payment: scored as usual, the YES is not used up.
    other = payment_body(trans_num=txn + "-other", amt="999.99")
    fake_decision(redis_client, txn + "-other")
    assert post(gateway, other).json()["reason"] == "verify_with_customer"
    assert received(redis_client, txn + "-other").payload.confirmed_retry_of is None

    # The customer taps again: same card, shop and amount. The REAL rules approve it.
    retry = payment_body(trans_num=txn + "-retry")
    fake_decision(redis_client, txn + "-retry")
    answer = post(gateway, retry).json()
    assert (answer["status"], answer["reason"]) == ("approved", "customer_confirmed_retry")
    assert received(redis_client, txn + "-retry").payload.confirmed_retry_of == txn

    # The YES allows ONE retry only.
    third = payment_body(trans_num=txn + "-again")
    fake_decision(redis_client, txn + "-again")
    assert post(gateway, third).json()["reason"] == "verify_with_customer"


def test_a_confirmed_retry_that_times_out_keeps_the_customers_yes(gateway, redis_client, monkeypatch):
    from fraud_agents.services.gateway import app as gateway_app
    first = payment_body()
    txn = json.loads(first)["trans_num"]
    fake_decision(redis_client, txn, "hold")
    post(gateway, first)
    resolve(redis_client, txn, "released")
    wait_for_outcome(gateway, txn)
    monkeypatch.setattr(gateway_app, "DECISION_DEADLINE_MS", 100)
    assert post(gateway, payment_body(trans_num=txn + "-slow")).json()["reason"] == "no_decision_in_time"
    monkeypatch.setattr(gateway_app, "DECISION_DEADLINE_MS", 2000)
    fake_decision(redis_client, txn + "-retry")
    assert post(gateway, payment_body(trans_num=txn + "-retry")).json()["reason"] == "customer_confirmed_retry"


def test_the_yes_is_bound_to_the_shop_and_expires(gateway, redis_client):
    first = payment_body()
    txn = json.loads(first)["trans_num"]
    fake_decision(redis_client, txn, "hold")
    post(gateway, first)
    resolve(redis_client, txn, "released")
    wait_for_outcome(gateway, txn)
    # Same card, shop name and amount, but another shop type: not the confirmed payment.
    fake_decision(redis_client, txn + "-cat")
    assert post(gateway, payment_body(trans_num=txn + "-cat", category="travel")).json()["reason"] \
        == "verify_with_customer"
    # The allowance expires: shorten it, wait, and the same retry is scored as usual.
    allow = next(redis_client.scan_iter("allow:*"))
    redis_client.pexpire(allow, 50)
    time.sleep(0.2)
    fake_decision(redis_client, txn + "-late")
    assert post(gateway, payment_body(trans_num=txn + "-late")).json()["reason"] == "verify_with_customer"


def test_a_redis_error_while_publishing_does_not_lose_the_payment(gateway, redis_client, monkeypatch):
    import redis as redis_lib
    from fraud_agents.services.gateway import app as gateway_app
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    real_xadd = gateway_app.app.state.redis.xadd

    async def broken_xadd(*args, **kwargs):
        raise redis_lib.ConnectionError("Redis blinked")
    monkeypatch.setattr(gateway_app.app.state.redis, "xadd", broken_xadd)
    with pytest.raises(redis_lib.ConnectionError):
        post(gateway, body)
    monkeypatch.setattr(gateway_app.app.state.redis, "xadd", real_xadd)
    fake_decision(redis_client, txn, "approve")                 # the processor resends it
    assert post(gateway, body).json()["reason"] == "low_risk"  # handled as new, not lost


@pytest.mark.parametrize("resolution, text", [("blocked", "declined: blocked"),
                                              ("sent_to_analyst", "declined for now: with a fraud analyst")])
def test_the_status_says_what_happened(gateway, redis_client, resolution, text):
    body = payment_body()
    txn = json.loads(body)["trans_num"]
    fake_decision(redis_client, txn, "hold")
    post(gateway, body)
    resolve(redis_client, txn, resolution)
    assert wait_for_outcome(gateway, txn)["status"] == text


def test_the_status_needs_a_signature(gateway):
    assert status(gateway, "anything", secret="not-the-secret").status_code == 401


def test_a_wrong_signature_is_refused(gateway):
    assert post(gateway, payment_body(), secret="not-the-secret").status_code == 401


def test_the_answer_key_is_refused(gateway):
    assert post(gateway, payment_body(is_fraud=1)).status_code == 422
