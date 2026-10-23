"""The mock SMS provider: one text per Idempotency-Key, and replies that survive a restart.
Uses Redis database 15 and a tiny stand-in answer key, so no dataset is needed."""

import pytest
from fastapi.testclient import TestClient

KEY = "test-sms_api_key"


@pytest.fixture
def provider(redis_client, tmp_path, monkeypatch):
    from fraud_agents.services.sms import provider as sms
    answers = tmp_path / "answers.csv"
    answers.write_text("trans_num,is_fraud\ntxn-fraud,1\ntxn-genuine,0\n")
    monkeypatch.setattr(sms, "TEST_FILE", str(answers))
    monkeypatch.setattr(sms, "REPLY_DELAY_SECONDS", (600, 600))   # replies far in the future: nothing fires
    monkeypatch.setattr(sms, "GATEWAY_REPLY_URL", "http://127.0.0.1:9/webhooks/sms-replies")  # never a real bank
    monkeypatch.setattr(sms, "NO_REPLY_PERCENT", 0)       # every simulated customer answers clearly,
    monkeypatch.setattr(sms, "UNCLEAR_PERCENT", 0)        # so the tests don't depend on a dice roll
    return sms


def send(client, body, key=None):
    headers = {"X-Api-Key": KEY} | ({"Idempotency-Key": key} if key else {})
    return client.post("/messages", json=body, headers=headers)


def test_the_same_key_sends_one_text(provider, redis_client):
    with TestClient(provider.app) as client:
        first = send(client, {"to": "card-1", "text": "Bank: hello", "txn_id": "t1"}, key="k1")
        again = send(client, {"to": "card-1", "text": "Bank: hello", "txn_id": "t1"}, key="k1")
    assert first.json()["status"] == "queued" and again.json()["status"] == "duplicate"
    assert again.json()["message_id"] == first.json()["message_id"]
    assert redis_client.xlen(provider.SMS_LOG) == 1


def test_a_question_and_its_reply_are_saved_together(provider, redis_client):
    with TestClient(provider.app) as client:
        response = send(client, {"to": "card-1", "text": "Bank: was this you?", "txn_id": "txn-fraud",
                                 "reference": "ref-1", "expects_reply": True}, key="k2")
    assert response.status_code == 202
    assert redis_client.xlen(provider.SMS_LOG) == 1                  # the text...
    assert redis_client.zcard(provider.PENDING_REPLIES) == 1         # ...and its reply, in one transaction
    assert redis_client.get("sms:idem:k2") == response.json()["message_id"]


def test_replies_half_done_at_a_crash_are_picked_up_again(provider, redis_client):
    import time
    item = ('{"to":"card-1","text":"Bank: was this you?","reference":"r","txn_id":"txn-fraud",'
            '"expects_reply":true,"message_id":"sms-out-abc"}')
    redis_client.hset(provider.REPLYING, item, 1)        # it was being answered when the provider "crashed"
    with TestClient(provider.app):                        # the provider starts again...
        for _ in range(40):                               # ...and the customer's reply is written again
            replies = [f for _, f in redis_client.xrange(provider.SMS_LOG) if f["direction"] == "in"]
            if replies:
                break
            time.sleep(0.1)
    assert [r["message_id"] for r in replies] == ["sms-in-abc"]    # one reply, with an ID tied to the text
    assert replies[0]["text"] in provider.SAYS_NO                    # txn-fraud: the customer says "not me"
    # The bank was unreachable, so the reply is still "being answered": a restart will deliver it.
    assert redis_client.hlen(provider.REPLYING) == 1


def test_a_wrong_api_key_is_refused(provider):
    with TestClient(provider.app) as client:
        assert client.post("/messages", json={"to": "c", "text": "t"}, headers={"X-Api-Key": "no"}).status_code == 401


def test_the_same_text_always_gets_the_same_answer(provider, redis_client):
    import asyncio
    msg = provider.OutgoingText(to="card-1", text="Bank: was this you?", reference="r", txn_id="txn-genuine",
                                expects_reply=True, message_id="sms-out-same")
    replies = []

    async def fake_webhook(state, reply):
        replies.append(reply["text"])
        return True

    class State:
        redis = __import__("redis.asyncio", fromlist=["Redis"]).Redis.from_url(
            "redis://localhost:6379/15", decode_responses=True)
        is_fraud = {"txn-genuine": 0}

    async def run():
        provider.send_reply_webhook = fake_webhook
        await provider._customer_answers(State, msg)
        await provider._customer_answers(State, msg)     # a retry of the same text
        await State.redis.aclose()

    original = provider.send_reply_webhook
    try:
        asyncio.run(run())
    finally:
        provider.send_reply_webhook = original
    assert len(replies) == 2 and replies[0] == replies[1] and replies[0] in provider.SAYS_YES
    assert sum(f["direction"] == "in" for _, f in redis_client.xrange(provider.SMS_LOG)) == 1   # logged once


def test_a_reply_the_bank_could_not_take_waits_to_be_tried_again(provider, redis_client, monkeypatch):
    import time

    async def bank_down(state, reply):
        return False

    monkeypatch.setattr(provider, "send_reply_webhook", bank_down)
    item = ('{"to":"card-1","text":"Bank: was this you?","reference":"r","txn_id":"txn-fraud",'
            '"expects_reply":true,"message_id":"sms-out-down"}')
    redis_client.zadd(provider.PENDING_REPLIES, {item: time.time()})          # due now
    with TestClient(provider.app):
        for _ in range(40):
            if redis_client.zscore(provider.PENDING_REPLIES, item) and \
                    redis_client.zscore(provider.PENDING_REPLIES, item) > time.time() + 5:
                break
            time.sleep(0.1)
    assert redis_client.zscore(provider.PENDING_REPLIES, item) > time.time() + 5   # back in the queue, later
    assert redis_client.hlen(provider.REPLYING) == 0
