"""A pretend SMS provider with a pretend customer (Part 2 · Step 5).

It plays two outside parties, so the bank side can be tested end to end:

1. The SMS company (like Twilio). The bank sends it texts:
       POST /messages        (bank -> provider, with the bank's API key)
   and it forwards customers' replies back to the bank as a signed webhook:
       POST <bank gateway>/webhooks/sms-replies   (provider -> bank, HMAC signature)

2. The customer's phone. When a text asks "was this you?", the simulated customer
   answers after a few seconds. The customer knows the truth (they know whether
   they made the payment), so the answer comes from the dataset's answer key.
   To exercise every path, some customers reply unclearly or not at all.

Every text in and out is kept in the Redis stream "sms:log" for the System Monitor.

Idempotency (Part 3 · Steps 3 and 6): the bank may send an "Idempotency-Key" header. A
second request with the same key is not sent again: it gets the first request's message_id.
Sending is ONE Redis transaction (record the text + schedule the customer's reply + save the
key): all of it happens or none of it does, so a crash can never leave a key without a text
(a retry would wrongly say "duplicate") or a text without a key.

Crash-safe replies (Part 3 · Step 6): a reply the customer will send is kept in Redis
("sms:pending_replies", by due time), not only in memory. A delivery loop answers the due
ones; anything half-done when the provider stopped is picked up again when it restarts.

Run from the project folder:
    uvicorn fraud_agents.services.sms.provider:app --port 8100
"""

import asyncio
import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import pandas as pd
import redis.asyncio as aioredis
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from fraud_agents.shared import config, signing

log = logging.getLogger("sms-provider")

SMS_LOG = "sms:log"
PENDING_REPLIES = "sms:pending_replies"   # sorted set: a reply to send -> when it is due
REPLYING = "sms:replying"                 # hash: replies being answered right now
RETRY_REPLY_AFTER = 30                    # seconds: when the bank couldn't be reached, try again
# Atomic "claim": remove from the waiting set and mark as being answered, or do nothing.
CLAIM = ("if redis.call('zrem', KEYS[1], ARGV[1]) == 1 then "
         "redis.call('hset', KEYS[2], ARGV[1], ARGV[2]) return 1 end return 0")
IDEMPOTENCY_DAYS = 7
REPLIES_LOGGED = "sms:replies_logged"     # reply IDs already written to sms:log
GATEWAY_REPLY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000") + "/webhooks/sms-replies"
TEST_FILE = "data/raw/fraudTest.csv"

# How the simulated customers behave (out of 100). The rest give a clear yes or no.
UNCLEAR_PERCENT = 10      # e.g. "who is this?"
NO_REPLY_PERCENT = 10     # they never answer
REPLY_DELAY_SECONDS = (3, 8)

SAYS_YES = ["YES that was me", "Yes, it was me", "yes I made that payment", "Y"]
SAYS_NO = ["NO that wasn't me!", "No. I did not make this payment", "no!! not me", "N"]
UNCLEAR = ["who is this?", "what payment?", "call me later", "hmm not sure"]


class OutgoingText(BaseModel):
    to: str = Field(min_length=1)          # the customer (in this project: their card token)
    text: str = Field(min_length=1, max_length=480)
    reference: str | None = None           # the bank's reference (verification.requested event_id)
    txn_id: str | None = None              # which payment the text is about
    expects_reply: bool = False            # True for "was this you?" questions
    message_id: str | None = None          # filled in by the provider when it sends the text

    @model_validator(mode="after")
    def reply_needs_reference(self):
        # A question must say which case and payment it is about, or the reply can't be matched.
        if self.expects_reply and not (self.reference and self.txn_id):
            raise ValueError("expects_reply needs both reference and txn_id")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app.state.api_key = config.require("SMS_API_KEY")
    app.state.webhook_secret = config.require("SMS_WEBHOOK_SECRET")
    app.state.redis = aioredis.Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    # The customers' memory: which payments they really made (the answer key).
    df = pd.read_csv(TEST_FILE, usecols=["trans_num", "is_fraud"])
    app.state.is_fraud = dict(zip(df["trans_num"], df["is_fraud"]))
    app.state.http = httpx.AsyncClient(timeout=10)
    app.state.tasks = set()
    # Replies that were being answered when the provider last stopped: answer them again.
    for item in await app.state.redis.hkeys(REPLYING):
        await app.state.redis.zadd(PENDING_REPLIES, {item: time.time()})
        await app.state.redis.hdel(REPLYING, item)
    loop = asyncio.create_task(deliver_due_replies(app.state))
    log.info("SMS provider ready: POST /messages (customers know %d payments; %d replies waiting)",
             len(app.state.is_fraud), await app.state.redis.zcard(PENDING_REPLIES))
    yield
    loop.cancel()
    await app.state.http.aclose()
    await app.state.redis.aclose()


app = FastAPI(title="Mock SMS provider", lifespan=lifespan)


def log_entry(direction: str, who: str, text: str, message_id: str, txn_id: str | None) -> dict:
    return {"direction": direction, "who": who, "text": text, "message_id": message_id,
            "txn_id": txn_id or "", "at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}


async def record(r, direction: str, who: str, text: str, message_id: str, txn_id: str | None):
    await r.xadd(SMS_LOG, log_entry(direction, who, text, message_id, txn_id), maxlen=5000, approximate=True)


async def send_reply_webhook(state, reply: dict) -> bool:
    """Deliver a customer's reply to the bank, signed, retrying a few times.
    True = delivered (or refused for good by the bank); False = the bank couldn't be reached."""
    body = json.dumps(reply).encode()
    for attempt in range(1, 6):
        ts = str(int(time.time()))
        headers = {"Content-Type": "application/json", "X-Timestamp": ts,
                   "X-Signature": signing.sign(state.webhook_secret, ts, body)}
        try:
            response = await state.http.post(GATEWAY_REPLY_URL, content=body, headers=headers)
            if 200 <= response.status_code < 300:
                log.info("Reply %s delivered to the bank: HTTP %s", reply["message_id"], response.status_code)
                return True
            if response.status_code < 500:   # the bank refused it: retrying won't help
                log.error("Bank refused reply %s: HTTP %s %s", reply["message_id"],
                          response.status_code, response.text[:200])
                return True
        except httpx.TransportError:
            pass
        await asyncio.sleep(2 ** attempt)
    log.warning("Could not deliver reply %s to the bank: will try again in %d s",
                reply["message_id"], RETRY_REPLY_AFTER)
    return False


async def deliver_due_replies(state) -> None:
    """Every half second: answer the replies that are due. Runs for as long as the provider."""
    while True:
        try:
            for item in await state.redis.zrangebyscore(PENDING_REPLIES, 0, time.time(), start=0, num=20):
                # Move it from "waiting" to "being answered" in ONE step, so a crash can't lose it.
                if await state.redis.eval(CLAIM, 2, PENDING_REPLIES, REPLYING, item, time.time()):
                    task = asyncio.create_task(customer_answers(state, item))
                    state.tasks.add(task)               # keep a reference until it finishes
                    task.add_done_callback(state.tasks.discard)
        except Exception:
            log.exception("Could not read the waiting replies; trying again")
        await asyncio.sleep(0.5)


async def customer_answers(state, item: str) -> None:
    """The simulated customer reads the text and (maybe) answers. Errors are logged, not lost."""
    try:
        done = await _customer_answers(state, OutgoingText.model_validate_json(item))
    except asyncio.CancelledError:
        raise   # the provider is stopping: leave it in REPLYING, so it is answered after a restart
    except Exception:
        log.exception("Simulated customer failed on %s", item[:120])
        done = True
    async with state.redis.pipeline(transaction=True) as pipe:
        if not done:   # the bank was unreachable: put it back to wait, don't drop it
            pipe.zadd(PENDING_REPLIES, {item: time.time() + RETRY_REPLY_AFTER})
        pipe.hdel(REPLYING, item)
        await pipe.execute()


async def _customer_answers(state, msg: OutgoingText) -> bool:
    """Returns False only if the bank couldn't be reached (then the reply is tried again)."""
    message_id = msg.message_id
    truth = state.is_fraud.get(msg.txn_id)
    dice = random.Random(message_id)           # the same text always gets the same answer, even on a retry
    roll = dice.randint(1, 100)
    if roll <= NO_REPLY_PERCENT:
        log.info("Customer %s did not reply to %s", msg.to, message_id)
        return True
    if truth is None or roll <= NO_REPLY_PERCENT + UNCLEAR_PERCENT:
        answer = dice.choice(UNCLEAR)          # unknown payment: a real customer would be confused
    elif truth == 1:
        answer = dice.choice(SAYS_NO)          # it really was fraud: "not me"
    else:
        answer = dice.choice(SAYS_YES)         # they really made it: "yes, me"
    reply_id = "sms-in-" + message_id.removeprefix("sms-out-")   # one reply per text, even if retried
    if not await state.redis.sismember(REPLIES_LOGGED, reply_id):   # log each reply once, even if retried
        await record(state.redis, "in", msg.to, answer, reply_id, msg.txn_id)
        await state.redis.sadd(REPLIES_LOGGED, reply_id)
        await state.redis.expire(REPLIES_LOGGED, IDEMPOTENCY_DAYS * 24 * 3600)   # doesn't grow forever
    return await send_reply_webhook(state, {
        "message_id": reply_id, "from": msg.to, "text": answer,
        "in_reply_to": message_id, "reference": msg.reference, "txn_id": msg.txn_id,
    })


@app.post("/messages", status_code=202)
async def send_text(msg: OutgoingText, x_api_key: str | None = Header(default=None),
                    idempotency_key: str | None = Header(default=None, max_length=200)):
    """The bank sends a text to a customer."""
    if x_api_key != app.state.api_key:
        raise HTTPException(status_code=401, detail="bad API key")
    message_id = "sms-out-" + uuid.uuid4().hex[:12]
    sent = msg.model_copy(update={"message_id": message_id})
    key = f"sms:idem:{idempotency_key}" if idempotency_key else None
    try:
        async with app.state.redis.pipeline(transaction=True) as pipe:
            if key:
                await pipe.watch(key)
                earlier = await pipe.get(key)
                if earlier:   # already sent for this key: don't text the customer again
                    await pipe.unwatch()
                    log.info("Repeat of %s (same Idempotency-Key): not sent again", earlier)
                    return {"status": "duplicate", "message_id": earlier}
            # ONE transaction: the text, its scheduled reply and the key all happen, or none do.
            pipe.multi()
            pipe.xadd(SMS_LOG, log_entry("out", msg.to, msg.text, message_id, msg.txn_id),
                      maxlen=5000, approximate=True)
            if msg.expects_reply:
                pipe.zadd(PENDING_REPLIES, {sent.model_dump_json(): time.time()
                                            + random.uniform(*REPLY_DELAY_SECONDS)})
            if key:
                pipe.set(key, message_id, ex=IDEMPOTENCY_DAYS * 24 * 3600)
            await pipe.execute()
    except aioredis.WatchError:   # the same key is being sent right now by another request
        raise HTTPException(status_code=409, detail="the same text is being sent; retry shortly")
    log.info("Text %s to %s: %s", message_id, msg.to, msg.text[:80])
    return {"status": "queued", "message_id": message_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
