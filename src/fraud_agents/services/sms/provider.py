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
    log.info("SMS provider ready: POST /messages (customers know %d payments)", len(app.state.is_fraud))
    yield
    await app.state.http.aclose()
    await app.state.redis.aclose()


app = FastAPI(title="Mock SMS provider", lifespan=lifespan)


async def record(r, direction: str, who: str, text: str, message_id: str, txn_id: str | None):
    await r.xadd(SMS_LOG, {"direction": direction, "who": who, "text": text,
                           "message_id": message_id, "txn_id": txn_id or "",
                           "at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
                 maxlen=5000, approximate=True)


async def send_reply_webhook(state, reply: dict) -> None:
    """Deliver a customer's reply to the bank, signed, retrying a few times."""
    body = json.dumps(reply).encode()
    for attempt in range(1, 6):
        ts = str(int(time.time()))
        headers = {"Content-Type": "application/json", "X-Timestamp": ts,
                   "X-Signature": signing.sign(state.webhook_secret, ts, body)}
        try:
            response = await state.http.post(GATEWAY_REPLY_URL, content=body, headers=headers)
            if 200 <= response.status_code < 300:
                log.info("Reply %s delivered to the bank: HTTP %s", reply["message_id"], response.status_code)
                return
            if response.status_code < 500:   # the bank refused it: retrying won't help
                log.error("Bank refused reply %s: HTTP %s %s", reply["message_id"],
                          response.status_code, response.text[:200])
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(2 ** attempt)
    log.warning("Could not deliver reply %s to the bank", reply["message_id"])


async def customer_answers(state, msg: OutgoingText, message_id: str) -> None:
    """The simulated customer reads the text and (maybe) answers. Errors are logged, not lost."""
    try:
        await _customer_answers(state, msg, message_id)
    except Exception:
        log.exception("Simulated customer failed on %s", message_id)


async def _customer_answers(state, msg: OutgoingText, message_id: str) -> None:
    await asyncio.sleep(random.uniform(*REPLY_DELAY_SECONDS))
    truth = state.is_fraud.get(msg.txn_id)
    roll = random.randint(1, 100)
    if roll <= NO_REPLY_PERCENT:
        log.info("Customer %s did not reply to %s", msg.to, message_id)
        return
    if truth is None or roll <= NO_REPLY_PERCENT + UNCLEAR_PERCENT:
        answer = random.choice(UNCLEAR)        # unknown payment: a real customer would be confused
    elif truth == 1:
        answer = random.choice(SAYS_NO)        # it really was fraud: "not me"
    else:
        answer = random.choice(SAYS_YES)       # they really made it: "yes, me"
    reply_id = "sms-in-" + uuid.uuid4().hex[:12]
    await record(state.redis, "in", msg.to, answer, reply_id, msg.txn_id)
    await send_reply_webhook(state, {
        "message_id": reply_id, "from": msg.to, "text": answer,
        "in_reply_to": message_id, "reference": msg.reference, "txn_id": msg.txn_id,
    })


@app.post("/messages", status_code=202)
async def send_text(msg: OutgoingText, x_api_key: str | None = Header(default=None)):
    """The bank sends a text to a customer."""
    if x_api_key != app.state.api_key:
        raise HTTPException(status_code=401, detail="bad API key")
    message_id = "sms-out-" + uuid.uuid4().hex[:12]
    await record(app.state.redis, "out", msg.to, msg.text, message_id, msg.txn_id)
    log.info("Text %s to %s: %s", message_id, msg.to, msg.text[:80])
    if msg.expects_reply:
        task = asyncio.create_task(customer_answers(app.state, msg, message_id))
        app.state.tasks.add(task)               # keep a reference until it finishes
        task.add_done_callback(app.state.tasks.discard)
    return {"status": "queued", "message_id": message_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
