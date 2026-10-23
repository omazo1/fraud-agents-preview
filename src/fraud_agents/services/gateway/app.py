"""The payment gateway: the system's front door (Part 1 · Step 7).

Card payments arrive as signed webhooks (HTTP POST). The gateway:
1. checks the signature (wax seal) and refuses fakes;
2. checks the message is a complete, correct payment;
3. answers a repeat of a payment with the answer it already gave (processors resend webhooks);
4. turns it into a "payment arrived" form and publishes it (Pub/Sub);
5. ANSWERS the payment source (feedback #2): it waits a short time (DECISION_DEADLINE_MS) for
   the Decision agent's verdict and replies "approved" or "declined" (soft). A card network
   needs its answer in a few hundred milliseconds, so it can't wait for the slow lane.

A SOFT decline means "declined for now": the customer is texted "was this you?". If they
say YES (or an analyst releases it), the same card, shop and amount is approved ONCE if tried
again within RETRY_WINDOW_SECONDS. No verdict in time also gives a soft decline: the customer
simply tries again.

The gateway is also a SUBSCRIBER: it listens live to transaction.decided (to answer) and,
as a durable consumer group, to transaction.resolved (to keep the payment's status). The
payment source can ask for the status at any time: GET /payments/{txn_id} (signed).

If the gateway crashes after claiming a payment but before answering, a resend of it is
answered with a soft decline (nothing has moved; the customer tries again).

Run from the project folder:
    uvicorn fraud_agents.services.gateway.app:app --port 8000
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal

import redis
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fraud_agents.shared import bus, config, signing
from fraud_agents.shared.cards import card_token
from fraud_agents.shared.bus import stream_for
from fraud_agents.shared.events import (
    CustomerReplied,
    CustomerRepliedPayload,
    TransactionDecided,
    TransactionReceived,
    TransactionReceivedPayload,
    TransactionResolved,
    derived_event_id,
    now_utc,
    parse_event,
    to_utc,
)

log = logging.getLogger("gateway")
SEEN_TTL_SECONDS = 7 * 24 * 3600
GROUP = "gateway"
DECISION_DEADLINE_MS = int(os.environ.get("DECISION_DEADLINE_MS", "300"))
RETRY_WINDOW_SECONDS = int(os.environ.get("RETRY_WINDOW_SECONDS", str(30 * 60)))

# What the payment source can read back later: (answer, slow-lane outcome) -> plain words.
STATUS_TEXT = {
    ("approved", ""): "approved",
    ("declined_soft", ""): "declined for now: waiting for the customer to confirm",
    ("declined_soft", "retry_allowed"): f"declined for now: confirmed, so a retry within "
                                        f"{RETRY_WINDOW_SECONDS // 60} minutes will be approved",
    ("declined_soft", "blocked"): "declined: blocked",
    ("declined_soft", "with_analyst"): "declined for now: with a fraud analyst",
}
NO_VERDICT_TEXT = "declined: no verdict in time; please try again"


def payment_key(txn_id: str) -> str:
    return f"payment:{txn_id}"


def allow_key(card_id: str, merchant: str, category: str, merch_lat, merch_long, amount) -> str:
    """The one retry a customer's YES allows: the same card, shop (name, type and location)
    and amount. Anything different is a new payment and is scored as usual."""
    return (f"allow:{card_id}:{merchant}:{category}:{float(merch_lat):.3f},{float(merch_long):.3f}:"
            f"{Decimal(str(amount)):.2f}")


def status_text(answer: str, outcome: str, reason: str) -> str:
    if reason == "no_decision_in_time":
        return NO_VERDICT_TEXT
    return STATUS_TEXT.get((answer, outcome), answer)


def answer_body(txn_id: str, answer: str, reason: str, decided_in_ms: int | None = None) -> dict:
    body = {"txn_id": txn_id, "status": "approved" if answer == "approved" else "declined",
            "reason": reason}
    if answer != "approved":
        body["decline_type"] = "soft"     # declined for now: the customer can confirm and retry
    if decided_in_ms is not None:
        body["decided_in_ms"] = decided_in_ms
    return body


async def settle(r, txn_id: str, answer: str, reason: str) -> tuple[str, str]:
    """Store the payment's answer ONCE (HSETNX) and return the stored one. If something else
    already answered this payment (e.g. a resend after a crash), that answer wins, so a
    payment never gets two different answers."""
    await r.hsetnx(payment_key(txn_id), "answer", f"{answer}:{reason}")
    await r.expire(payment_key(txn_id), SEEN_TTL_SECONDS)
    stored = await r.hget(payment_key(txn_id), "answer")
    answer, _, reason = stored.partition(":")
    return answer, reason


async def repeat_answer(r, txn_id: str, claimed_at: float) -> dict:
    """A resend of a payment we already have: give the SAME answer as the first request.
    If the first request is still waiting, wait for it; if it died (e.g. the gateway
    restarted mid-wait), answer a soft decline: nothing has moved, the customer tries again."""
    give_up_at = claimed_at + DECISION_DEADLINE_MS / 1000 + 1
    while time.time() < give_up_at:
        stored = await r.hget(payment_key(txn_id), "answer")
        if stored:
            answer, _, reason = stored.partition(":")
            return answer_body(txn_id, answer, reason)
        await asyncio.sleep(0.02)
    answer, reason = await settle(r, txn_id, "declined_soft", "no_decision_in_time")
    return answer_body(txn_id, answer, reason)


# ---------- The gateway as a SUBSCRIBER (two background threads) ----------

def on_decided(app, r, event: TransactionDecided) -> None:
    """A verdict arrived: wake the request that is waiting for it (if any)."""
    waiting = app.state.waiting.get(event.txn_id)
    if waiting is not None:
        future, loop = waiting
        loop.call_soon_threadsafe(lambda: future.done() or future.set_result(event.payload.decision))
    if r.exists(payment_key(event.txn_id)):      # only payments this gateway is tracking
        r.hset(payment_key(event.txn_id), "decision", event.payload.decision)


def listen_for_verdicts(app) -> None:
    """LIVE listening to transaction.decided: a plain XREAD from the newest entry, no consumer
    group. Every gateway process reads every NEW verdict and wakes its own waiting requests;
    old verdicts are never replayed. Missing one is safe: that request times out into a soft
    decline. (The durable consumer groups are for work that must not be lost.)"""
    r = bus.connect()
    stream = stream_for("transaction.decided")
    newest = r.xrevrange(stream, count=1)
    last = newest[0][0] if newest else "0-0"
    while not app.state.stop.is_set():
        try:
            reply = r.xread({stream: last}, block=1000, count=100)
        except redis.RedisError as error:          # never let the listener die quietly
            log.warning("Verdict listener: Redis error (%s); retrying in 1 s", error)
            time.sleep(1)
            continue
        for _, entries in reply or []:
            for entry_id, fields in entries:
                last = entry_id
                try:
                    on_decided(app, r, parse_event(fields["event"]))
                except Exception:
                    log.exception("Could not read verdict %s", entry_id)


def on_resolved(r, event: TransactionResolved) -> None:
    """The slow lane finished: record it, and allow one retry if the payment was released."""
    key = payment_key(event.txn_id)
    if not r.exists(key):        # not a payment this gateway answered: nothing to update
        return
    resolution = event.payload.resolution
    outcome = {"released": "retry_allowed", "blocked": "blocked", "sent_to_analyst": "with_analyst"}[resolution]
    r.hset(key, "outcome", outcome)
    if resolution == "released":
        fields = r.hmget(key, "card_id", "merchant", "category", "merch_lat", "merch_long", "amount")
        if all(fields):
            r.set(allow_key(*fields), event.txn_id, ex=RETRY_WINDOW_SECONDS)
            log.info("txn %s  released: one retry allowed for %d min", event.txn_id[:8],
                     RETRY_WINDOW_SECONDS // 60)


def start_listeners(app) -> list[threading.Thread]:
    verdicts = threading.Thread(target=listen_for_verdicts, args=(app,), daemon=True,
                                name="gateway-verdicts")
    # Outcomes must not be missed (they allow retries), so this one is a durable consumer group.
    r = bus.connect()
    outcomes = threading.Thread(target=bus.consume, daemon=True, name="gateway-outcomes",
                                args=(r, stream_for("transaction.resolved"), GROUP,
                                      lambda event: on_resolved(r, event)),
                                kwargs={"block_ms": 1000, "stop_event": app.state.stop})
    for thread in (verdicts, outcomes):
        thread.start()
    app.state.listeners = [verdicts, outcomes]
    return [verdicts, outcomes]


class IncomingPayment(BaseModel):
    """What the card processor sends us (the dataset's own column names).

    extra="forbid": anything else, e.g. the answer key is_fraud, is refused.
    """

    model_config = ConfigDict(extra="forbid")

    trans_num: str = Field(min_length=1)
    trans_date_trans_time: str
    cc_num: str = Field(min_length=1)
    merchant: str
    category: str
    amt: Decimal = Field(gt=0)
    lat: float
    long: float
    city_pop: int = Field(ge=0)
    merch_lat: float
    merch_long: float


def to_event(p: IncomingPayment, token_key: str) -> TransactionReceived:
    # A time with a zone is converted to UTC; the dataset's times have none, so they are
    # treated as UTC (assumption from Step 3). Part 3 · Step 3: no longer overwrites a zone.
    occurred = to_utc(datetime.fromisoformat(p.trans_date_trans_time))
    return TransactionReceived(
        event_id=derived_event_id(p.trans_num, "gateway"),
        published_at=now_utc(),
        txn_id=p.trans_num,
        occurred_at=occurred,
        payload=TransactionReceivedPayload(
            card_id=card_token(p.cc_num, token_key),
            amount=p.amt,
            category=p.category,
            merchant=p.merchant,
            merchant_lat=p.merch_lat,
            merchant_long=p.merch_long,
            home_lat=p.lat,
            home_long=p.long,
            city_pop=p.city_pop,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app.state.secret = config.require("WEBHOOK_SECRET")
    app.state.token_key = config.require("CARD_TOKEN_KEY")
    app.state.sms_secret = config.require("SMS_WEBHOOK_SECRET")
    app.state.redis = aioredis.Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
    )
    await app.state.redis.ping()
    app.state.waiting = {}                  # txn_id -> (future, loop): requests waiting for a verdict
    app.state.stop = threading.Event()
    threads = start_listeners(app)
    log.info("Gateway ready: POST /webhooks/payments (answers within %d ms)", DECISION_DEADLINE_MS)
    yield
    app.state.stop.set()
    for thread in threads:
        thread.join(timeout=3)
    await app.state.redis.aclose()


app = FastAPI(title="Fraud system payment gateway", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    listening = all(t.is_alive() for t in getattr(request.app.state, "listeners", []))
    return {"status": "ok" if listening else "degraded", "redis": await request.app.state.redis.ping(),
            "listening_for_verdicts": listening}


@app.post("/webhooks/payments")
async def receive_payment(request: Request):
    body = await request.body()

    # 1. The wax seal.
    if not signing.verify(
        request.app.state.secret,
        request.headers.get("X-Timestamp"),
        request.headers.get("X-Signature"),
        body,
    ):
        log.warning("Refused: bad or missing signature")
        raise HTTPException(status_code=401, detail="bad or missing signature")

    # 2. A complete, correct payment?
    try:
        payment = IncomingPayment.model_validate_json(body)
        event = to_event(payment, request.app.state.token_key)
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)[:500])

    # 3. Claim the payment: only ONE request per payment goes on (SET NX is atomic). A resend
    # gets the first request's answer, never a second, different one.
    r = request.app.state.redis
    txn = payment.trans_num
    seen_key = f"gateway:seen:{txn}"
    if not await r.set(seen_key, time.time(), nx=True, ex=SEEN_TTL_SECONDS):
        return await repeat_answer(r, txn, float(await r.get(seen_key) or 0))

    # 4. Is this the retry a customer's YES allowed? (used once: GETDEL)
    p = event.payload
    allow = allow_key(p.card_id, p.merchant, p.category, p.merchant_lat, p.merchant_long, p.amount)
    window_left = await r.ttl(allow)
    confirmed = await r.getdel(allow)
    if confirmed:
        event = event.model_copy(update={"payload": p.model_copy(update={"confirmed_retry_of": confirmed})})

    # 5. Publish it and wait for the verdict. The wait is registered BEFORE publishing, so a
    # very fast verdict can't arrive before anyone is listening for it.
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    request.app.state.waiting[txn] = (future, loop)
    started = time.perf_counter()
    try:
        try:
            await r.hset(payment_key(txn), mapping={
                "card_id": p.card_id, "merchant": p.merchant, "category": p.category,
                "merch_lat": str(p.merchant_lat), "merch_long": str(p.merchant_long), "amount": str(p.amount)})
            await r.expire(payment_key(txn), SEEN_TTL_SECONDS)
            await r.xadd(stream_for(event.event_type), {"event": event.model_dump_json()})
        except redis.RedisError:
            # Couldn't publish it: undo the claim (and give back the YES), so the processor's
            # resend is handled as a NEW attempt instead of being lost. It gets a 500 and retries.
            await r.delete(seen_key, payment_key(txn))
            if confirmed:
                await r.set(allow, confirmed, ex=max(window_left, 1))
            raise
        try:
            decision = await asyncio.wait_for(future, DECISION_DEADLINE_MS / 1000)
        except asyncio.TimeoutError:
            decision = None
    finally:
        request.app.state.waiting.pop(txn, None)
    took = round((time.perf_counter() - started) * 1000)

    # 6. The answer at the till. Only "approve" lets the money move now.
    if decision == "approve":
        answer, reason = "approved", ("customer_confirmed_retry" if confirmed else "low_risk")
    elif decision in ("flag", "hold"):
        answer, reason = "declined_soft", "verify_with_customer"
    else:
        answer, reason = "declined_soft", "no_decision_in_time"   # the customer just tries again
    answer, reason = await settle(r, txn, answer, reason)
    if confirmed and answer != "approved":
        # Don't use up the customer's YES on a slow or failed retry: give it back.
        await r.set(allow, confirmed, ex=max(window_left, 1))
    log.info("txn %s  $%s  %s -> %s (%s, %d ms)", txn[:8], payment.amt, payment.category,
             answer.upper(), reason, took)
    return answer_body(txn, answer, reason, took) | {"event_id": event.event_id}


def status_message(txn_id: str) -> bytes:
    """What a status request signs: the method and the path, so it can't be mixed up with a
    signed payment webhook."""
    return f"GET /payments/{txn_id}".encode()


@app.get("/payments/{txn_id}")
async def payment_status(txn_id: str, request: Request):
    """The payment source asks: what is this payment's status now? (signed)"""
    if not signing.verify(request.app.state.secret, request.headers.get("X-Timestamp"),
                          request.headers.get("X-Signature"), status_message(txn_id)):
        raise HTTPException(status_code=401, detail="bad or missing signature")
    stored, outcome = await request.app.state.redis.hmget(payment_key(txn_id), "answer", "outcome")
    if not stored:
        raise HTTPException(status_code=404, detail="unknown payment (or no answer yet)")
    answer, _, reason = stored.partition(":")
    return {"txn_id": txn_id, "answer": answer, "reason": reason, "outcome": outcome,
            "status": status_text(answer, outcome or "", reason)}


# ---------- Customers' text replies (Part 2 · Step 5) ----------

class IncomingSmsReply(BaseModel):
    """What the SMS provider sends when a customer replies to one of our texts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message_id: str = Field(min_length=1)
    from_: str = Field(min_length=1, alias="from")
    text: str = Field(min_length=1, max_length=500)
    in_reply_to: str = Field(min_length=1)      # the provider's ID of our outgoing text
    reference: str = Field(min_length=1)         # our verification.requested event_id
    txn_id: str = Field(min_length=1)


@app.post("/webhooks/sms-replies", status_code=202)
async def receive_sms_reply(request: Request):
    body = await request.body()
    if not signing.verify(request.app.state.sms_secret, request.headers.get("X-Timestamp"),
                          request.headers.get("X-Signature"), body):
        log.warning("Refused SMS reply: bad or missing signature")
        raise HTTPException(status_code=401, detail="bad or missing signature")
    try:
        reply = IncomingSmsReply.model_validate_json(body)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)[:500])

    r = request.app.state.redis
    seen_key = f"gateway:sms-seen:{reply.message_id}"
    if await r.exists(seen_key):
        return JSONResponse({"status": "duplicate"}, status_code=200)

    event = CustomerReplied(
        # Built from the provider's message ID, so a resent webhook is spotted as a duplicate.
        event_id=derived_event_id(reply.message_id, "sms-gateway"),
        published_at=now_utc(),
        txn_id=reply.txn_id,
        occurred_at=now_utc(),
        payload=CustomerRepliedPayload(
            card_id=reply.from_, reply_text=reply.text,
            message_id=reply.message_id, in_reply_to=reply.reference,
        ),
    )
    await r.xadd(stream_for(event.event_type), {"event": event.model_dump_json()})
    await r.set(seen_key, 1, ex=SEEN_TTL_SECONDS)
    log.info("Customer reply for txn %s: %r", reply.txn_id[:8], reply.text)
    return {"status": "accepted", "event_id": event.event_id}
