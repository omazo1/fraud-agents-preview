"""The payment gateway: the system's front door (Part 1 · Step 7).

Card payments arrive as signed webhooks (HTTP POST). The gateway:
1. checks the signature (wax seal) and refuses fakes;
2. checks the message is a complete, correct payment;
3. ignores a payment it has already accepted (processors resend webhooks);
4. turns it into a "payment arrived" form and drops it into Redis.

Run from the project folder:
    uvicorn fraud_agents.services.gateway.app:app --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fraud_agents.shared import config, signing
from fraud_agents.shared.cards import card_token
from fraud_agents.shared.bus import stream_for
from fraud_agents.shared.events import (
    CustomerReplied,
    CustomerRepliedPayload,
    TransactionReceived,
    TransactionReceivedPayload,
    derived_event_id,
    now_utc,
)

log = logging.getLogger("gateway")
SEEN_TTL_SECONDS = 7 * 24 * 3600


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
    # Assumption (Step 3): the dataset's times have no time zone; we treat them as UTC.
    occurred = datetime.fromisoformat(p.trans_date_trans_time).replace(tzinfo=timezone.utc)
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
    log.info("Gateway ready: POST /webhooks/payments")
    yield
    await app.state.redis.aclose()


app = FastAPI(title="Fraud system payment gateway", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "redis": await request.app.state.redis.ping()}


@app.post("/webhooks/payments", status_code=202)
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

    # 3. Seen it before?
    r = request.app.state.redis
    seen_key = f"gateway:seen:{payment.trans_num}"
    if await r.exists(seen_key):
        return JSONResponse({"status": "duplicate", "txn_id": payment.trans_num}, status_code=200)

    # 4. Into Redis as a "payment arrived" form, THEN remember it.
    # In this order, a failure can never lose a payment: at worst it is published
    # twice with the same event_id, and the agents skip the second copy.
    await r.xadd(stream_for(event.event_type), {"event": event.model_dump_json()})
    await r.set(seen_key, 1, ex=SEEN_TTL_SECONDS)
    log.info("Accepted txn %s  $%s  %s", payment.trans_num[:8], payment.amt, payment.category)
    return {"status": "accepted", "txn_id": payment.trans_num, "event_id": event.event_id}


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
