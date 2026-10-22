"""The event forms for Part 1. Every event = envelope (same on all) + payload."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Literal, Union
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


def now_utc() -> datetime:
    """The current time, in UTC."""
    return datetime.now(timezone.utc)


def new_event_id() -> str:
    """A fresh random ID for a new event."""
    return str(uuid4())


def derived_event_id(source_event_id: str, producer: str) -> str:
    """The ID for an event an agent makes in reply to another event.

    It is the same every time for the same input event and agent, so if an agent
    retries after a crash, the form it publishes again has the same ID and the
    next agent can spot the duplicate.
    """
    return str(uuid5(NAMESPACE_URL, f"fraud-agents/{producer}/{source_event_id}"))


class Form(BaseModel):
    """Base for every form: no unknown boxes, and no changes once filled in."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Envelope(Form):
    """The label on every event (the courier sticker).

    event_id and published_at are required, so a form read back from Redis
    without them is rejected. The sender fills them with new_event_id() and now_utc().
    """

    event_id: str = Field(min_length=1)
    event_type: str
    schema_version: int = 1
    txn_id: str = Field(min_length=1)
    occurred_at: AwareDatetime
    published_at: AwareDatetime
    producer: str


# ---------- Form 1: "a payment arrived" ----------

class TransactionReceivedPayload(Form):
    card_id: str
    amount: Decimal = Field(gt=0)
    category: str
    merchant: str
    merchant_lat: float = Field(ge=-90, le=90)
    merchant_long: float = Field(ge=-180, le=180)
    home_lat: float = Field(ge=-90, le=90)
    home_long: float = Field(ge=-180, le=180)
    city_pop: int = Field(ge=0)


class TransactionReceived(Envelope):
    event_type: Literal["transaction.received"] = "transaction.received"
    producer: Literal["gateway"] = "gateway"
    payload: TransactionReceivedPayload


# ---------- Form 2: "I scored it" ----------

class RiskScoredPayload(Form):
    score: float = Field(ge=0, le=1)
    model_version: str
    top_factors: list[str] = Field(default_factory=list)


class RiskScored(Envelope):
    event_type: Literal["risk.scored"] = "risk.scored"
    producer: Literal["risk-scoring"] = "risk-scoring"
    payload: RiskScoredPayload


# ---------- Form 3: "I decided" ----------

class TransactionDecidedPayload(Form):
    decision: Literal["approve", "flag", "hold"]
    rule_id: str
    rules_version: str
    score: float = Field(ge=0, le=1)
    based_on_event_id: str = Field(min_length=1)


class TransactionDecided(Envelope):
    event_type: Literal["transaction.decided"] = "transaction.decided"
    producer: Literal["decision"] = "decision"
    payload: TransactionDecidedPayload


# ---------- Any of the three forms (for readers of every type, e.g. the audit log) ----------

AnyEvent = Annotated[
    Union[TransactionReceived, RiskScored, TransactionDecided],
    Field(discriminator="event_type"),
]
_any_event = TypeAdapter(AnyEvent)


def parse_event(text: str) -> TransactionReceived | RiskScored | TransactionDecided:
    """Read a form from text (as stored in Redis); event_type picks the right form."""
    return _any_event.validate_json(text)


# ---------- Self-check: python -m fraud_agents.shared.events ----------

if __name__ == "__main__":
    when = datetime(2019, 1, 1, 0, 0, 18, tzinfo=timezone.utc)

    received = TransactionReceived(
        event_id=new_event_id(),
        published_at=now_utc(),
        txn_id="0b242abb623afc578575680df30655b9",
        occurred_at=when,
        payload=TransactionReceivedPayload(
            card_id="card-demo-1",
            amount=Decimal("4.97"),
            category="misc_net",
            merchant="fraud_Rippin, Kub and Mann",
            merchant_lat=36.011293,
            merchant_long=-82.048315,
            home_lat=36.0788,
            home_long=-81.1781,
            city_pop=3495,
        ),
    )
    # Redis stores text: turn the form into text and back, and check nothing changed.
    as_text = received.model_dump_json()
    assert parse_event(as_text) == received
    print("1. Payment-arrived form: OK")

    scored = RiskScored(
        event_id=new_event_id(),
        published_at=now_utc(),
        txn_id=received.txn_id,
        occurred_at=when,
        payload=RiskScoredPayload(score=0.03, model_version="demo"),
    )
    print("2. Scored form: OK")

    decided = TransactionDecided(
        event_id=new_event_id(),
        published_at=now_utc(),
        txn_id=received.txn_id,
        occurred_at=when,
        payload=TransactionDecidedPayload(
            decision="approve",
            rule_id="default_approve",
            rules_version="demo",
            score=scored.payload.score,
            based_on_event_id=scored.event_id,
        ),
    )
    assert parse_event(decided.model_dump_json()) == decided
    print("3. Decided form: OK")

    try:
        TransactionDecidedPayload(
            decision="maybe", rule_id="x", rules_version="x", score=1.5, based_on_event_id="x"
        )
        print("4. Bad form was NOT caught (problem!)")
    except ValidationError as error:
        print(f"4. Bad form caught: {error.error_count()} mistakes found")

    print("All checks passed.")
