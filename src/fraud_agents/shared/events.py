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


def derived_event_id(source_event_id: str, producer: str, event_type: str | None = None) -> str:
    """The ID for an event an agent makes in reply to another event.

    It is the same every time for the same input event, agent and form type, so if an
    agent retries after a crash, the form it publishes again has the same ID and the
    next agent can spot the duplicate. Pass event_type when one agent can publish
    more than one kind of form from the same input (e.g. a report OR an escalation).
    """
    key = f"fraud-agents/{producer}/{source_event_id}"
    if event_type:
        key += f"/{event_type}"
    return str(uuid5(NAMESPACE_URL, key))


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


# =====================================================================
# Part 2 forms: the slow lane (investigation, customer check, outcome)
# =====================================================================

# ---------- Form 4: "I investigated it" (the Investigation agent, an LLM) ----------

class InvestigationCompletedPayload(Form):
    verdict: Literal["likely_fraud", "likely_genuine", "unsure"]
    summary: str = Field(min_length=10, max_length=1000)
    evidence: list[str] = Field(default_factory=list, max_length=6)
    tools_used: list[str] = Field(default_factory=list)
    recommended_action: Literal["verify_with_customer", "escalate_to_analyst", "no_further_action"]
    model: str
    based_on_event_id: str = Field(min_length=1)


class InvestigationCompleted(Envelope):
    event_type: Literal["investigation.completed"] = "investigation.completed"
    producer: Literal["investigation"] = "investigation"
    payload: InvestigationCompletedPayload


# ---------- Form 5: "I asked the customer" (the Verification agent) ----------

class VerificationRequestedPayload(Form):
    """The event_id of this form is the verification's reference: the customer's
    reply comes back with it (in_reply_to), so it is matched to exactly this case."""

    card_id: str = Field(min_length=1)
    channel: Literal["sms"] = "sms"
    message: str = Field(min_length=1, max_length=480)
    based_on_event_id: str = Field(min_length=1)


class VerificationRequested(Envelope):
    event_type: Literal["verification.requested"] = "verification.requested"
    producer: Literal["verification"] = "verification"
    payload: VerificationRequestedPayload


# ---------- Form 6: "the customer replied" (arrives as a webhook from the SMS provider) ----------

class CustomerRepliedPayload(Form):
    card_id: str = Field(min_length=1)
    reply_text: str = Field(min_length=1, max_length=500)
    message_id: str = Field(min_length=1)           # the SMS provider's ID for this reply
    in_reply_to: str = Field(min_length=1)          # the verification.requested event_id


class CustomerReplied(Envelope):
    event_type: Literal["customer.replied"] = "customer.replied"
    producer: Literal["sms-gateway"] = "sms-gateway"
    payload: CustomerRepliedPayload


# ---------- Form 7: "the customer check is finished" (the Verification agent) ----------

class VerificationCompletedPayload(Form):
    outcome: Literal["confirmed_by_customer", "denied_by_customer", "unclear", "no_reply"]
    reply_text: str | None = None
    based_on_event_id: str = Field(min_length=1)    # the verification.requested event_id


class VerificationCompleted(Envelope):
    event_type: Literal["verification.completed"] = "verification.completed"
    producer: Literal["verification"] = "verification"
    payload: VerificationCompletedPayload


# ---------- Form 8: "final outcome" (the Decision agent's rules again: rules decide) ----------

class TransactionResolvedPayload(Form):
    resolution: Literal["released", "blocked", "sent_to_analyst"]
    rule_id: str
    rules_version: str
    based_on_event_id: str = Field(min_length=1)


class TransactionResolved(Envelope):
    event_type: Literal["transaction.resolved"] = "transaction.resolved"
    producer: Literal["decision"] = "decision"
    payload: TransactionResolvedPayload


# ---------- Form 9: "I told the customer" (the Notification agent) ----------

class NotificationSentPayload(Form):
    card_id: str = Field(min_length=1)
    channel: Literal["sms"] = "sms"
    message: str = Field(min_length=1, max_length=480)
    used_template: bool          # True = the AI draft failed the checks, so the safe template was sent
    based_on_event_id: str = Field(min_length=1)


class NotificationSent(Envelope):
    event_type: Literal["notification.sent"] = "notification.sent"
    producer: Literal["notification"] = "notification"
    payload: NotificationSentPayload


# ---------- Form 10: "a human must look at this" (any agent) ----------

class CaseEscalatedPayload(Form):
    reason: str = Field(min_length=5, max_length=500)
    based_on_event_id: str = Field(min_length=1)


class CaseEscalated(Envelope):
    event_type: Literal["case.escalated"] = "case.escalated"
    producer: Literal["investigation", "verification", "notification", "decision"]
    payload: CaseEscalatedPayload


# ---------- Form 11: "a human reviewed the case" (the analyst, from the Monitor's Analyst page) ----------

class CaseReviewedPayload(Form):
    decision: Literal["release", "block"]
    analyst: str = Field(min_length=1, max_length=60)
    note: str | None = Field(default=None, max_length=300)
    based_on_event_id: str = Field(min_length=1)    # the transaction.resolved (sent_to_analyst) event_id


class CaseReviewed(Envelope):
    event_type: Literal["case.reviewed"] = "case.reviewed"
    producer: Literal["analyst"] = "analyst"
    payload: CaseReviewedPayload


# ---------- Any of the forms (for readers of every type, e.g. the audit log) ----------

ALL_EVENTS = (
    TransactionReceived, RiskScored, TransactionDecided,
    InvestigationCompleted, VerificationRequested, CustomerReplied,
    VerificationCompleted, TransactionResolved, NotificationSent, CaseEscalated,
    CaseReviewed,
)
EVENT_TYPES = [form.model_fields["event_type"].default for form in ALL_EVENTS]

AnyEvent = Annotated[Union[ALL_EVENTS], Field(discriminator="event_type")]
_any_event = TypeAdapter(AnyEvent)


def parse_event(text: str):
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

    # The Part 2 forms: build one of each and read it back from text.
    def label(**kw):
        return dict(event_id=new_event_id(), published_at=now_utc(), txn_id=received.txn_id,
                    occurred_at=when, **kw)

    part2 = [
        InvestigationCompleted(**label(payload=InvestigationCompletedPayload(
            verdict="likely_fraud", summary="Large online purchase at night, far above usual.",
            evidence=["amount 30x the card's average"], tools_used=["card_history"],
            recommended_action="verify_with_customer", model="demo",
            based_on_event_id=decided.event_id))),
        VerificationRequested(**label(payload=VerificationRequestedPayload(
            card_id="card-demo-1", message="Did you just pay $4.97 at Rippin? Reply YES or NO.",
            based_on_event_id=decided.event_id))),
        CustomerReplied(**label(payload=CustomerRepliedPayload(
            card_id="card-demo-1", reply_text="NO that wasn't me", message_id="sms-1",
            in_reply_to="demo-verification-id"))),
        VerificationCompleted(**label(payload=VerificationCompletedPayload(
            outcome="denied_by_customer", reply_text="NO that wasn't me",
            based_on_event_id=decided.event_id))),
        TransactionResolved(**label(payload=TransactionResolvedPayload(
            resolution="blocked", rule_id="customer_denied", rules_version="demo",
            based_on_event_id=decided.event_id))),
        NotificationSent(**label(payload=NotificationSentPayload(
            card_id="card-demo-1", message="We blocked a payment on your card.",
            used_template=False, based_on_event_id=decided.event_id))),
        CaseEscalated(**label(producer="investigation", payload=CaseEscalatedPayload(
            reason="Report failed the checks twice", based_on_event_id=decided.event_id))),
        CaseReviewed(**label(payload=CaseReviewedPayload(
            decision="release", analyst="demo analyst", note="Known shop for this customer",
            based_on_event_id=decided.event_id))),
    ]
    for form in part2:
        assert parse_event(form.model_dump_json()) == form
    # One agent, one input, two kinds of form -> two different IDs.
    assert derived_event_id("x", "investigation", "investigation.completed") != \
        derived_event_id("x", "investigation", "case.escalated")
    # Part 1 IDs are unchanged when event_type is not given.
    assert derived_event_id("x", "decision") == str(uuid5(NAMESPACE_URL, "fraud-agents/decision/x"))
    print(f"3b. Part 2 and 3 forms: {len(part2)} OK ({', '.join(f.event_type for f in part2)})")

    try:
        TransactionDecidedPayload(
            decision="maybe", rule_id="x", rules_version="x", score=1.5, based_on_event_id="x"
        )
        print("4. Bad form was NOT caught (problem!)")
    except ValidationError as error:
        print(f"4. Bad form caught: {error.error_count()} mistakes found")

    print("All checks passed.")
