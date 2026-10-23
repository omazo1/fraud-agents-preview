"""The Decision agent (Part 1 · Step 6, extended in Part 2 · Step 6). Plain rules, no AI.

1. First decision: reads "I scored it" forms (risk.scored) and publishes "I decided"
   forms (transaction.decided): approve, flag (let it through, investigate afterwards)
   or hold (pause it until checked). The lines live in rules_v1.json.
2. Final outcome: reads the slow lane's results (verification.completed,
   investigation.completed, case.escalated) and publishes transaction.resolved:
   released, blocked or sent_to_analyst. The rules live in resolution_rules_v1.json.
   The AI agents only advise; this is where the outcome is decided.
3. Analyst override (Part 3 · Step 1): reads case.reviewed (Release / Block clicked on the
   Monitor's Analyst page) and publishes a new transaction.resolved that replaces a
   sent_to_analyst outcome. Only a case that is still waiting for a human can be changed.

Run from the project folder:  python -m fraud_agents.agents.decision.agent
Stop it with Ctrl + C.
"""

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

from fraud_agents.shared import bus, db
from fraud_agents.shared.events import (
    CaseEscalated,
    CaseReviewed,
    InvestigationCompleted,
    RiskScored,
    TransactionDecided,
    TransactionDecidedPayload,
    TransactionResolved,
    TransactionResolvedPayload,
    VerificationCompleted,
    derived_event_id,
    now_utc,
)

PRODUCER = "decision"
GROUP = "decision"
RULES_FILE = Path(__file__).with_name("rules_v1.json")
RESOLUTION_FILE = Path(__file__).with_name("resolution_rules_v1.json")

log = logging.getLogger(PRODUCER)


def load_rules(path: Path = RULES_FILE) -> dict:
    rules = json.loads(path.read_text())
    if not 0 < rules["flag_at"] < rules["hold_at"] <= 1:
        raise ValueError("rules: need 0 < flag_at < hold_at <= 1")
    return rules


def decide(scored: RiskScored, rules: dict) -> TransactionDecided:
    """Apply the rule book to one score. The same score always gives the same decision."""
    score = scored.payload.score
    if score >= rules["hold_at"]:
        decision, rule_id = "hold", "score_at_or_above_hold_line"
    elif score >= rules["flag_at"]:
        decision, rule_id = "flag", "score_at_or_above_flag_line"
    else:
        decision, rule_id = "approve", "default_approve"

    return TransactionDecided(
        event_id=derived_event_id(scored.event_id, PRODUCER),
        published_at=now_utc(),
        txn_id=scored.txn_id,
        occurred_at=scored.occurred_at,
        payload=TransactionDecidedPayload(
            decision=decision,
            rule_id=rule_id,
            rules_version=rules["rules_version"],
            score=score,
            based_on_event_id=scored.event_id,
        ),
    )


def first_decision_of(decided_event_id: str) -> str | None:
    """approve / flag / hold for a transaction.decided event, from the audit log."""
    with db.connect() as conn:
        row = conn.execute("SELECT decision FROM events WHERE event_id = %s", (decided_event_id,)).fetchone()
    return row[0] if row else None


def is_analyst_case(outcome_event_id: str, txn_id: str) -> bool:
    """True only if this event is a sent_to_analyst final outcome for this payment.

    Raises if the audit log hasn't recorded it yet, so the review is tried again later."""
    with db.connect() as conn:
        row = conn.execute("SELECT txn_id, decision FROM events WHERE event_id = %s "
                           "AND event_type = 'transaction.resolved'", (outcome_event_id,)).fetchone()
    if row is None:
        raise RuntimeError("the reviewed outcome is not in the audit log (yet)")
    return row[0] == txn_id and row[1] == "sent_to_analyst"


def claim_final_outcome(r, txn_id: str, outcome_event_id: str) -> bool:
    """One final outcome per payment: the first one wins (SET NX). True if this outcome may be
    published: it is the first, or a retry of the one that won (same ID: readers skip it)."""
    key = f"resolved:{txn_id}"
    return bool(r.set(key, outcome_event_id, nx=True)) or r.get(key) == outcome_event_id


def apply_review(r, review: CaseReviewed, new_outcome_id: str, latest_outcome=None,
                 analyst_case=None) -> str:
    """May this analyst review replace the payment's final outcome? Returns why or why not:
    "apply" (swap done), "retry" (already applied: publish the same ID again),
    "not_an_analyst_case" or "not_waiting" (ignored).

    An analyst can replace ONLY a sent_to_analyst outcome of the same payment that is still
    the current one. The swap is atomic (WATCH/MULTI); a WatchError means retry later."""
    latest_outcome = latest_outcome or latest_outcome_id
    analyst_case = analyst_case or is_analyst_case
    key = f"resolved:{review.txn_id}"
    case_id = review.payload.based_on_event_id
    current = r.get(key)
    if current == new_outcome_id:
        return "retry"
    if current != case_id and latest_outcome(review.txn_id) == case_id:
        # The Redis note was lost (e.g. Redis was reset), but the audit log shows the case
        # is still waiting for a human: restore the note instead of ignoring the review.
        r.set(key, case_id)
    if not analyst_case(case_id, review.txn_id):
        return "not_an_analyst_case"
    with r.pipeline() as pipe:
        pipe.watch(key)
        if pipe.get(key) != case_id:
            pipe.unwatch()
            return "not_waiting"
        pipe.multi()
        pipe.set(key, new_outcome_id)
        pipe.execute()                 # raises WatchError if it changed meanwhile: retried
    return "apply"


def latest_outcome_id(txn_id: str) -> str | None:
    """The newest final outcome of a payment, from the audit log."""
    with db.connect() as conn:
        row = conn.execute("SELECT event_id FROM events WHERE txn_id = %s AND event_type = "
                           "'transaction.resolved' ORDER BY published_at DESC LIMIT 1", (txn_id,)).fetchone()
    return row[0] if row else None


def resolve(event, rules: dict, first_decision=first_decision_of) -> TransactionResolved | None:
    """The final outcome for one slow-lane result, by plain rules. None = nothing to decide."""
    if isinstance(event, VerificationCompleted):
        resolution, rule_id = rules["verification"][event.payload.outcome]
    elif isinstance(event, CaseReviewed):
        resolution, rule_id = rules["analyst"][event.payload.decision]
    elif isinstance(event, CaseEscalated):
        resolution, rule_id = rules["escalation"]
    elif isinstance(event, InvestigationCompleted):
        action = event.payload.recommended_action
        if action == "verify_with_customer":
            return None  # the Verification agent takes it from here
        if action == "no_further_action":
            first = first_decision(event.payload.based_on_event_id)
            if first is None:   # the audit log hasn't recorded it yet: try again later
                raise RuntimeError("first decision not in the audit log yet")
            action = "no_further_action_after_flag" if first == "flag" else "no_further_action_after_hold"
        resolution, rule_id = rules["investigation"][action]
    else:
        return None
    return TransactionResolved(
        event_id=derived_event_id(event.event_id, PRODUCER, "transaction.resolved"),
        published_at=now_utc(),
        txn_id=event.txn_id,
        occurred_at=event.occurred_at,
        payload=TransactionResolvedPayload(
            resolution=resolution, rule_id=rule_id, rules_version=rules["rules_version"],
            based_on_event_id=event.event_id,
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rules = load_rules()
    r = bus.connect()
    r.ping()

    def handle(event):
        if not isinstance(event, RiskScored):
            raise ValueError(f"Expected risk.scored, got {event.event_type}")
        decided = decide(event, rules)
        bus.publish(r, decided)
        log.info(
            "txn %s  score %.4f  ->  %-7s (%s)",
            event.txn_id[:8], event.payload.score,
            decided.payload.decision.upper(), decided.payload.rule_id,
        )

    resolution_rules = json.loads(RESOLUTION_FILE.read_text())

    def give_up_result(event, reason: str) -> None:
        """A slow-lane result could not be decided after every try: send it to a human,
        so a flagged or held payment never ends without an outcome (Part 3 · Step 3)."""
        if not isinstance(event, (VerificationCompleted, InvestigationCompleted, CaseEscalated)):
            return
        resolution, rule_id = resolution_rules["gave_up"]
        resolved = TransactionResolved(   # its own ID: never the same as a normal outcome's
            event_id=derived_event_id(event.event_id, PRODUCER, "transaction.resolved/gave_up"),
            published_at=now_utc(), txn_id=event.txn_id, occurred_at=event.occurred_at,
            payload=TransactionResolvedPayload(resolution=resolution, rule_id=rule_id,
                                               rules_version=resolution_rules["rules_version"],
                                               based_on_event_id=event.event_id))
        if claim_final_outcome(r, event.txn_id, resolved.event_id):
            bus.publish(r, resolved)
            log.warning("txn %s  could not decide (%s): SENT_TO_ANALYST", event.txn_id[:8], reason[:120])

    def handle_result(event):
        resolved = resolve(event, resolution_rules)
        if resolved is None:
            return
        # One final outcome per payment: the first one wins (SET NX), later ones are ignored.
        # A retry of the winner finds its own ID there and publishes again (same ID: readers skip it).
        if not claim_final_outcome(r, event.txn_id, resolved.event_id):
            log.info("txn %s  already has a final outcome: %s ignored", event.txn_id[:8], event.event_type)
            return
        bus.publish(r, resolved)
        log.info("txn %s  FINAL: %-15s (%s, from %s)", event.txn_id[:8],
                 resolved.payload.resolution.upper(), resolved.payload.rule_id, event.event_type)

    def handle_review(event):
        if not isinstance(event, CaseReviewed):
            raise ValueError(f"Expected case.reviewed, got {event.event_type}")
        resolved = resolve(event, resolution_rules)
        verdict = apply_review(r, event, resolved.event_id)
        if verdict == "not_an_analyst_case":
            log.warning("txn %s  analyst review ignored: it doesn't point at a sent_to_analyst "
                        "outcome of this payment", event.txn_id[:8])
            return
        if verdict == "not_waiting":
            log.warning("txn %s  analyst review ignored: the case is not waiting for an analyst",
                        event.txn_id[:8])
            return
        bus.publish(r, resolved)               # a retry publishes the same ID again; readers skip it
        log.info("txn %s  FINAL (analyst %s): %-8s (%s)", event.txn_id[:8], event.payload.analyst,
                 resolved.payload.resolution.upper(), resolved.payload.rule_id)

    host = f"{socket.gethostname()}-{os.getpid()}"
    streams = [("risk.scored", handle), ("verification.completed", handle_result),
               ("investigation.completed", handle_result), ("case.escalated", handle_result),
               ("case.reviewed", handle_review)]
    threads = []
    for stream, handler in streams:
        t = threading.Thread(target=bus.consume, daemon=True, name=stream, args=(
            bus.connect(), bus.stream_for(stream), GROUP, handler),
            kwargs={"consumer": f"{host}-{stream}",
                    "on_give_up": None if stream in ("risk.scored", "case.reviewed") else give_up_result})
        t.start()
        threads.append(t)
    log.info(
        "Listening on %s (group %s). Rules %s: hold at %.2f, flag at %.2f; final outcomes %s. "
        "Press Ctrl + C to stop.", ", ".join(x for x, _ in streams), GROUP,
        rules["rules_version"], rules["hold_at"], rules["flag_at"], resolution_rules["rules_version"],
    )
    try:
        while all(t.is_alive() for t in threads):
            time.sleep(1)
        log.error("A listening thread stopped (%s); see the error above.",
                  ", ".join(t.name for t in threads if not t.is_alive()))
        raise SystemExit(1)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
