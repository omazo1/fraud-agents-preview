"""The Decision agent (Part 1 · Step 6, extended in Part 2 · Step 6). Plain rules, no AI.

1. First decision: reads "I scored it" forms (risk.scored) and publishes "I decided"
   forms (transaction.decided): approve, flag (let it through, investigate afterwards)
   or hold (pause it until checked). The lines live in rules_v1.json.
2. Final outcome: reads the slow lane's results (verification.completed,
   investigation.completed, case.escalated) and publishes transaction.resolved:
   released, blocked or sent_to_analyst. The rules live in resolution_rules_v1.json.
   The AI agents only advise; this is where the outcome is decided.

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


def resolve(event, rules: dict, first_decision=first_decision_of) -> TransactionResolved | None:
    """The final outcome for one slow-lane result, by plain rules. None = nothing to decide."""
    if isinstance(event, VerificationCompleted):
        resolution, rule_id = rules["verification"][event.payload.outcome]
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

    def handle_result(event):
        resolved = resolve(event, resolution_rules)
        if resolved is None:
            return
        # One final outcome per payment: the first one wins (SET NX), later ones are ignored.
        if not r.set(f"resolved:{event.txn_id}", resolved.event_id, nx=True):
            log.info("txn %s  already has a final outcome: %s ignored", event.txn_id[:8], event.event_type)
            return
        try:
            bus.publish(r, resolved)
        except Exception:
            r.delete(f"resolved:{event.txn_id}")   # let the retry through
            raise
        log.info("txn %s  FINAL: %-15s (%s, from %s)", event.txn_id[:8],
                 resolved.payload.resolution.upper(), resolved.payload.rule_id, event.event_type)

    host = f"{socket.gethostname()}-{os.getpid()}"
    streams = [("risk.scored", handle), ("verification.completed", handle_result),
               ("investigation.completed", handle_result), ("case.escalated", handle_result)]
    threads = []
    for stream, handler in streams:
        t = threading.Thread(target=bus.consume, daemon=True, name=stream, args=(
            bus.connect(), bus.stream_for(stream), GROUP, handler),
            kwargs={"consumer": f"{host}-{stream}"})
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
