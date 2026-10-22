"""The Decision agent (Part 1 · Step 6). Plain rules, no AI, on purpose.

Reads "I scored it" forms (risk.scored) and publishes "I decided" forms
(transaction.decided): approve, flag (let it through, investigate afterwards)
or hold (pause it until checked). The lines live in rules_v1.json.

Run from the project folder:  python -m fraud_agents.agents.decision.agent
Stop it with Ctrl + C.
"""

import json
import logging
from pathlib import Path

from fraud_agents.shared import bus
from fraud_agents.shared.events import (
    RiskScored,
    TransactionDecided,
    TransactionDecidedPayload,
    derived_event_id,
    now_utc,
)

PRODUCER = "decision"
GROUP = "decision"
RULES_FILE = Path(__file__).with_name("rules_v1.json")

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rules = load_rules()
    r = bus.connect()
    r.ping()
    source = bus.stream_for("risk.scored")

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

    log.info(
        "Listening on %s (group %s). Rules %s: hold at %.2f, flag at %.2f. Press Ctrl + C to stop.",
        source, GROUP, rules["rules_version"], rules["hold_at"], rules["flag_at"],
    )
    try:
        bus.consume(r, source, GROUP, handle)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
