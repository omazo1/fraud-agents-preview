"""The safety rules that decide outcomes (moved out of the running agents so they can be tested)."""

import uuid

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from fraud_agents.agents.decision.agent import apply_review, claim_final_outcome
from fraud_agents.agents.investigation.agent import guard_action, to_event
from fraud_agents.agents.notification.agent import is_replaced
from fraud_agents.agents.verification.agent import accept_ai_label
from fraud_agents.shared.events import (CaseReviewed, CaseReviewedPayload, TransactionDecided,
                                        TransactionDecidedPayload, TransactionResolved,
                                        TransactionResolvedPayload, new_event_id, now_utc)


# ---------- the AI reading a customer's reply ----------

@pytest.mark.parametrize("ai_label, reply, label", [
    ("confirmed", "sure, that one's fine", "unclear"),          # an AI "yes" never releases a payment
    ("denied", "No problem, yes that was me", "unclear"),       # mixed signals: a human decides
    ("denied", "I never bought that", "denied"),
    ("unclear", "call me later", "unclear"),
])
def test_what_the_ai_may_decide(ai_label, reply, label):
    assert accept_ai_label(ai_label, reply)[0] == label


# ---------- one final outcome per payment ----------

def test_the_first_final_outcome_wins(redis_client):
    assert claim_final_outcome(redis_client, "t1", "outcome-A")
    assert not claim_final_outcome(redis_client, "t1", "outcome-B")     # a late one is ignored
    assert claim_final_outcome(redis_client, "t1", "outcome-A")         # a retry of the winner is fine


def review(case_id="case-1", txn="t1"):
    return CaseReviewed(event_id=new_event_id(), published_at=now_utc(), txn_id=txn, occurred_at=now_utc(),
                        payload=CaseReviewedPayload(decision="block", analyst="a", based_on_event_id=case_id))


def test_an_analyst_can_replace_a_waiting_case(redis_client):
    redis_client.set("resolved:t1", "case-1")
    verdict = apply_review(redis_client, review(), "new-1", lambda t: "case-1", lambda c, t: True)
    assert verdict == "apply" and redis_client.get("resolved:t1") == "new-1"
    assert apply_review(redis_client, review(), "new-1", lambda t: "case-1", lambda c, t: True) == "retry"


def test_an_analyst_cannot_replace_a_blocked_or_released_payment(redis_client):
    redis_client.set("resolved:t1", "blocked-1")
    verdict = apply_review(redis_client, review("blocked-1"), "new-1", lambda t: "blocked-1", lambda c, t: False)
    assert verdict == "not_an_analyst_case" and redis_client.get("resolved:t1") == "blocked-1"


def test_a_case_that_is_no_longer_waiting_is_not_changed(redis_client):
    redis_client.set("resolved:t1", "later-outcome")
    verdict = apply_review(redis_client, review(), "new-1", lambda t: "later-outcome", lambda c, t: True)
    assert verdict == "not_waiting" and redis_client.get("resolved:t1") == "later-outcome"


def test_a_lost_redis_note_is_restored_from_the_audit_log(redis_client):
    verdict = apply_review(redis_client, review(), "new-1", lambda t: "case-1", lambda c, t: True)
    assert verdict == "apply"


# ---------- no outdated texts ----------

def outcome(event_id):
    return TransactionResolved(event_id=event_id, published_at=now_utc(), txn_id="t1", occurred_at=now_utc(),
                               payload=TransactionResolvedPayload(resolution="sent_to_analyst", rule_id="x",
                                                                  rules_version="t", based_on_event_id="b"))


def test_a_replaced_outcome_is_not_texted(redis_client):
    redis_client.set("resolved:t1", "analyst-outcome")
    assert is_replaced(redis_client, outcome("old-reviewing-outcome"))
    assert not is_replaced(redis_client, outcome("analyst-outcome"))


# ---------- the investigation guard, end to end ----------

TOOLS = [HumanMessage("Investigate payment abc.\nFacts: {'amount': 318.72, 'score': 0.62}"),
         ToolMessage(content="{'largest_before': 425.10, 'fraud_rate': 0.0141}", tool_call_id="1")]


def report(summary, evidence):
    return {"verdict": "likely_genuine", "recommended_action": "no_further_action", "summary": summary,
            "evidence": evidence}


@pytest.mark.parametrize("evidence", [["Made 1 earlier payment at $425.10"],      # "1" is not 0.62 rounded
                                      ["Shop fraud rate 0% and $425.10 before"]])  # "0%" is not 1.41%
def test_tiny_numbers_do_not_round_into_found_ones(evidence):
    assert guard_action(report("The $318.72 payment is normal.", evidence), TOOLS)[0] == "verify_with_customer"


def test_the_ais_own_numbers_are_not_a_source():
    messages = TOOLS + [AIMessage("The card made 12 payments here before.")]
    r = report("The $318.72 payment is normal.", ["12 payments here before, largest $425.10"])
    assert guard_action(r, messages)[0] == "verify_with_customer"


def test_the_published_report_uses_the_guard():
    decided = TransactionDecided(event_id=new_event_id(), published_at=now_utc(), txn_id=uuid.uuid4().hex,
                                 occurred_at=now_utc(), payload=TransactionDecidedPayload(
                                     decision="flag", rule_id="x", rules_version="t", score=0.6,
                                     based_on_event_id="s"))
    result = {"problems": [], "tools_used": ["a", "b"], "messages": TOOLS,
              "report": report("The $318.72 payment was made here 12 times before.", ["Seen 12 times"])}
    assert to_event(decided, result, "test").payload.recommended_action == "verify_with_customer"


def test_failed_investigations_and_customer_checks_go_to_a_human(redis_client):
    from fraud_agents.agents.investigation.agent import give_up as investigation_gives_up
    from fraud_agents.shared import bus
    from fraud_agents.shared.events import parse_event
    held = TransactionDecided(event_id=new_event_id(), published_at=now_utc(), txn_id="t9", occurred_at=now_utc(),
                              payload=TransactionDecidedPayload(decision="hold", rule_id="x", rules_version="t",
                                                                score=0.95, based_on_event_id="s"))
    investigation_gives_up(redis_client, held, "failed 3 times: test")
    approved = held.model_copy(update={"event_id": new_event_id(),
                                       "payload": held.payload.model_copy(update={"decision": "approve"})})
    investigation_gives_up(redis_client, approved, "failed 3 times: test")    # nothing to escalate
    escalated = [parse_event(f["event"]) for _, f in redis_client.xrange(bus.stream_for("case.escalated"))]
    assert [e.txn_id for e in escalated] == ["t9"] and "human" in escalated[0].payload.reason
