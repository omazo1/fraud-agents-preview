"""The Decision agent's plain rules: the first decision and the final outcome."""

import json
from pathlib import Path

import pytest

from fraud_agents.agents.decision.agent import RESOLUTION_FILE, decide, load_rules, resolve
from fraud_agents.shared.events import (CaseEscalated, CaseEscalatedPayload, CaseReviewed,
                                        CaseReviewedPayload, InvestigationCompleted,
                                        InvestigationCompletedPayload, RiskScored, RiskScoredPayload,
                                        VerificationCompleted, VerificationCompletedPayload,
                                        new_event_id, now_utc)

RULES = load_rules()
FINAL = json.loads(Path(RESOLUTION_FILE).read_text())


def label(**payload):
    return dict(event_id=new_event_id(), published_at=now_utc(), txn_id="t1", occurred_at=now_utc(), **payload)


@pytest.mark.parametrize("score, decision", [(0.10, "approve"), (0.149, "approve"), (0.15, "flag"),
                                             (0.89, "flag"), (0.90, "hold"), (1.00, "hold")])
def test_first_decision_lines(score, decision):
    scored = RiskScored(**label(payload=RiskScoredPayload(score=score, model_version="test")))
    assert decide(scored, RULES).payload.decision == decision


def test_a_retry_the_customer_confirmed_is_approved_even_with_a_high_score():
    scored = RiskScored(**label(payload=RiskScoredPayload(score=0.99, model_version="test",
                                                          confirmed_retry_of="t0")))
    decided = decide(scored, RULES)
    assert (decided.payload.decision, decided.payload.rule_id) == ("approve", "customer_confirmed_retry")


def test_the_same_score_always_gives_the_same_decision_and_id():
    scored = RiskScored(**label(payload=RiskScoredPayload(score=0.7, model_version="test")))
    first, again = decide(scored, RULES), decide(scored, RULES)
    assert (first.event_id, first.payload.decision) == (again.event_id, again.payload.decision)


@pytest.mark.parametrize("outcome, final", [("denied_by_customer", "blocked"),
                                            ("confirmed_by_customer", "released"),
                                            ("unclear", "sent_to_analyst"), ("no_reply", "sent_to_analyst")])
def test_customer_answers(outcome, final):
    done = VerificationCompleted(**label(payload=VerificationCompletedPayload(outcome=outcome,
                                                                              based_on_event_id="r")))
    assert resolve(done, FINAL).payload.resolution == final


def investigation(action):
    return InvestigationCompleted(**label(payload=InvestigationCompletedPayload(
        verdict="likely_genuine", summary="A summary long enough for the form.", evidence=["x"],
        tools_used=["a"], recommended_action=action, model="test", based_on_event_id="d")))


def test_the_ai_never_releases_a_held_payment_alone():
    assert resolve(investigation("no_further_action"), FINAL, lambda _: "hold").payload.resolution == "sent_to_analyst"
    assert resolve(investigation("no_further_action"), FINAL, lambda _: "flag").payload.resolution == "released"


def test_asking_the_customer_is_left_to_verification():
    assert resolve(investigation("verify_with_customer"), FINAL) is None


def test_escalations_and_reviews():
    esc = CaseEscalated(**label(producer="investigation", payload=CaseEscalatedPayload(
        reason="test reason", based_on_event_id="d")))
    assert resolve(esc, FINAL).payload.resolution == "sent_to_analyst"
    review = CaseReviewed(**label(payload=CaseReviewedPayload(decision="block", analyst="a", based_on_event_id="o")))
    assert (resolve(review, FINAL).payload.resolution, resolve(review, FINAL).payload.rule_id) == \
        ("blocked", "analyst_blocked")


def test_the_rule_files_are_valid():
    assert 0 < RULES["flag_at"] < RULES["hold_at"] <= 1
    assert FINAL["gave_up"][0] == "sent_to_analyst"
