"""What may release a flagged payment without asking anyone (Investigation agent).

The tool texts below follow the real tools' sentence format (agents/investigation/tools.py)."""

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from fraud_agents.agents.investigation.agent import InvestigationReport, check, guard_action, ungrounded_numbers

MESSAGES = [
    HumanMessage("Investigate payment 8192ed35a33ff1ef264b735b2c89f8e3.\nFacts: Payment 8192ed35a33ff1ef264b735b2c89f8e3: "
                 "$318.72 at Kiehn Inc (shop type grocery_pos), time 2020-10-07 00:53:00+00:00 (UTC), 22 km from the "
                 "cardholder's home, card card-f3f1a40f013b7f4f. Fraud model score: 0.62 (factors that raised it: amount)."),
    ToolMessage(content="In the 24 hours before this payment, card card-f3f1a40f013b7f4f made 1 payment totalling $125.32, "
                        "in shop types: grocery_pos. Most recent: 2020-10-06 01:10 $125.32 grocery_pos.", tool_call_id="1"),
    ToolMessage(content="Card card-f3f1a40f013b7f4f made 412 payments in the previous 180 days. Typical amount $44.72 (95% of "
                        "payments were at or below $425.10); this payment is 7.1 times the typical amount. Most used shop "
                        "types: grocery_pos, gas_transport; this shop type (grocery_pos) was used 20 times before.",
                tool_call_id="2"),
    ToolMessage(content="In past records, 1.41% of 129,384 payments in shop type grocery_pos were fraud (all shop types: "
                        "0.58%). Typical amount in this shop type: $105.12.", tool_call_id="3"),
]


def report(verdict="likely_genuine", action="no_further_action",
           summary="The $318.72 payment is below the largest earlier amount of $425.10 for this card.",
           evidence=("95% of the card's payments were at or below $425.10",)):
    return {"verdict": verdict, "recommended_action": action, "summary": summary, "evidence": list(evidence)}


def test_the_ai_alone_never_releases_a_payment():
    # Even a fully grounded "no further action" becomes "ask the customer" (Part 3 · Step 6).
    action, why = guard_action(report(), MESSAGES)
    assert action == "verify_with_customer" and "never releases" in why


def test_the_old_behaviour_can_be_switched_back(monkeypatch):
    from fraud_agents.agents.investigation import agent
    monkeypatch.setattr(agent, "ALLOW_AI_RELEASE", True)
    assert agent.guard_action(report(), MESSAGES) == ("no_further_action", None)
    assert agent.guard_action(report(verdict="unsure"), MESSAGES)[0] == "verify_with_customer"


@pytest.mark.parametrize("line", [
    "The payment is 7.1 times the typical amount ($44.72).", "Shop type grocery_pos was used 20 times before.",
    "1.41% of payments in this shop type were fraud, against 0.58% overall.", "Fraud model score 0.62.",
    "The card spent $125.32 in the 24 hours before.", "The $318.72 payment was 22 km from home.",
    "The card made 412 payments in the previous 180 days.", "Typical amount in this shop type is $105.12.",
])
def test_honest_lines_are_grounded(line):
    assert ungrounded_numbers({"summary": "", "evidence": [line]}, MESSAGES) == []


@pytest.mark.parametrize("line", [
    "24 verified purchases at this shop",                 # "24" is from "in the 24 hours" (the review's example)
    "The customer has shopped here 20 times this week",   # "20" is about the shop type, not this week
    "The card made 412 trips abroad",                     # shares only generic words with "made 412 payments"
    "The card was used 180 times in Paris",
    "The payment is 3 times the typical amount",          # a number no tool gave
    "Made 1 earlier payment of $500",
    "Made 10 earlier purchases at Kiehn Inc",             # "10" is only in the date 2020-10-07
])
def test_made_up_lines_are_caught(line):
    assert ungrounded_numbers({"summary": "", "evidence": [line]}, MESSAGES)


def test_the_ais_own_numbers_are_not_a_source():
    messages = MESSAGES + [AIMessage("The card made 9 refunds at this shop before.")]
    assert ungrounded_numbers({"summary": "", "evidence": ["9 refunds at this shop before"]}, messages)


def test_a_made_up_number_cannot_release():
    action, why = guard_action(report(summary="The $318.72 payment was made here 12 times before."), MESSAGES)
    assert action == "verify_with_customer" and "12" in why


def test_evidence_in_words_only_cannot_release():
    action, _ = guard_action(report(evidence=["The customer has shopped here weekly for years"]), MESSAGES)
    assert action == "verify_with_customer"


def test_an_unsure_verdict_cannot_release():
    assert guard_action(report(verdict="unsure"), MESSAGES)[0] == "verify_with_customer"


def test_other_actions_pass_through():
    assert guard_action(report(verdict="likely_fraud", action="verify_with_customer"), MESSAGES) == \
        ("verify_with_customer", None)


def test_the_report_check_rejects_unsure_with_no_action():
    bad = InvestigationReport(verdict="unsure", summary="This payment is unclear and needs more checks overall.",
                              evidence=["x"], recommended_action="no_further_action")
    assert any("no_further_action" in p for p in check(bad, ["a", "b"]))
