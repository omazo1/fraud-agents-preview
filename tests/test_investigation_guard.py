"""What may release a flagged payment without asking anyone (Investigation agent)."""

from langchain.messages import HumanMessage, ToolMessage

from fraud_agents.agents.investigation.agent import InvestigationReport, check, guard_action

MESSAGES = [HumanMessage("Investigate payment abc.\nFacts: {'amount': 318.72, 'score': 0.62}"),
            ToolMessage(content="{'typical_amount': 44.72, 'largest_before': 425.10, 'fraud_rate': 0.0141}",
                        tool_call_id="1")]


def report(verdict="likely_genuine", action="no_further_action", summary="The $318.72 payment is below "
           "the card's largest earlier payment of $425.10.", evidence=("Largest earlier payment $425.10",)):
    return {"verdict": verdict, "recommended_action": action, "summary": summary, "evidence": list(evidence)}


def test_a_fully_grounded_genuine_report_may_release():
    assert guard_action(report(), MESSAGES) == ("no_further_action", None)


def test_a_made_up_number_cannot_release():
    action, why = guard_action(report(summary="The $318.72 payment was made here 12 times before."), MESSAGES)
    assert action == "verify_with_customer" and "12" in why


def test_evidence_in_words_only_cannot_release():
    action, _ = guard_action(report(evidence=["The customer has shopped here weekly for years"]), MESSAGES)
    assert action == "verify_with_customer"


def test_an_unsure_verdict_cannot_release():
    assert guard_action(report(verdict="unsure"), MESSAGES)[0] == "verify_with_customer"


def test_percentages_match_tool_fractions():
    r = report(evidence=["Shop type risk 1.41%", "Largest earlier payment $425.10"])
    assert guard_action(r, MESSAGES) == ("no_further_action", None)


def test_other_actions_pass_through():
    assert guard_action(report(verdict="likely_fraud", action="verify_with_customer"), MESSAGES) == \
        ("verify_with_customer", None)


def test_the_report_check_rejects_unsure_with_no_action():
    bad = InvestigationReport(verdict="unsure", summary="This payment is unclear and needs more checks overall.",
                              evidence=["x"], recommended_action="no_further_action")
    assert any("no_further_action" in p for p in check(bad, ["a", "b"]))
