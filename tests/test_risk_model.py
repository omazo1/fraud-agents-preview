"""The fraud-scoring model and its features (needs the saved model in models/, no database:
the card's past comes from a stand-in instead of PostgreSQL)."""

from decimal import Decimal

import pytest

from fraud_agents.agents.risk_scoring.agent import RiskScorer
from fraud_agents.agents.risk_scoring.features import history_values


def card_in_a_burst(card_id, occurred_at, merchant, category, amount):
    """The real past of a card just before a real fraud in the test set (txn 1799ffe4): 520 earlier
    payments, usually about $62, but $1,400.85 spent in the last 24 hours; never at this shop."""
    in_category = {"shopping_net": (43, 88.96), "grocery_pos": (41, 128.30)}.get(category, (0, None))
    return history_values(amount, 520, 62.47, 1006.48, 0, 1400.85, 0, *in_category)


def big_spender(card_id, occurred_at, merchant, category, amount):
    """A card that often pays this shop about this much."""
    return history_values(amount, 500, 800.0, 1500.0, 0, 900.0, 5, 200, 850.0)


@pytest.fixture(scope="module")
def scorer():
    return RiskScorer(history=card_in_a_burst)


def test_a_score_is_a_probability(scorer, payment):
    scored = scorer.score(payment)
    assert 0 <= scored.payload.score <= 1 and scored.payload.top_factors


def test_the_model_tells_a_risky_payment_from_an_ordinary_one(scorer, payment):
    # The fixture: an $843.91 online purchase at 22:06, on a card in a spending burst.
    ordinary = payment.model_copy(update={
        "occurred_at": payment.occurred_at.replace(hour=12),
        "payload": payment.payload.model_copy(update={"amount": Decimal("4.50"),
                                                      "category": "grocery_pos"})})
    risky, small = scorer.score(payment).payload.score, scorer.score(ordinary).payload.score
    assert risky >= 0.5 > small and risky > 100 * small


def test_the_cards_own_past_changes_the_score(scorer, payment):
    # The same $843.91 is far less suspicious on a card that often spends that much there.
    usual = RiskScorer(history=big_spender).score(payment).payload.score
    assert usual < scorer.score(payment).payload.score / 10


def test_the_same_payment_gets_the_same_score(scorer, payment):
    assert scorer.score(payment).payload.score == scorer.score(payment).payload.score


def test_an_unknown_shop_type_is_refused(scorer, payment):
    odd = payment.model_copy(update={"payload": payment.payload.model_copy(update={"category": "space_travel"})})
    with pytest.raises(ValueError):
        scorer.score(odd)


def test_a_confirmed_retry_is_passed_on_to_the_decision_rules(scorer, payment):
    retry = payment.model_copy(update={"payload": payment.payload.model_copy(
        update={"confirmed_retry_of": "first-txn"})})
    assert scorer.score(retry).payload.confirmed_retry_of == "first-txn"
    assert scorer.score(payment).payload.confirmed_retry_of is None
