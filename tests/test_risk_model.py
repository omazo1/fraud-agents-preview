"""The fraud-scoring model and its features (needs the saved model in models/)."""

import pytest

from fraud_agents.agents.risk_scoring.agent import RiskScorer


@pytest.fixture(scope="module")
def scorer():
    return RiskScorer()


def test_a_score_is_a_probability(scorer, payment):
    scored = scorer.score(payment)
    assert 0 <= scored.payload.score <= 1 and scored.payload.top_factors


def test_the_model_tells_a_risky_payment_from_an_ordinary_one(scorer, payment):
    # The fixture: an $843.91 online purchase at 22:06, far above typical (a real fraud in the test set).
    ordinary = payment.model_copy(update={
        "occurred_at": payment.occurred_at.replace(hour=12),
        "payload": payment.payload.model_copy(update={"amount": __import__("decimal").Decimal("4.50"),
                                                      "category": "grocery_pos"})})
    risky, small = scorer.score(payment).payload.score, scorer.score(ordinary).payload.score
    assert risky >= 0.5 > small and risky > 100 * small


def test_the_same_payment_gets_the_same_score(scorer, payment):
    assert scorer.score(payment).payload.score == scorer.score(payment).payload.score


def test_an_unknown_shop_type_is_refused(scorer, payment):
    odd = payment.model_copy(update={"payload": payment.payload.model_copy(update={"category": "space_travel"})})
    with pytest.raises(ValueError):
        scorer.score(odd)
