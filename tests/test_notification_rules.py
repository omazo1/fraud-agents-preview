"""The rules every customer text must pass (Notification agent)."""

import pytest

from fraud_agents.agents.notification.agent import MAX_LENGTH, TEMPLATES, check, template_text

AMOUNT, SHOP = "812.66", "Durgan-Auer"


@pytest.mark.parametrize("outcome, text", [
    ("blocked", "Bank: Your $812.66 payment to Durgan-Auer was blocked to protect your card."),
    ("blocked", "Bank: We blocked a $812.66 payment to Durgan-Auer as you reported it was not authorized."),
    ("released", "Bank: The payment is approved. The $812.66 payment at Durgan-Auer is fine."),
    ("sent_to_analyst", "Bank: A $812.66 payment was made to Durgan-Auer. Our fraud team is reviewing "
                        "the transaction and will contact you."),
])
def test_good_texts_pass(outcome, text):
    assert check(text, outcome, AMOUNT, SHOP) == []


@pytest.mark.parametrize("outcome, text", [
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. It is now approved and released."),
    ("released", "Bank: the $812.66 payment at Durgan-Auer is approved, but your card is blocked."),
    ("blocked", "Bank: the $812.66 payment at Durgan-Auer wasn't blocked."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Reply with your account number."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Send us the one-time code."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Call us at 416-555-0199."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Visit bank-secure.com."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Send your CVC."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Reply with your P.I.N."),
    ("blocked", "Bank: we blocked the $12.66 payment at Durgan-Auer."),          # wrong amount
    ("blocked", "We blocked the $812.66 payment at Durgan-Auer."),               # no "Bank:"
    ("blocked", "Bank: we blocked the $812.66 payment."),                         # no shop
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Your card is now protected."),  # overclaim
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. A new card is on its way."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Your card is now fully protected."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. We froze your card."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer and blocked your card."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. A replacement is on its way."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer. Your card has been suspended."),
    ("blocked", "Bank: we blocked the $812.66 payment at Durgan-Auer and stopped your card."),
])
def test_bad_texts_are_rejected(outcome, text):
    assert check(text, outcome, AMOUNT, SHOP)


@pytest.mark.parametrize("key", list(TEMPLATES))
@pytest.mark.parametrize("shop", ["Durgan-Auer", "Yost, Block and Koepp", "Schmeler, Bashirian and Price"])
def test_every_template_passes_its_own_rules(key, shop):
    assert check(template_text(key, AMOUNT, shop), key, AMOUNT, shop) == []


def test_a_very_long_shop_name_still_fits_one_text():
    assert len(template_text("blocked", AMOUNT, "X" * 400)) <= MAX_LENGTH


@pytest.mark.parametrize("text", [
    "Bank: Your $812.66 payment to Durgan-Auer was blocked to protect your card.",   # a purpose, not a claim
    "Bank: Your card payment of $812.66 to Durgan-Auer has been blocked.",            # about the payment
    "Bank: we blocked your card payment of $812.66 at Durgan-Auer.",
    "Bank: we blocked the $812.66 payment at Durgan-Auer; no new card is needed.",
])
def test_saying_what_happened_to_the_payment_is_fine(text):
    assert check(text, "blocked", AMOUNT, SHOP) == []


def test_a_template_that_fails_its_checks_is_never_sent(monkeypatch):
    from fraud_agents.agents.notification import agent
    monkeypatch.setitem(agent.TEMPLATES, "blocked", "Bank: your card is now protected.")   # a broken template

    class NoAI:
        model = "none"
        def invoke(self, *args, **kwargs):
            raise RuntimeError("the AI is down (test)")

    graph, _ = agent.build_graph(NoAI())
    result = graph.invoke({"resolution": "blocked", "amount": AMOUNT, "merchant": SHOP, "draft": None,
                           "problems": [], "message": None, "used_template": False})
    assert result["message"] == agent.NEUTRAL_TEXT
