"""Reading the customer's reply (Verification agent). Rules decide only clean answers."""

import pytest

from fraud_agents.agents.verification.agent import YES_SIGNAL, keyword_label


@pytest.mark.parametrize("reply, label", [
    # clean answers: the rules decide
    ("YES that was me", "confirmed"), ("Yes, it was me", "confirmed"), ("Y", "confirmed"),
    ("yes I made that payment", "confirmed"), ("yeah that was me", "confirmed"),
    ("NO that wasn't me!", "denied"), ("No. I did not make this payment", "denied"),
    ("no!! not me", "denied"), ("N", "denied"), ("not mine", "denied"),
    # a question back is never an answer
    ("who is this?", "question"), ("what payment?", "question"),
    # everything else goes to the AI (the review's examples first)
    ("Yesterday I made a different payment", "unknown"),
    ("No problem, yes that was me", "unknown"), ("no, it was me", "unknown"),
    ("Yes I did not make it", "unknown"), ("call me later", "unknown"),
    ("Yes ❌", "unknown"), ("yes 👎", "unknown"), ("Yes नहीं", "unknown"),
])
def test_keyword_label(reply, label):
    assert keyword_label(reply) == label


@pytest.mark.parametrize("reply", ["I did NOT buy that", "That wasn't mine",
                                   "No, I did not make this payment, please block my card"])
def test_denials_do_not_look_like_a_yes(reply):
    assert not YES_SIGNAL.search(reply)


def test_a_mixed_reply_has_a_yes_signal():
    assert YES_SIGNAL.search("No problem, yes that was me")
