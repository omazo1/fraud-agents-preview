"""Signed webhooks and card tokens."""

import time

from fraud_agents.shared.cards import card_token
from fraud_agents.shared.signing import sign, verify

SECRET = "test-secret"


def test_a_signed_body_verifies():
    ts, body = str(int(time.time())), b'{"amt": 4.97}'
    assert verify(SECRET, ts, sign(SECRET, ts, body), body)


def test_a_changed_body_fails():
    ts = str(int(time.time()))
    assert not verify(SECRET, ts, sign(SECRET, ts, b'{"amt": 4.97}'), b'{"amt": 9999}')


def test_the_wrong_secret_fails():
    ts, body = str(int(time.time())), b"x"
    assert not verify(SECRET, ts, sign("other", ts, body), body)


def test_an_old_signature_fails():
    ts, body = str(int(time.time()) - 3600), b"x"       # an hour old: a replay attack
    assert not verify(SECRET, ts, sign(SECRET, ts, body), body)


def test_card_tokens_are_stable_keyed_and_hide_the_number():
    token = card_token("4613314721966", key="k1")
    assert token == card_token("4613314721966", key="k1")
    assert token != card_token("4613314721966", key="k2")
    assert token.startswith("card-") and "4613314721966" not in token


def test_the_sample_sender_uses_the_gateways_token():
    from fraud_agents.services.replayer import send_sample
    assert send_sample.card_token is card_token
