"""The event forms (Pydantic): what every event must contain."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fraud_agents.services.gateway.app import IncomingPayment
from fraud_agents.shared.events import (EVENT_TYPES, CaseReviewed, CaseReviewedPayload,
                                        TransactionDecidedPayload, derived_event_id, new_event_id,
                                        now_utc, parse_event, to_utc)


def test_a_form_survives_the_trip_through_text(payment):
    assert parse_event(payment.model_dump_json()) == payment


def test_all_eleven_forms_are_known():
    assert len(EVENT_TYPES) == 11 and "case.reviewed" in EVENT_TYPES


GOOD = dict(decision="hold", rule_id="x", rules_version="x", score=0.95, based_on_event_id="x")


def test_a_good_decision_form_is_accepted():
    assert TransactionDecidedPayload(**GOOD).decision == "hold"


@pytest.mark.parametrize("change", [{"decision": "maybe"}, {"score": 1.5}, {"score": -0.1}])
def test_one_bad_value_is_enough_to_refuse_a_form(change):
    with pytest.raises(ValidationError):
        TransactionDecidedPayload(**(GOOD | change))


def test_unknown_fields_are_refused(payment):
    data = payment.model_dump(mode="json")
    data["payload"]["is_fraud"] = 1          # the answer key must never travel in an event
    with pytest.raises(ValidationError):
        parse_event(__import__("json").dumps(data))


VALID_PAYMENT = {"trans_num": "t1", "trans_date_trans_time": "2020-06-21 12:14:25", "cc_num": "4613314721966",
                 "merchant": "fraud_Kilback LLC", "category": "shopping_net", "amt": "843.91", "lat": 40.0,
                 "long": -75.0, "city_pop": 5000, "merch_lat": 40.1, "merch_long": -75.2}


def test_the_gateway_accepts_a_complete_payment():
    assert IncomingPayment.model_validate(VALID_PAYMENT).trans_num == "t1"


def test_the_gateway_refuses_the_answer_key():
    with pytest.raises(ValidationError, match="is_fraud"):     # refused for being extra, nothing else
        IncomingPayment.model_validate(VALID_PAYMENT | {"is_fraud": 1})


def test_forms_cannot_be_changed_after_creation(payment):
    with pytest.raises(ValidationError):
        payment.txn_id = "something else"


def test_derived_ids_are_stable_and_distinct():
    assert derived_event_id("x", "decision") == derived_event_id("x", "decision")
    assert derived_event_id("x", "investigation", "investigation.completed") != \
        derived_event_id("x", "investigation", "case.escalated")


def test_only_analysts_send_case_reviewed():
    common = dict(event_id=new_event_id(), published_at=now_utc(), txn_id="t", occurred_at=now_utc(),
                  payload=CaseReviewedPayload(decision="block", analyst="omair", based_on_event_id="o"))
    assert CaseReviewed(**common).producer == "analyst"
    with pytest.raises(ValidationError):
        CaseReviewed(**common, producer="decision")


@pytest.mark.parametrize("text, expected", [
    ("2020-06-21 12:14:25", "2020-06-21T12:14:25+00:00"),          # no zone: treated as UTC
    ("2020-06-21T12:14:25-04:00", "2020-06-21T16:14:25+00:00"),    # a zone: converted, not overwritten
])
def test_times_end_up_in_utc(text, expected):
    assert to_utc(datetime.fromisoformat(text)).isoformat() == expected
