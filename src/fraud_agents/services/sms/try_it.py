"""Try the SMS round trip by hand (Part 2 · Step 5), before the Verification agent exists.

Sends "was this you?" for one caught payment to the mock SMS provider, then waits for
the customer's reply to come back through the gateway as a customer.replied event.

Run from the project folder (gateway on 8000 and SMS provider on 8100 running):
    python -m fraud_agents.services.sms.try_it [txn_id]
"""

import json
import os
import sys
import time

import httpx

from fraud_agents.shared import bus, config, db
from fraud_agents.shared.events import new_event_id, parse_event

PROVIDER_URL = os.environ.get("SMS_PROVIDER_URL", "http://localhost:8100") + "/messages"


def main() -> None:
    conn = db.connect()
    if len(sys.argv) > 1:
        txn = sys.argv[1]
    else:
        row = conn.execute(
            "SELECT txn_id FROM events WHERE event_type = 'transaction.decided' "
            "AND decision IN ('flag', 'hold') ORDER BY published_at DESC LIMIT 1").fetchone()
        if row is None:
            raise SystemExit("No flagged or held payment yet: run a replay first.")
        txn = row[0]
    body = conn.execute("SELECT body FROM events WHERE txn_id = %s AND event_type = "
                        "'transaction.received'", (txn,)).fetchone()[0]
    payment = (body if isinstance(body, dict) else json.loads(body))["payload"]
    merchant = payment["merchant"].removeprefix("fraud_")
    text = (f"Bank alert: did you just pay ${float(payment['amount']):.2f} at {merchant}? "
            f"Reply YES or NO.")
    reference = new_event_id()   # in Step 6 this will be the verification.requested event_id

    r = bus.connect()
    start_id = "$"  # only replies that arrive from now on
    last = r.xrevrange(bus.stream_for("customer.replied"), count=1)
    if last:
        start_id = last[0][0]

    response = httpx.post(
        PROVIDER_URL, headers={"X-Api-Key": config.require("SMS_API_KEY")},
        json={"to": payment["card_id"], "text": text, "reference": reference,
              "txn_id": txn, "expects_reply": True},
    )
    print(f"Sent to {payment['card_id']}: {text}\nProvider said: {response.status_code} {response.json()}\n")
    print("Waiting up to 20 s for the customer's reply ...")

    deadline = time.time() + 20
    while time.time() < deadline:
        reply = r.xread({bus.stream_for("customer.replied"): start_id}, count=10, block=2000)
        for _, entries in reply or []:
            for entry_id, fields in entries:
                start_id = entry_id
                event = parse_event(fields["event"])
                if event.payload.in_reply_to == reference:
                    print(f"Customer replied: {event.payload.reply_text!r}")
                    print(f"(customer.replied event {event.event_id}, matched by in_reply_to)")
                    return
    print("No reply within 20 s. Some simulated customers never answer: that is on purpose.")


if __name__ == "__main__":
    main()
