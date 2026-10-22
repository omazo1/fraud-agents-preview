"""The Investigation agent's tools (Part 2 · Step 3).

Each tool is a plain Python function that looks something up and returns a short
text answer. The @tool decorator (LangChain) turns it into something the AI model
can choose to call: the function name, its argument and its docstring are what the
model sees, so the docstrings are written for the model.

Every tool takes the payment's txn_id and only looks at the card's history BEFORE
that payment, like an investigator would at that moment.

Self-check:  python -m fraud_agents.agents.investigation.tools [txn_id]
"""

import json
import math
import statistics
import sys
import threading

from langchain.tools import tool

from fraud_agents.shared import db

_local = threading.local()


def _conn():
    """One database connection per thread, reconnecting if it dropped."""
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        conn = _local.conn = db.connect()
    return conn


def _payment(txn_id: str) -> dict | None:
    """The payment's details from the audit log (its "payment arrived" form)."""
    row = _conn().execute(
        "SELECT body FROM events WHERE txn_id = %s AND event_type = 'transaction.received'",
        (txn_id,),
    ).fetchone()
    if row is None:
        return None
    body = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    payment = dict(body["payload"])
    payment["occurred_at"] = body["occurred_at"]
    payment["amount"] = float(payment["amount"])
    # Every shop name in this simulated dataset starts with "fraud_", genuine or not.
    # Strip it so the AI model isn't misled by the name.
    payment["merchant"] = payment["merchant"].removeprefix("fraud_")
    return payment


def _km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def _not_found(txn_id: str) -> str:
    return f"No payment found with txn_id {txn_id}."


@tool
def get_payment_details(txn_id: str) -> str:
    """Get the facts of the payment under investigation: amount, shop type, shop,
    time, distance from the cardholder's home, and the fraud model's score with the
    factors that raised it. Use this first."""
    p = _payment(txn_id)
    if p is None:
        return _not_found(txn_id)
    score = _conn().execute(
        "SELECT body FROM events WHERE txn_id = %s AND event_type = 'risk.scored'", (txn_id,)
    ).fetchone()
    scored = (score[0] if isinstance(score[0], dict) else json.loads(score[0]))["payload"] if score else {}
    km = _km(p["home_lat"], p["home_long"], p["merchant_lat"], p["merchant_long"])
    return (
        f"Payment {txn_id}: ${p['amount']:.2f} at {p['merchant']} (shop type {p['category']}), "
        f"time {p['occurred_at']} (UTC), {km:.0f} km from the cardholder's home, card {p['card_id']}. "
        f"Fraud model score: {scored.get('score', 'unknown')} "
        f"(factors that raised it: {', '.join(scored.get('top_factors', [])) or 'none'})."
    )


@tool
def card_recent_activity(txn_id: str) -> str:
    """Check what the same card did in the 24 hours before this payment: how many
    payments, the total spent, which shop types, and the last few payments.
    Bursts of spending in a short time are a common fraud sign."""
    p = _payment(txn_id)
    if p is None:
        return _not_found(txn_id)
    rows = _conn().execute(
        "SELECT occurred_at, amount, category FROM card_payments "
        "WHERE card_id = %s AND occurred_at < %s AND occurred_at >= %s::timestamptz - interval '24 hours' "
        "ORDER BY occurred_at DESC",
        (p["card_id"], p["occurred_at"], p["occurred_at"]),
    ).fetchall()
    if not rows:
        return f"Card {p['card_id']} made no other payments in the 24 hours before this one."
    total = sum(float(r[1]) for r in rows)
    categories = sorted({r[2] for r in rows})
    last = "; ".join(f"{r[0]:%Y-%m-%d %H:%M} ${float(r[1]):.2f} {r[2]}" for r in rows[:5])
    return (
        f"In the 24 hours before this payment, card {p['card_id']} made {len(rows)} "
        f"payment{'s' if len(rows) != 1 else ''} "
        f"totalling ${total:.2f}, in shop types: {', '.join(categories)}. Most recent: {last}."
    )


@tool
def card_spending_profile(txn_id: str) -> str:
    """Compare this payment with the card's normal habits over the previous 180 days:
    the usual amount, the usual shop types and the usual hours. Tells you whether
    this payment is out of character for this card."""
    p = _payment(txn_id)
    if p is None:
        return _not_found(txn_id)
    rows = _conn().execute(
        "SELECT amount, category, extract(hour FROM occurred_at AT TIME ZONE 'UTC') FROM card_payments "
        "WHERE card_id = %s AND occurred_at < %s AND occurred_at >= %s::timestamptz - interval '180 days'",
        (p["card_id"], p["occurred_at"], p["occurred_at"]),
    ).fetchall()
    if len(rows) < 5:
        return (f"Card {p['card_id']} has too little history ({len(rows)} earlier payments in 180 days) "
                "to build a profile; treat that as unusual but not proof of fraud.")
    amounts = [float(r[0]) for r in rows]
    median = statistics.median(amounts)
    p95 = statistics.quantiles(amounts, n=20)[-1]  # 95% of payments were at or below this
    counts: dict[str, int] = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[:3]
    hour = int(p["occurred_at"][11:13])
    same_hour_share = sum(1 for r in rows if int(r[2]) == hour) / len(rows)
    ratio = p["amount"] / median if median else float("inf")
    return (
        f"Card {p['card_id']} made {len(rows)} payments in the previous 180 days. "
        f"Typical amount ${median:.2f} (95% of payments were at or below ${p95:.2f}); "
        f"this payment is {ratio:.1f} times the typical amount. "
        f"Most used shop types: {', '.join(top)}; this shop type "
        f"({p['category']}) was used {counts.get(p['category'], 0)} times before. "
        f"{same_hour_share:.0%} of the card's past payments were made in the same hour of day ({hour}:00 UTC)."
    )


@tool
def shop_type_risk(txn_id: str) -> str:
    """Look up how often payments in this payment's shop type turned out to be fraud
    in the bank's past records, compared with all shop types."""
    p = _payment(txn_id)
    if p is None:
        return _not_found(txn_id)
    conn = _conn()
    row = conn.execute(
        "SELECT payments, fraud_rate, median_amount FROM category_stats WHERE category = %s",
        (p["category"],),
    ).fetchone()
    overall = conn.execute(
        "SELECT sum(fraud_rate * payments) / sum(payments) FROM category_stats"
    ).fetchone()[0]
    if row is None:
        return f"No past records for shop type {p['category']}."
    return (
        f"In past records, {row[1]:.2%} of {row[0]:,} payments in shop type {p['category']} were fraud "
        f"(all shop types: {overall:.2%}). Typical amount in this shop type: ${row[2]:.2f}."
    )


TOOLS = [get_payment_details, card_recent_activity, card_spending_profile, shop_type_risk]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        txn = sys.argv[1]
    else:  # the most recent payment the Decision agent flagged or held
        found = _conn().execute(
            "SELECT txn_id FROM events WHERE event_type = 'transaction.decided' "
            "AND decision IN ('flag', 'hold') ORDER BY published_at DESC LIMIT 1"
        ).fetchone()
        if found is None:
            raise SystemExit("No flagged or held payment yet: run a replay first.")
        txn = found[0]
    print(f"Investigating {txn}\n")
    for t in TOOLS:
        print(f"[{t.name}]\n{t.invoke({'txn_id': txn})}\n")
