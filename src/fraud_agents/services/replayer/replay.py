"""The replayer: plays the card processor (Part 1 · Step 7).

Picks real payments from fraudTest.csv, signs each one, and sends it to the
gateway as a webhook, at a steady speed. The answer key (is_fraud) is never sent;
it is only printed here so you can compare it with what the agents decide.

Feedback #2: the gateway now ANSWERS each payment (approved, or declined for now = a soft
decline), and the replayer prints the answer and how long it took. It also plays the
customer at the till:
- declined because there was no verdict in time: they simply tap again (a new payment);
- declined to check with them: when the bank says they confirmed (GET /payments/{id}),
  they try the same payment again, and it should be approved.

Run from the project folder (the gateway must be running):
    python -m fraud_agents.services.replayer.replay --genuine 45 --fraud 5 --rate 5
    python -m fraud_agents.services.replayer.replay --natural 1000 --rate 10   # real mix (~0.4% fraud)
"""

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timedelta

import httpx
import pandas as pd

from fraud_agents.shared import config, signing

TEST_FILE = "data/raw/fraudTest.csv"
SENT_FIELDS = [
    "trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "lat", "long", "city_pop", "merch_lat", "merch_long",
]


def pick(genuine: int, fraud: int, seed: int, natural: int = 0) -> pd.DataFrame:
    """Chosen numbers of genuine and fraud payments (a test mix, e.g. 10% fraud), or with
    `natural` that many payments drawn at random, so fraud comes at the test file's real
    rate (about 0.4%), like real card traffic (feedback #3)."""
    df = pd.read_csv(TEST_FILE, usecols=SENT_FIELDS + ["is_fraud"], dtype={"cc_num": str})
    if natural:
        chosen = df.sample(natural, random_state=seed)
    else:
        chosen = pd.concat([
            df[df.is_fraud == 0].sample(genuine, random_state=seed),
            df[df.is_fraud == 1].sample(fraud, random_state=seed),
        ])
    return chosen.sort_values("trans_date_trans_time")


def send(client: httpx.Client, url: str, secret: str, row, retry_of: str | None = None,
         later: timedelta = timedelta(0)) -> dict:
    """Send one payment; returns the gateway's answer (or {"status": "failed"}).
    A retry is a NEW payment (new trans_num) for the same card, shop and amount."""
    payment = {field: getattr(row, field) for field in SENT_FIELDS}
    payment["amt"] = str(payment["amt"])          # money as exact text, not a float
    if retry_of:
        payment["trans_num"] = f"{retry_of}-retry"
        when = datetime.fromisoformat(str(payment["trans_date_trans_time"])) + later
        payment["trans_date_trans_time"] = when.isoformat(sep=" ")
    body = json.dumps(payment, default=str).encode()
    for attempt in range(1, 4):                    # up to 3 tries
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Timestamp": timestamp,
            "X-Signature": signing.sign(secret, timestamp, body),
        }
        try:
            response = client.post(url, content=body, headers=headers)
            if response.status_code < 500:
                answer = response.json()
                return answer if "status" in answer else {"status": str(response.status_code)}
        except httpx.TransportError:
            pass
        time.sleep(2 ** (attempt - 1))             # wait 1 s, then 2 s, before retrying
    return {"status": "failed"}


def status_of(client: httpx.Client, base: str, secret: str, txn_id: str) -> dict:
    """Ask the bank for a payment's status now (signed, like the webhook)."""
    timestamp = str(int(time.time()))
    signed = f"GET /payments/{txn_id}".encode()          # the method and path, not just the ID
    headers = {"X-Timestamp": timestamp, "X-Signature": signing.sign(secret, timestamp, signed)}
    try:
        response = client.get(f"{base}/payments/{txn_id}", headers=headers)
        return response.json() if response.status_code == 200 else {}
    except httpx.TransportError:
        return {}


def show(answer: dict, row, label: str = "") -> None:
    truth = "FRAUD" if row.is_fraud else "genuine"
    took = f"{answer['decided_in_ms']:>4} ms" if "decided_in_ms" in answer else "       "
    kind = f"{answer['status']} ({answer.get('reason', '')})" if answer.get("reason") else answer["status"]
    print(f"{label}{kind:<38} {took}  txn {row.trans_num[:8]}  ${row.amt:>8.2f}  "
          f"{row.category:<14} (answer key: {truth})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine", type=int, default=45)
    parser.add_argument("--fraud", type=int, default=5)
    parser.add_argument("--natural", type=int, default=0,
                        help="send this many RANDOM payments (real fraud rate, ~0.4%%) instead of --genuine/--fraud")
    parser.add_argument("--rate", type=float, default=5, help="payments per second")
    parser.add_argument("--seed", type=int, default=None, help="same seed = same payments")
    parser.add_argument("--url", default="http://localhost:8000/webhooks/payments")
    parser.add_argument("--retry-wait", type=float, default=600,
                        help="seconds to wait for customers' answers before retrying (0 = no retries)")
    args = parser.parse_args()

    secret = config.require("WEBHOOK_SECRET")
    seed = args.seed if args.seed is not None else random.randint(1, 1_000_000)
    rows = pick(args.genuine, args.fraud, seed, args.natural)
    mix = f"real mix: {int(rows.is_fraud.sum())} fraud" if args.natural else "chosen mix"
    print(f"Replaying {len(rows)} payments ({mix}, seed {seed}) at {args.rate}/s to {args.url}\n")

    base = args.url.split("/webhooks/")[0]
    counts: dict[str, int] = {}
    times: list[int] = []
    to_check = []                           # soft-declined to ask the customer: check back later
    with httpx.Client(timeout=10) as client:
        for row in rows.itertuples():
            answer = send(client, args.url, secret, row)
            kind = f"{answer['status']} ({answer['reason']})" if answer.get("reason") else answer["status"]
            counts[kind] = counts.get(kind, 0) + 1
            if "decided_in_ms" in answer:
                times.append(answer["decided_in_ms"])
            show(answer, row)
            if answer.get("reason") == "no_decision_in_time" and args.retry_wait > 0:
                time.sleep(1)                # the customer just taps the card again
                show(send(client, args.url, secret, row, retry_of=row.trans_num,
                          later=timedelta(seconds=5)), row, "  tapped again -> ")
            elif answer.get("reason") == "verify_with_customer":
                to_check.append(row)
            time.sleep(1 / args.rate)
        print(f"\nAnswers: {counts}")
        if times:
            print(f"Time to answer: median {statistics.median(times):.0f} ms · "
                  f"95th percentile {sorted(times)[int(0.95 * (len(times) - 1))]} ms · max {max(times)} ms")
        if to_check and args.retry_wait > 0:
            retry_confirmed(client, args.url, base, secret, to_check, args.retry_wait)


def retry_confirmed(client, url, base, secret, to_check, wait_s: float) -> None:
    """Wait for the customers' answers; whoever confirmed tries the payment again."""
    print(f"\nWaiting up to {wait_s:.0f} s for {len(to_check)} customers to answer the bank's text ...")
    done: dict[str, dict] = {}
    deadline = time.time() + wait_s
    while len(done) < len(to_check) and time.time() < deadline:
        for row in to_check:
            if row.trans_num not in done:
                status = status_of(client, base, secret, row.trans_num)
                if status.get("outcome"):
                    done[row.trans_num] = status
        time.sleep(5)
    retried = 0
    for row in to_check:
        status = done.get(row.trans_num)
        if status is None:
            print(f"still waiting                                 txn {row.trans_num[:8]}")
        elif status["outcome"] == "retry_allowed":
            retried += 1
            show(send(client, url, secret, row, retry_of=row.trans_num, later=timedelta(minutes=2)),
                 row, "  confirmed, tried again -> ")
        else:
            print(f"{status['status']:<45} txn {row.trans_num[:8]}  ${row.amt:>8.2f}  "
                  f"(answer key: {'FRAUD' if row.is_fraud else 'genuine'})")
    print(f"\nDone: {retried} confirmed payments tried again.")


if __name__ == "__main__":
    main()
