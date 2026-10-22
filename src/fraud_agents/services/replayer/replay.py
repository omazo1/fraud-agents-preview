"""The replayer: plays the card processor (Part 1 · Step 7).

Picks real payments from fraudTest.csv, signs each one, and sends it to the
gateway as a webhook, at a steady speed. The answer key (is_fraud) is never sent;
it is only printed here so you can compare it with what the agents decide.

Run from the project folder (the gateway must be running):
    python -m fraud_agents.services.replayer.replay --genuine 45 --fraud 5 --rate 5
"""

import argparse
import json
import random
import time

import httpx
import pandas as pd

from fraud_agents.shared import config, signing

TEST_FILE = "data/raw/fraudTest.csv"
SENT_FIELDS = [
    "trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "lat", "long", "city_pop", "merch_lat", "merch_long",
]


def pick(genuine: int, fraud: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(TEST_FILE, usecols=SENT_FIELDS + ["is_fraud"], dtype={"cc_num": str})
    chosen = pd.concat([
        df[df.is_fraud == 0].sample(genuine, random_state=seed),
        df[df.is_fraud == 1].sample(fraud, random_state=seed),
    ])
    return chosen.sort_values("trans_date_trans_time")


def send(client: httpx.Client, url: str, secret: str, row) -> str:
    payment = {field: getattr(row, field) for field in SENT_FIELDS}
    payment["amt"] = str(payment["amt"])          # money as exact text, not a float
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
                return response.json().get("status", str(response.status_code))
        except httpx.TransportError:
            pass
        time.sleep(2 ** (attempt - 1))             # wait 1 s, then 2 s, before retrying
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine", type=int, default=45)
    parser.add_argument("--fraud", type=int, default=5)
    parser.add_argument("--rate", type=float, default=5, help="payments per second")
    parser.add_argument("--seed", type=int, default=None, help="same seed = same payments")
    parser.add_argument("--url", default="http://localhost:8000/webhooks/payments")
    args = parser.parse_args()

    secret = config.require("WEBHOOK_SECRET")
    seed = args.seed if args.seed is not None else random.randint(1, 1_000_000)
    rows = pick(args.genuine, args.fraud, seed)
    print(f"Replaying {len(rows)} payments (seed {seed}) at {args.rate}/s to {args.url}\n")

    counts: dict[str, int] = {}
    with httpx.Client(timeout=10) as client:
        for row in rows.itertuples():
            status = send(client, args.url, secret, row)
            counts[status] = counts.get(status, 0) + 1
            answer = "FRAUD" if row.is_fraud else "genuine"
            print(f"{status:<9} txn {row.trans_num[:8]}  ${row.amt:>8.2f}  {row.category:<14} (answer key: {answer})")
            time.sleep(1 / args.rate)
    print(f"\nDone: {counts}")


if __name__ == "__main__":
    main()
