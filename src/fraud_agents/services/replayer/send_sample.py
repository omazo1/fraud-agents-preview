"""Send a few real payments from the test set as "payment arrived" forms (Part 1 · Step 5).

A first, small version of the replayer; the full replayer comes in Step 7.
Run from the project folder:  python -m fraud_agents.services.replayer.send_sample
"""

import argparse
from decimal import Decimal

import pandas as pd

from fraud_agents.shared import bus
from fraud_agents.shared.cards import card_token   # the same keyed token as the gateway
from fraud_agents.shared.events import (
    TransactionReceived,
    TransactionReceivedPayload,
    derived_event_id,
    now_utc,
    to_utc,
)

TEST_FILE = "data/raw/fraudTest.csv"
COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "lat", "long", "city_pop", "trans_num", "merch_lat", "merch_long", "is_fraud",
]


def to_event(row) -> TransactionReceived:
    return TransactionReceived(
        # Built from the payment's own ID, so sending the same payment twice is spotted
        # as a duplicate instead of being scored again.
        event_id=derived_event_id(row.trans_num, "gateway"),
        published_at=now_utc(),
        txn_id=row.trans_num,
        # Assumption (Step 3): dataset times have no time zone; we treat them as UTC.
        occurred_at=to_utc(pd.Timestamp(row.trans_date_trans_time).to_pydatetime()),
        payload=TransactionReceivedPayload(
            card_id=card_token(row.cc_num),
            amount=Decimal(str(row.amt)),
            category=row.category,
            merchant=row.merchant,
            merchant_lat=row.merch_lat,
            merchant_long=row.merch_long,
            home_lat=row.lat,
            home_long=row.long,
            city_pop=int(row.city_pop),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genuine", type=int, default=10)
    parser.add_argument("--fraud", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(TEST_FILE, usecols=COLUMNS, nrows=200_000)
    sample = pd.concat([
        df[df.is_fraud == 0].head(args.genuine),
        df[df.is_fraud == 1].head(args.fraud),
    ]).sort_values("trans_date_trans_time")

    r = bus.connect()
    for row in sample.itertuples():
        bus.publish(r, to_event(row))  # is_fraud is NOT sent: it's only the answer key
        answer = "FRAUD" if row.is_fraud else "genuine"
        print(f"sent txn {row.trans_num[:8]}  ${row.amt:>8.2f}  {row.category:<14}  (answer key: {answer})")
    print(f"\nSent {len(sample)} payments to {bus.stream_for('transaction.received')}.")


if __name__ == "__main__":
    main()
