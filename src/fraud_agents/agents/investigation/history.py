"""Load the bank's payment history into PostgreSQL (Part 2 · Step 3). Run once.

The Investigation agent's tools look up a card's past payments, like a real bank's
records. This copies every payment in fraudTrain.csv and fraudTest.csv into the
card_payments table, with the card number replaced by its token.
The answer key (is_fraud) is NOT copied into card_payments.

category_stats holds each shop type's historical fraud rate, from fraudTrain.csv only
(the past), like a bank's record of confirmed fraud cases.

Run from the project folder:  python -m fraud_agents.agents.investigation.history
"""

import io
import time

import pandas as pd

from fraud_agents.shared import db
from fraud_agents.shared.cards import card_token

FILES = ["data/raw/fraudTrain.csv", "data/raw/fraudTest.csv"]
COLUMNS = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt",
           "lat", "long", "merch_lat", "merch_long", "is_fraud"]

SCHEMA = """
DROP TABLE IF EXISTS card_payments;
CREATE TABLE card_payments (
    card_id        TEXT NOT NULL,
    occurred_at    TIMESTAMPTZ NOT NULL,
    amount         NUMERIC(12, 2) NOT NULL,
    category       TEXT NOT NULL,
    merchant       TEXT NOT NULL,
    merchant_lat   DOUBLE PRECISION,
    merchant_long  DOUBLE PRECISION,
    home_lat       DOUBLE PRECISION,
    home_long      DOUBLE PRECISION
);
DROP TABLE IF EXISTS category_stats;
CREATE TABLE category_stats (
    category       TEXT PRIMARY KEY,
    payments       INTEGER NOT NULL,
    fraud_rate     DOUBLE PRECISION NOT NULL,
    median_amount  DOUBLE PRECISION NOT NULL
);
"""


def main() -> None:
    start = time.time()
    print("Reading the dataset ...")
    frames = [pd.read_csv(f, usecols=COLUMNS, dtype={"cc_num": str}) for f in FILES]
    train, df = frames[0], pd.concat(frames, ignore_index=True)

    tokens = {cc: card_token(cc) for cc in df["cc_num"].unique()}
    out = pd.DataFrame({
        "card_id": df["cc_num"].map(tokens),
        # Assumption (Step 3): the dataset's times have no time zone; we treat them as UTC.
        "occurred_at": df["trans_date_trans_time"] + "+00",
        "amount": df["amt"],
        "category": df["category"],
        "merchant": df["merchant"],
        "merchant_lat": df["merch_lat"],
        "merchant_long": df["merch_long"],
        "home_lat": df["lat"],
        "home_long": df["long"],
    })

    stats = train.groupby("category").agg(
        payments=("amt", "size"), fraud_rate=("is_fraud", "mean"), median_amount=("amt", "median")
    ).reset_index()

    with db.connect() as conn:
        conn.execute(SCHEMA)
        print(f"Copying {len(out):,} payments for {len(tokens):,} cards into card_payments ...")
        with conn.cursor().copy(
            "COPY card_payments (card_id, occurred_at, amount, category, merchant, merchant_lat,"
            " merchant_long, home_lat, home_long) FROM STDIN WITH (FORMAT csv)"
        ) as copy:
            for chunk_start in range(0, len(out), 200_000):
                buffer = io.StringIO()
                out.iloc[chunk_start:chunk_start + 200_000].to_csv(buffer, header=False, index=False)
                copy.write(buffer.getvalue())
        conn.execute("CREATE INDEX card_payments_card_time ON card_payments (card_id, occurred_at)")
        with conn.cursor().copy(
            "COPY category_stats (category, payments, fraud_rate, median_amount) FROM STDIN"
        ) as copy:
            for row in stats.itertuples(index=False):
                copy.write_row(tuple(row))
        conn.execute("ANALYZE card_payments")
        rows = conn.execute("SELECT count(*), count(DISTINCT card_id) FROM card_payments").fetchone()

    print(f"Done in {time.time() - start:.0f} s: {rows[0]:,} payments, {rows[1]:,} cards, "
          f"{len(stats)} shop categories.")


if __name__ == "__main__":
    main()
