"""The Risk Scoring agent (Part 1 · Step 5; model v2 with card history: feedback #4).

Reads "payment arrived" forms (transaction.received), scores each payment with the
trained model, and publishes an "I scored it" form (risk.scored).

Model v2 also looks at the CARD'S OWN PAST: one indexed query to PostgreSQL's card_payments
table (the bank's records, loaded by investigation/history.py) for the card's payments
strictly before this one. It runs inside the gateway's 300 ms answer time, so it must stay fast.

Run from the project folder:  python -m fraud_agents.agents.risk_scoring.agent
Stop it with Ctrl + C.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb

import psycopg

from fraud_agents.agents.risk_scoring.features import FEATURES, HISTORY_SQL, history_values, make_features
from fraud_agents.shared import bus, db
from fraud_agents.shared.events import (
    RiskScored,
    RiskScoredPayload,
    TransactionReceived,
    derived_event_id,
    now_utc,
)

PRODUCER = "risk-scoring"
GROUP = "risk-scoring"
MODEL_FILE = Path("models/risk_model_v2.json")
META_FILE = Path("models/risk_model_v2.meta.json")

log = logging.getLogger(PRODUCER)


class CardHistory:
    """The card's past, from PostgreSQL (one connection, reopened if it drops).

    It runs inside the gateway's 300 ms answer time, so it gives up quickly: 1 s to connect,
    150 ms per query (normally 1-5 ms). If it fails, the payment isn't scored yet: the gateway
    answers a soft decline (no verdict in time) and the event is retried by the bus."""

    def __init__(self, conn=None):
        self.conn = conn                      # tests pass their own connection

    def __call__(self, card_id: str, occurred_at, merchant: str, category: str, amount: float) -> dict:
        args = {"card": card_id, "t": occurred_at, "merchant": merchant, "category": category}
        for attempt in (1, 2):
            try:
                if self.conn is None or self.conn.closed:
                    self.conn = db.connect(connect_timeout=1, statement_timeout_ms=150)
                return history_values(amount, *self.conn.execute(HISTORY_SQL, args).fetchone())
            except psycopg.OperationalError:
                self.conn = None              # PostgreSQL restarted: reconnect once, then give up
                if attempt == 2:
                    raise


class RiskScorer:
    """Loads the model once, then scores one payment at a time."""

    def __init__(self, model_file: Path = MODEL_FILE, meta_file: Path = META_FILE, history=None):
        self.meta = json.loads(meta_file.read_text())
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_file)
        self.booster = self.model.get_booster()
        self.history = history or CardHistory()   # tests pass a stand-in (no database)

    def score(self, event: TransactionReceived) -> RiskScored:
        p = event.payload
        past = self.history(p.card_id, event.occurred_at, p.merchant, p.category, float(p.amount))
        row = pd.DataFrame([past | {
            "occurred_at": event.occurred_at,
            "amount": float(p.amount),
            "category": p.category,
            "merchant_lat": p.merchant_lat,
            "merchant_long": p.merchant_long,
            "home_lat": p.home_lat,
            "home_long": p.home_long,
            "city_pop": p.city_pop,
        }])
        features = make_features(row, self.meta["categories"])
        score = float(self.model.predict_proba(features)[0, 1])

        # SHAP values: how much each feature pushed the score up or down (built into XGBoost).
        contributions = self.booster.predict(
            xgb.DMatrix(features, enable_categorical=True), pred_contribs=True
        )[0][:-1]  # the last column is the starting point (bias), not a feature
        pushed_up = sorted(
            (pair for pair in zip(FEATURES, contributions) if pair[1] > 0),
            key=lambda pair: pair[1], reverse=True,
        )
        top_factors = [name for name, _ in pushed_up[:3]]

        return RiskScored(
            event_id=derived_event_id(event.event_id, PRODUCER),
            published_at=now_utc(),
            txn_id=event.txn_id,
            occurred_at=event.occurred_at,
            payload=RiskScoredPayload(
                score=round(score, 6),
                model_version=self.meta["model_version"],
                top_factors=top_factors,
                confirmed_retry_of=event.payload.confirmed_retry_of,
            ),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    scorer = RiskScorer()
    r = bus.connect()
    r.ping()
    source = bus.stream_for("transaction.received")

    def handle(event):
        if not isinstance(event, TransactionReceived):
            raise ValueError(f"Expected transaction.received, got {event.event_type}")
        scored = scorer.score(event)
        bus.publish(r, scored)
        log.info(
            "txn %s  score %.4f  top factors %s",
            event.txn_id[:8], scored.payload.score, scored.payload.top_factors,
        )

    log.info("Listening on %s (group %s). Press Ctrl + C to stop.", source, GROUP)
    try:
        bus.consume(r, source, GROUP, handle)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
