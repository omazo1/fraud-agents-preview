"""Train the fraud-scoring model (Part 1 · Step 4; v2 with card history: feedback #4).

Run from the project folder:  python -m fraud_agents.agents.risk_scoring.train
Reads  data/raw/fraudTrain.csv and data/raw/fraudTest.csv
Writes models/risk_model_v2.json (the model) and models/risk_model_v2.meta.json (its details)

Split by TIME, never at random: it trains on fraudTrain.csv (Jan 2019 to Jun 2020; its
latest 10% only decides when to stop) and tests on fraudTest.csv (Jun to Dec 2020), which
comes after it. The card-history features use each card's strictly earlier payments,
across both files, so a test payment sees the card's past, never its future.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_agents.agents.risk_scoring.features import FEATURES, history_from_frame, make_features

MODEL_VERSION = "risk_model_v2"
DATA = Path("data/raw")
MODELS = Path("models")

# Dataset column -> our event-form name (the same mapping the gateway will use).
RENAME = {
    "trans_date_trans_time": "occurred_at",
    "amt": "amount",
    "merch_lat": "merchant_lat",
    "merch_long": "merchant_long",
    "lat": "home_lat",
    "long": "home_long",
}
COLUMNS = list(RENAME) + ["category", "city_pop", "is_fraud", "cc_num", "merchant"]


def load(path: Path) -> pd.DataFrame:
    """Read only the columns we use. is_fraud is the answer key, kept apart from the features."""
    df = pd.read_csv(path, usecols=COLUMNS, dtype={"cc_num": str}).rename(columns=RENAME)
    # Assumption (Step 3): the dataset's times have no time zone; we treat them as UTC.
    df["occurred_at"] = pd.to_datetime(df["occurred_at"]).dt.tz_localize("UTC")
    return df.sort_values("occurred_at", kind="stable").reset_index(drop=True)


def with_history(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the card-history features, from each card's strictly earlier payments in BOTH files
    (like the live agent's card_payments table). The answer key is not used for them."""
    both = pd.concat([train.assign(part="train"), test.assign(part="test")], ignore_index=True)
    both = pd.concat([both, history_from_frame(both, card="cc_num")], axis=1)
    return (both[both.part == "train"].reset_index(drop=True),
            both[both.part == "test"].reset_index(drop=True))


LINES = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9)   # possible flag lines to report on


def first_alarms(frame: pd.DataFrame, scores: np.ndarray, line: float) -> dict:
    """Fraud comes in BURSTS here: a stolen card is used again and again. Catching the 5th
    payment of a burst counts as a catch, but it is the FIRST alarm that matters most. Per
    card with fraud: was its first fraud caught, and how much fraud money got through before
    the first alarm?"""
    fraud = frame.assign(score=scores)[frame["is_fraud"] == 1].sort_values("occurred_at", kind="stable")
    first_caught, let_through = 0, 0.0
    for _, card in fraud.groupby("cc_num"):
        hits = np.flatnonzero(card["score"].to_numpy() >= line)
        first_caught += int(len(hits) > 0 and hits[0] == 0)
        let_through += float(card["amount"].to_numpy()[: hits[0] if len(hits) else len(card)].sum())
    return {"fraud_cards": int(fraud["cc_num"].nunique()), "first_fraud_caught": first_caught,
            "fraud_money_before_first_alarm": round(let_through, 2)}


def report(name: str, answers: np.ndarray, scores: np.ndarray, frame: pd.DataFrame) -> dict:
    """How good are the scores? Prints and returns the numbers."""
    result = {
        "rows": int(len(answers)),
        "fraud_rows": int(answers.sum()),
        "roc_auc": round(float(roc_auc_score(answers, scores)), 4),
        "pr_auc": round(float(average_precision_score(answers, scores)), 4),
        "at_threshold": {},
    }
    print(f"\n{name}: {result['rows']:,} payments, {result['fraud_rows']:,} fraud")
    print(f"  ROC-AUC {result['roc_auc']}   PR-AUC {result['pr_auc']}")
    print("  If we flag every payment with a score of at least ...")
    for t in LINES:
        flagged = scores >= t
        caught = int((flagged & (answers == 1)).sum())
        false_alarms = int((flagged & (answers == 0)).sum())
        missed = int(answers.sum()) - caught
        first = first_alarms(frame, scores, t)
        result["at_threshold"][str(t)] = {
            "caught": caught, "missed": missed, "false_alarms": false_alarms, **first,
        }
        print(f"    {t}: caught {caught:,}, missed {missed:,}, false alarms {false_alarms:,}; "
              f"first fraud caught on {first['first_fraud_caught']} of {first['fraud_cards']} cards, "
              f"${first['fraud_money_before_first_alarm']:,.0f} let through before the first alarm")
    return result


def main() -> None:
    print("Reading the dataset ...")
    train = load(DATA / "fraudTrain.csv")
    test = load(DATA / "fraudTest.csv")
    print("Adding each card's history (earlier payments only) ...")
    train, test = with_history(train, test)
    categories = sorted(train["category"].unique().tolist())

    # Hold back the latest 10% of the training data to decide when to stop training.
    cut = int(len(train) * 0.9)
    fit, check = train.iloc[:cut], train.iloc[cut:]

    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
    )
    print("Training (this takes about a minute) ...")
    model.fit(
        make_features(fit, categories), fit["is_fraud"],
        eval_set=[(make_features(check, categories), check["is_fraud"])],
        verbose=False,
    )
    print(f"Stopped after {model.best_iteration + 1} trees.")

    # The flag line is CHOSEN here, on the validation slice (training time: not trained on, but
    # also used for early stopping above), not on the test file: see rules_v2.json for the rule.
    check_scores = model.predict_proba(make_features(check, categories))[:, 1]
    check_result = report("Validation slice (latest 10% of training time, for choosing the flag line)",
                          check["is_fraud"].to_numpy(), check_scores, check)

    # The real test: fraudTest.csv, which the model has never seen.
    test_scores = model.predict_proba(make_features(test, categories))[:, 1]
    test_result = report("Test set (never seen)", test["is_fraud"].to_numpy(), test_scores, test)

    # Keep only the best trees, so anyone loading the file gets exactly the tested model.
    MODELS.mkdir(exist_ok=True)
    best = model.get_booster()[: model.best_iteration + 1]
    best.save_model(MODELS / f"{MODEL_VERSION}.json")
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "categories": categories,
        "trees": int(model.best_iteration + 1),
        "trained_on": "data/raw/fraudTrain.csv (first 90% by time; last 10% used for early stopping)",
        "tested_on": "data/raw/fraudTest.csv (after the training data in time)",
        "card_history": "each card's strictly earlier payments in both files (card_payments table when live)",
        "validation": check_result,
        "test": test_result,
        "xgboost_version": xgb.__version__,
    }
    (MODELS / f"{MODEL_VERSION}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {MODELS / (MODEL_VERSION + '.json')} and its details file.")


if __name__ == "__main__":
    main()
