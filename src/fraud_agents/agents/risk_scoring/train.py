"""Train the fraud-scoring model (Part 1 · Step 4).

Run from the project folder:  python -m fraud_agents.agents.risk_scoring.train
Reads  data/raw/fraudTrain.csv and data/raw/fraudTest.csv
Writes models/risk_model_v1.json (the model) and models/risk_model_v1.meta.json (its details)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_agents.agents.risk_scoring.features import FEATURES, make_features

MODEL_VERSION = "risk_model_v1"
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
COLUMNS = list(RENAME) + ["category", "city_pop", "is_fraud"]


def load(path: Path) -> pd.DataFrame:
    """Read only the columns we use. is_fraud is the answer key, kept apart from the features."""
    df = pd.read_csv(path, usecols=COLUMNS).rename(columns=RENAME)
    # Assumption (Step 3): the dataset's times have no time zone; we treat them as UTC.
    df["occurred_at"] = pd.to_datetime(df["occurred_at"]).dt.tz_localize("UTC")
    return df.sort_values("occurred_at").reset_index(drop=True)


def report(name: str, answers: np.ndarray, scores: np.ndarray) -> dict:
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
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        flagged = scores >= t
        caught = int((flagged & (answers == 1)).sum())
        false_alarms = int((flagged & (answers == 0)).sum())
        missed = int(answers.sum()) - caught
        result["at_threshold"][str(t)] = {
            "caught": caught, "missed": missed, "false_alarms": false_alarms,
        }
        print(f"    {t}: caught {caught:,}, missed {missed:,}, false alarms {false_alarms:,}")
    return result


def main() -> None:
    print("Reading the dataset ...")
    train = load(DATA / "fraudTrain.csv")
    test = load(DATA / "fraudTest.csv")
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

    # The real test: fraudTest.csv, which the model has never seen.
    test_scores = model.predict_proba(make_features(test, categories))[:, 1]
    test_result = report("Test set (never seen)", test["is_fraud"].to_numpy(), test_scores)

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
        "tested_on": "data/raw/fraudTest.csv",
        "test": test_result,
        "xgboost_version": xgb.__version__,
    }
    (MODELS / f"{MODEL_VERSION}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {MODELS / (MODEL_VERSION + '.json')} and its details file.")


if __name__ == "__main__":
    main()
