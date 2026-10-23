"""The card-history features (feedback #4): training and the live agent must compute them
the same way, from the card's STRICTLY EARLIER payments only. No database needed: the
live definition (the same as HISTORY_SQL) is written out here by brute force."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fraud_agents.agents.risk_scoring.features import HISTORY, history_from_frame, history_values

T0 = datetime(2020, 6, 21, 12, 0, tzinfo=timezone.utc)


def payments() -> pd.DataFrame:
    rows = [  # card, minutes after T0, amount, merchant, category
        ("a", 0, 10.0, "m1", "grocery"),
        ("a", 30, 20.0, "m1", "grocery"),
        ("a", 60, 300.0, "m2", "shopping"),     # exactly 1 h after the first: inside [t - 1 h, t)
        ("a", 60, 5.0, "m3", "shopping"),       # same second as the one above: they don't see each other
        ("b", 61, 50.0, "m1", "grocery"),       # another card: never mixed in
        ("a", 61 + 24 * 60, 900.0, "m2", "shopping"),   # 24 h later: the first payments fall out
        ("a", 62 + 24 * 60, 15.0, "m4", "travel"),
    ]
    return pd.DataFrame([{"card_id": c, "occurred_at": T0 + timedelta(minutes=m), "amount": a,
                          "merchant": me, "category": ca} for c, m, a, me, ca in rows])


def live_definition(df: pd.DataFrame, i: int) -> dict:
    """What the live agent computes (HISTORY_SQL + history_values), done by hand."""
    row = df.loc[i]
    t = row.occurred_at
    past = df[(df.card_id == row.card_id) & (df.occurred_at < t)]
    in_cat = past[past.category == row.category]
    return history_values(
        row.amount, len(past), past.amount.mean() if len(past) else None,
        past.amount.max() if len(past) else None,
        int((past.occurred_at >= t - timedelta(hours=1)).sum()),
        past.amount[past.occurred_at >= t - timedelta(hours=24)].sum(),
        int((past.merchant == row.merchant).sum()), len(in_cat),
        in_cat.amount.mean() if len(in_cat) else None)


def test_training_features_match_the_live_definition_row_by_row():
    df = payments()
    built = history_from_frame(df)
    for i in df.index:
        expected = live_definition(df, i)
        for name in HISTORY:
            got, want = built.loc[i, name], expected[name]
            assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want), (i, name, got, want)


def test_the_first_payment_has_no_history_and_ties_do_not_see_each_other():
    built = history_from_frame(payments())
    assert built.loc[0, "card_history_n"] == 0 and np.isnan(built.loc[0, "amount_vs_card_avg"])
    assert built.loc[2, "card_history_n"] == built.loc[3, "card_history_n"] == 2
    assert built.loc[4, "card_history_n"] == 0                      # card b's first payment
    assert built.loc[5, "card_payments_1h"] == 0 and built.loc[5, "new_merchant"] == 0


def test_a_payment_far_above_the_cards_normal_spend_stands_out():
    built = history_from_frame(payments())
    assert built.loc[5, "amount_vs_card_max"] == pytest.approx(900 / 300)
    assert built.loc[5, "amount_vs_card_avg"] == pytest.approx(900 / ((10 + 20 + 300 + 5) / 4))


def test_the_real_sql_gives_the_same_features():
    """Runs HISTORY_SQL itself on the synthetic rows (in a temporary table that hides the real
    one for this connection only). Needs PostgreSQL: skipped without it."""
    import os
    import psycopg
    from fraud_agents.agents.risk_scoring.agent import CardHistory
    from fraud_agents.shared import db
    try:
        conn = db.connect(connect_timeout=2)
    except psycopg.OperationalError:
        if os.environ.get("CI_POSTGRES"):
            pytest.fail("PostgreSQL should be running")
        pytest.skip("PostgreSQL is not running (start it with: docker compose up -d)")
    with conn:
        conn.execute("CREATE TEMP TABLE card_payments (card_id TEXT, occurred_at TIMESTAMPTZ, "
                     "amount NUMERIC(12, 2), category TEXT, merchant TEXT)")
        df = payments()
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO card_payments VALUES (%s, %s, %s, %s, %s)",
                            [(r.card_id, r.occurred_at, r.amount, r.category, r.merchant) for r in df.itertuples()])
        live = CardHistory(conn=conn)
        built = history_from_frame(df)
        for r in df.itertuples():
            got = live(r.card_id, r.occurred_at, r.merchant, r.category, r.amount)
            for name in HISTORY:
                a, b = got[name], built.loc[r.Index, name]
                assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), (r.Index, name, a, b)
