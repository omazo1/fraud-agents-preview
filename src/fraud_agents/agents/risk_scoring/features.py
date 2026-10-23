"""Turn payments into the numbers the fraud model reads.

Training (train.py) and the Risk Scoring agent both use make_features(), so the model
always sees features built in exactly the same way.

Two kinds of features:
- about the payment itself: amount, shop type, time of day, day of week, distance from
  home, city size;
- about the CARD'S OWN PAST (feedback #4, model v2): how this amount compares with what the
  card normally spends, how busy the card has been in the last hour and day, and whether
  this shop (or shop type) is new for the card.

"Past" always means the card's payments STRICTLY BEFORE this one (occurred_at < this
payment's time), never later ones, so the model can't peek at the future:
- training computes them from the dataset with history_from_frame();
- the live agent asks PostgreSQL's card_payments table for the same numbers (HISTORY_SQL)
  and turns them into features with history_values(). A test checks the two agree.
"""

import numpy as np
import pandas as pd

BASE = ["amount", "category", "hour", "day_of_week", "distance_km", "city_pop"]
HISTORY = [
    "card_history_n",               # how many earlier payments the card has
    "amount_vs_card_avg",           # this amount / the card's average amount so far
    "amount_vs_card_max",           # this amount / the card's largest amount so far
    "card_payments_1h",             # the card's payments in the hour before
    "card_spend_24h",               # what the card spent in the 24 hours before
    "new_merchant",                 # 1 = the card has never paid this shop before
    "card_category_n",              # the card's earlier payments in this shop type
    "amount_vs_card_category_avg",  # this amount / the card's average in this shop type
]
FEATURES = BASE + HISTORY

EARTH_RADIUS_KM = 6371.0

# The card's past, as PostgreSQL computes it for the live agent (one indexed query).
HISTORY_SQL = """
SELECT count(*), avg(amount), max(amount),
       count(*) FILTER (WHERE occurred_at >= %(t)s - interval '1 hour'),
       coalesce(sum(amount) FILTER (WHERE occurred_at >= %(t)s - interval '24 hours'), 0),
       count(*) FILTER (WHERE merchant = %(merchant)s),
       count(*) FILTER (WHERE category = %(category)s),
       avg(amount) FILTER (WHERE category = %(category)s)
FROM card_payments
WHERE card_id = %(card)s AND occurred_at < %(t)s
"""


def ratio(amount, base):
    """amount / base, or NaN when there is nothing to compare with (the model handles NaN)."""
    return np.nan if base is None or pd.isna(base) or float(base) == 0 else float(amount) / float(base)


def history_values(amount, n, avg, biggest, n_1h, spend_24h, merchant_n, category_n, category_avg) -> dict:
    """The history features of one payment, from the card's past numbers (HISTORY_SQL's row)."""
    return {
        "card_history_n": int(n),
        "amount_vs_card_avg": ratio(amount, avg),
        "amount_vs_card_max": ratio(amount, biggest),
        "card_payments_1h": int(n_1h),
        "card_spend_24h": float(spend_24h or 0),
        "new_merchant": int(merchant_n == 0),
        "card_category_n": int(category_n),
        "amount_vs_card_category_avg": ratio(amount, category_avg),
    }


def _strictly_before(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """For each row: count, sum and max of the amounts of the EARLIER rows with the same `by`
    values. Rows at the same second don't count each other (like occurred_at < t in SQL)."""
    key = by + ["occurred_at"]
    per = df.groupby(key, sort=False)["amount"].agg(["count", "sum", "max"])   # one row per second
    groups = per.groupby(level=list(range(len(by))), sort=False)
    running_max = groups["max"].cummax()                                       # up to this second
    prior = pd.DataFrame({
        "n": groups["count"].cumsum() - per["count"],
        "sum": groups["sum"].cumsum() - per["sum"],
        "max": running_max.groupby(level=list(range(len(by))), sort=False).shift(1),   # before it
    })
    return df[key].merge(prior, left_on=key, right_index=True, how="left").set_index(df.index)


def history_from_frame(df: pd.DataFrame, card: str = "card_id") -> pd.DataFrame:
    """The HISTORY features for every row of `df` (columns: card, occurred_at, amount,
    merchant, category), each from the same card's strictly earlier rows in `df`."""
    if not df.index.is_unique:
        raise ValueError("history_from_frame needs a unique row index")   # values are placed by index
    df = df.sort_values("occurred_at", kind="stable")
    out = pd.DataFrame(index=df.index)
    card_past = _strictly_before(df, [card])
    out["card_history_n"] = card_past["n"].astype(int)
    out["amount_vs_card_avg"] = df["amount"] / (card_past["sum"] / card_past["n"].replace(0, np.nan))
    out["amount_vs_card_max"] = df["amount"] / card_past["max"]

    # Time windows [t - 1 h, t) and [t - 24 h, t): earlier rows only, like the SQL.
    timed = df.set_index("occurred_at")
    order = df.sort_values(card, kind="stable").index             # groupby's row order
    for name, window, how in (("card_payments_1h", "1h", "count"), ("card_spend_24h", "24h", "sum")):
        rolled = getattr(timed.groupby(card)["amount"].rolling(window, closed="left"), how)()
        out.loc[order, name] = rolled.fillna(0).to_numpy()

    merchant_past = _strictly_before(df, [card, "merchant"])
    out["new_merchant"] = (merchant_past["n"] == 0).astype(int)
    category_past = _strictly_before(df, [card, "category"])
    out["card_category_n"] = category_past["n"].astype(int)
    out["amount_vs_card_category_avg"] = df["amount"] / (
        category_past["sum"] / category_past["n"].replace(0, np.nan))
    out["card_payments_1h"] = out["card_payments_1h"].astype(int)
    return out[HISTORY].sort_index()


def distance_km(lat1, lon1, lat2, lon2):
    """Straight-line distance on the globe (haversine formula), in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def make_features(payments: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Build the model's features.

    `payments` needs the columns: occurred_at, amount, category, merchant_lat,
    merchant_long, home_lat, home_long, city_pop (the same names as our event form), plus
    the HISTORY columns (from history_from_frame or history_values).
    `categories` is the fixed list of shop types saved with the model.
    """
    unknown = set(payments["category"]) - set(categories)
    if unknown:
        # A shop type the model never saw would silently score lower, so refuse it.
        raise ValueError(f"Unknown shop category: {sorted(unknown)}")

    # Times are in UTC everywhere (Step 3 assumption), in training and in the live events.
    when = pd.to_datetime(payments["occurred_at"], utc=True)
    out = pd.DataFrame(index=payments.index)
    out["amount"] = payments["amount"].astype(float)
    out["category"] = pd.Categorical(payments["category"], categories=categories)
    out["hour"] = when.dt.hour
    out["day_of_week"] = when.dt.day_of_week   # same values; "dayofweek" is deprecated in pandas 3
    out["distance_km"] = distance_km(
        payments["home_lat"], payments["home_long"],
        payments["merchant_lat"], payments["merchant_long"],
    )
    out["city_pop"] = payments["city_pop"].astype(float)
    for name in HISTORY:
        out[name] = payments[name].astype(float)
    return out[FEATURES]
