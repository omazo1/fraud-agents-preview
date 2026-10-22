"""Turn payments into the numbers the fraud model reads.

Training (train.py) and the Risk Scoring agent (Step 5) both use make_features(),
so the model always sees features built in exactly the same way.
"""

import numpy as np
import pandas as pd

FEATURES = ["amount", "category", "hour", "day_of_week", "distance_km", "city_pop"]

EARTH_RADIUS_KM = 6371.0


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
    merchant_long, home_lat, home_long, city_pop (the same names as our event form).
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
    out["day_of_week"] = when.dt.dayofweek
    out["distance_km"] = distance_km(
        payments["home_lat"], payments["home_long"],
        payments["merchant_lat"], payments["merchant_long"],
    )
    out["city_pop"] = payments["city_pop"].astype(float)
    return out[FEATURES]
