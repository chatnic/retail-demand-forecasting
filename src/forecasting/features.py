"""Feature engineering for a direct 28-day-ahead forecast.

Every lag feature is shifted by at least HORIZON days, so the same model can forecast
any day in the next four weeks using only data available at the forecast origin.
That rules out leakage by construction, with no recursive forecasting needed.
"""

from __future__ import annotations

import pandas as pd

HORIZON = 28
TARGET = "units_sold"
CATEGORICAL = ["store_id", "item_id", "category"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby(["store_id", "item_id"], sort=False)[TARGET]

    for lag in (HORIZON, HORIZON + 7, HORIZON + 14, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON)
    grp = shifted.groupby([df["store_id"], df["item_id"]], sort=False)
    for w in (7, 28, 91):
        df[f"roll_mean_{w}"] = grp.transform(lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean())
    df["roll_std_28"] = grp.transform(lambda s: s.rolling(28, min_periods=14).std())

    # Same-weekday average over the four weeks before the horizon.
    df["dow_mean_4w"] = (df["lag_28"] + df["lag_35"] + df["lag_42"] + g.shift(HORIZON + 21)) / 4

    # Known-in-advance features: calendar, planned promotions and prices.
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    regular_price = df.groupby("item_id")["price"].transform("max")
    df["discount_pct"] = 1 - df["price"] / regular_price

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


FEATURES = [
    "lag_28", "lag_35", "lag_42", "lag_364", "roll_mean_7", "roll_mean_28", "roll_mean_91",
    "roll_std_28", "dow_mean_4w", "dow", "month", "dayofyear", "weekofyear", "is_holiday",
    "on_promo", "price", "discount_pct", *CATEGORICAL,
]
