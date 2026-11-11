"""Generate synthetic daily store-item sales for demand forecasting.

The simulator combines the patterns that make real retail forecasting hard:
weekly and annual seasonality, item-level trends, store size effects, promotions
with price elasticity, holiday spikes, and intermittent demand for slow movers.

Usage:
    python data/generate_data.py --stores 10 --items 50 --days 730 --out data/raw/sales.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "household", "personal_care", "beverages", "snacks"]
HOLIDAYS = {  # month-day: demand multiplier
    "01-01": 0.7, "02-14": 1.25, "05-27": 1.3, "07-04": 1.45, "09-02": 1.2,
    "11-28": 1.6, "11-29": 1.5, "12-23": 1.5, "12-24": 1.7, "12-25": 0.5, "12-31": 1.35,
}


def generate(stores: int = 10, items: int = 50, days: int = 730, seed: int = 42,
             start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)

    # Store and item attributes
    store_size = rng.lognormal(0, 0.35, stores)
    item_base = rng.lognormal(1.6, 0.9, items)          # mean daily units at an average store
    item_cat = rng.choice(CATEGORIES, items)
    item_price = np.round(rng.uniform(1.5, 25, items), 2)
    item_trend = rng.normal(0, 0.25, items)              # annual growth
    elasticity = rng.uniform(1.2, 3.0, items)            # promo price elasticity
    weekly_amp = rng.uniform(0.1, 0.4, items)
    annual_amp = np.where(np.isin(item_cat, ["beverages", "snacks"]), rng.uniform(0.15, 0.35, items),
                          rng.uniform(0.0, 0.12, items))

    dow = dates.dayofweek.values
    weekly = 1 + np.where(dow >= 5, 1, np.where(dow == 4, 0.5, -0.25))[None, :] * weekly_amp[:, None]
    annual = 1 + annual_amp[:, None] * np.sin(2 * np.pi * (t[None, :] - 120) / 365.25)
    trend = np.exp(item_trend[:, None] * t[None, :] / 365.25)
    md = dates.strftime("%m-%d")
    holiday = np.array([HOLIDAYS.get(d, 1.0) for d in md])

    rows = []
    for s in range(stores):
        # Promotions: ~8% of item-days, in multi-day blocks.
        promo = np.zeros((items, days), dtype=bool)
        for i in range(items):
            for _ in range(rng.poisson(days / 365 * 6)):
                st = rng.integers(0, days - 7)
                promo[i, st:st + rng.integers(3, 8)] = True
        discount = np.where(promo, rng.choice([0.1, 0.2, 0.3], size=(items, days)), 0.0)
        promo_lift = (1 - discount) ** (-elasticity[:, None])

        mu = (item_base[:, None] * store_size[s] * weekly * annual * trend * holiday[None, :] * promo_lift)
        noise = rng.gamma(shape=6, scale=1 / 6, size=mu.shape)   # multiplicative noise
        units = rng.poisson(mu * noise)
        price = np.round(item_price[:, None] * (1 - discount), 2)

        for i in range(items):
            rows.append(pd.DataFrame({
                "date": dates,
                "store_id": f"S{s + 1:02d}",
                "item_id": f"I{i + 1:03d}",
                "category": item_cat[i],
                "price": price[i],
                "on_promo": promo[i].astype(int),
                "units_sold": units[i],
            }))
    df = pd.concat(rows, ignore_index=True)
    df["is_holiday"] = df["date"].dt.strftime("%m-%d").isin(HOLIDAYS).astype(int)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stores", type=int, default=10)
    p.add_argument("--items", type=int, default=50)
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw/sales.csv"))
    a = p.parse_args()
    df = generate(a.stores, a.items, a.days, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows ({df.store_id.nunique()} stores x {df.item_id.nunique()} items) to {a.out}")


if __name__ == "__main__":
    main()
