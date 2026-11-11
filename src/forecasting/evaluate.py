"""Forecast accuracy metrics and inventory impact simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|error| / sum(actual)."""
    return float(np.abs(y - yhat).sum() / max(np.sum(y), 1e-9))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Positive = over-forecasting."""
    return float((yhat - y).sum() / max(np.sum(y), 1e-9))


def fill_rate(y: np.ndarray, stock: np.ndarray) -> float:
    return float(np.minimum(y, stock).sum() / max(np.sum(y), 1e-9))


def stock_for_service_level(y: np.ndarray, yhat: np.ndarray, sigma: np.ndarray,
                            target: float = 0.95) -> tuple[float, np.ndarray]:
    """Find the safety-stock multiplier z so that stock = yhat + z * sigma hits the target fill rate.

    sigma is a per-series estimate of forecast error (from the training period), so items
    the model is less sure about get more buffer — the way a planner would set safety stock.
    """
    lo, hi = -2.0, 8.0
    for _ in range(60):
        z = (lo + hi) / 2
        stock = np.maximum(yhat + z * sigma, 0)
        if fill_rate(y, stock) < target:
            lo = z
        else:
            hi = z
    z = hi
    return z, np.maximum(yhat + z * sigma, 0)


def inventory_impact(frame: pd.DataFrame, models: list[str], target: float = 0.95) -> pd.DataFrame:
    """Excess inventory each model needs to reach the same fill rate.

    `frame` needs columns: units_sold, unit_cost, sigma (recent demand volatility per series,
    known at the forecast origin) and pred_<model>. Every model gets the same volatility
    signal; a more accurate forecast reaches the target with a smaller multiplier.
    """
    rows = []
    y = frame["units_sold"].to_numpy()
    sigma = frame["sigma"].fillna(frame["sigma"].median()).to_numpy()
    for m in models:
        z, stock = stock_for_service_level(y, frame[f"pred_{m}"].to_numpy(), sigma, target)
        excess = np.maximum(stock - y, 0)
        rows.append({
            "model": m,
            "safety_multiplier_z": z,
            "fill_rate": fill_rate(y, stock),
            "excess_units": float(excess.sum()),
            "excess_value_at_cost": float((excess * frame["unit_cost"]).sum()),
        })
    return pd.DataFrame(rows)
