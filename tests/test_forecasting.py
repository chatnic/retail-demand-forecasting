import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate  # noqa: E402
from forecasting.evaluate import fill_rate, stock_for_service_level, wape  # noqa: E402
from forecasting.features import HORIZON, build_features  # noqa: E402


@pytest.fixture(scope="module")
def feats():
    return build_features(generate(stores=2, items=3, days=400, seed=1))


def test_generator_shape():
    df = generate(stores=2, items=3, days=50, seed=0)
    assert len(df) == 2 * 3 * 50
    assert (df["units_sold"] >= 0).all()


def test_lags_respect_horizon(feats):
    """lag_28 on day t must equal units on day t-28 for the same series — no leakage."""
    s = feats[(feats.store_id == "S01") & (feats.item_id == "I001")].reset_index(drop=True)
    assert s.loc[100, "lag_28"] == s.loc[100 - HORIZON, "units_sold"]
    assert s.loc[: HORIZON - 1, "lag_28"].isna().all()


def test_rolling_features_use_only_past(feats):
    s = feats[(feats.store_id == "S01") & (feats.item_id == "I001")].reset_index(drop=True)
    expected = s.loc[200 - HORIZON - 6: 200 - HORIZON, "units_sold"].mean()
    assert s.loc[200, "roll_mean_7"] == pytest.approx(expected)


def test_wape():
    assert wape(np.array([10, 10]), np.array([12, 8])) == pytest.approx(0.2)


def test_service_level_search_hits_target():
    rng = np.random.default_rng(0)
    y = rng.poisson(10, 5000)
    yhat = np.full(5000, 10.0)
    sigma = np.full(5000, 3.0)
    _, stock = stock_for_service_level(y, yhat, sigma, 0.95)
    assert fill_rate(y, stock) == pytest.approx(0.95, abs=0.005)
