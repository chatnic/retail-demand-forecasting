"""Forecasting models: two baselines and a global gradient-boosting model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecasting.features import CATEGORICAL, FEATURES, TARGET


class SeasonalNaive:
    """Same weekday, four weeks earlier (the latest value known at a 28-day horizon)."""

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["lag_28"].fillna(test["roll_mean_28"]).fillna(0).to_numpy()


class WeekdayAverage:
    """Average of the same weekday over the four weeks before the horizon."""

    name = "weekday_average"

    def fit(self, train: pd.DataFrame) -> "WeekdayAverage":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["dow_mean_4w"].fillna(test["roll_mean_28"]).fillna(0).to_numpy()


class GlobalGBM:
    """One gradient-boosting model across every store-item series, with Poisson loss."""

    name = "gradient_boosting"

    def __init__(self, seed: int = 42):
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.06,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=100,
            l2_regularization=1.0,
            categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )

    def fit(self, train: pd.DataFrame) -> "GlobalGBM":
        train = train.dropna(subset=["lag_42"])
        self.model.fit(train[FEATURES], train[TARGET])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(test[FEATURES]), 0, None)


def all_models() -> list:
    return [SeasonalNaive(), WeekdayAverage(), GlobalGBM()]
