"""Prediction module for university submission."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

try:
    from utils import one_day_row
except ImportError:
    from .utils import one_day_row


def load_model(model_path: str | Path) -> Any:
    """Load and return the fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Construct one-day lag row, apply fitted preprocessing, and predict one non-negative scalar.

    history_df contains all observations strictly before the prediction date,
    including past chlorophyll. day_features_df contains exactly the current
    day's exogenous variables and date, but never current-day chlorophyll.
    """
    # Do not modify inputs in place
    hist = history_df.copy()
    day = day_features_df.copy()

    # 1. Build today's lagged features row
    lag_spec = model.get("lag_spec")
    row = one_day_row(hist, day, lag_spec=lag_spec)

    # 2. Select only the features the model was trained on
    feature_names = model["feature_names"]
    row_features = row[feature_names]

    # 3. Transform using the fitted scaler
    scaler = model["scaler"]
    row_scaled = scaler.transform(row_features)

    # 4. Predict with fitted estimator
    estimator = model["estimator"]
    pred = estimator.predict(row_scaled)

    # 5. Chlorophyll concentration is strictly non-negative
    pred = np.maximum(0.0, pred)

    return np.asarray(pred, dtype=float).reshape(1)

    # 1. Build today's lagged features row
    lag_spec = model.get("lag_spec")
    row = one_day_row(hist, day, lag_spec=lag_spec)

    # 2. Select only the features the model was trained on
    feature_names = model["feature_names"]
    row_features = row[feature_names]

    # 3. Transform using the fitted scaler
    scaler = model["scaler"]
    row_scaled = scaler.transform(row_features)

    # 4. Predict with fitted estimator
    estimator = model["estimator"]
    pred = estimator.predict(row_scaled)

    # 5. Chlorophyll concentration is strictly non-negative
    pred = np.maximum(0.0, pred)

    return np.asarray(pred, dtype=float).reshape(1)

