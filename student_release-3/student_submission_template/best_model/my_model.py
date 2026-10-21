"""Prediction module for university submission."""

from __future__ import annotations
from pathlib import Path
import pickle
from typing import Any
import numpy as np
import pandas as pd


def load_model(model_path: str | Path) -> Any:
    """Load and return the fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Construct one-day lag row, apply fitted preprocessing, and predict one non-negative scalar.

    history_df contains all observations strictly before the prediction date,
    including past chlorophyll. day_features_df contains exactly the current
    day's exogenous variables and date, but never current-day chlorophyll."""
   
    hist = history_df.copy()
    day = day_features_df.copy()

    #build today's lagged features row
    lag_spec = model.get("lag_spec")
    row = one_day_row(hist, day, lag_spec=lag_spec)

    #select only the features the model was trained on
    feature_names = model["feature_names"]
    row_features = row[feature_names]

    #transform using the fitted scaler
    scaler = model["scaler"]
    row_scaled = scaler.transform(row_features)

    #predict with fitted estimator
    estimator = model["estimator"]
    pred = estimator.predict(row_scaled)

    #chlorophyll concentration is strictly non-negative
    pred = np.maximum(0.0, pred)

    return np.asarray(pred, dtype=float).reshape(1)

    #building today's lagged features row
    lag_spec = model.get("lag_spec")
    row = one_day_row(hist, day, lag_spec=lag_spec)

    #select only the features the model was trained on
    feature_names = model["feature_names"]
    row_features = row[feature_names]

    #transform using the fitted scaler
    scaler = model["scaler"]
    row_scaled = scaler.transform(row_features)

    #predict with fitted estimator
    estimator = model["estimator"]
    pred = estimator.predict(row_scaled)

    #chlorophyll concentration is strictly non-negative
    pred = np.maximum(0.0, pred)

    return np.asarray(pred, dtype=float).reshape(1)
