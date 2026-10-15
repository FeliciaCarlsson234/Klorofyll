"""ARX Ridge model for one-day-ahead chlorophyll-a forecasting."""
 
from __future__ import annotations
 
from pathlib import Path
import pickle
from typing import Any
 
import numpy as np
import pandas as pd
 
from utils import lagged_row
 
 
def load_model(model_path: str | Path) -> Any:
    """Load and return the submitted fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)
 
 
def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Return the one-day-ahead chlorophyll-a prediction as a NumPy array with shape (1,).
 
    history_df contains all observations strictly before the prediction date,
    including past chlorophyll. day_features_df contains exactly the current
    day's exogenous variables and date, but never current-day chlorophyll.
 
    Spikes in past chlorophyll are cleaned, the lagged vector is built
    explicitly, standardized with the saved training mean/std, and the linear
    prediction is clipped to be non-negative.
    """
    x = lagged_row(history_df, day_features_df, model["lags"], model["mad"])
    z = np.nan_to_num((x - np.asarray(model["mu"])) / np.asarray(model["sd"]))  # missing -> train mean
    y = model["intercept"] + z @ np.asarray(model["coef"])
    return np.asarray([max(float(y), 0.0)], dtype=float)  # chlorophyll is non-negative
 