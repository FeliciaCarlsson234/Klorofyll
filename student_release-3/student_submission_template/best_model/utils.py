"""Auxiliary prediction functions imported by my_model.py."""
 
from __future__ import annotations
 
from typing import Any
 
import numpy as np
import pandas as pd
 
TARGET = "chlorophyll_a_mg_m3"
 
 
def clean_spikes(df: pd.DataFrame, mad: float, k: float = 6.0, window: int = 15) -> pd.DataFrame:
    """Replace chlorophyll retrieval errors by the rolling median (same rule as in training)."""
    df = df.copy()
    med = df[TARGET].rolling(window, center=True, min_periods=1).median()
    bad = (df[TARGET] - med).abs() > k * mad
    df.loc[bad, TARGET] = med[bad]
    return df
 
 
def lagged_row(history: pd.DataFrame, day: pd.DataFrame, lags: dict, mad: float) -> np.ndarray:
    """Build today's regressor vector, in the same column order as build_lagged_frame."""
    hist = clean_spikes(history.assign(date=pd.to_datetime(history["date"])).sort_values("date"), mad)
    df = pd.concat([hist, day.assign(date=pd.to_datetime(day["date"]))], ignore_index=True)
    feats = [float(df[col].iloc[-1 - lag]) for col, col_lags in lags.items() for lag in col_lags]
    doy = df["date"].iloc[-1].dayofyear
    feats += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.asarray(feats, dtype=float)
 
 
def constant_prediction(model: Any) -> float:
    """Extract the dummy model's constant prediction."""
    return float(model["constant_prediction"])
 