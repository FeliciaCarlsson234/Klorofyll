"""explicit feature construction shared by training and prediction.
this file is stored twice: best_model/aux.py (used at prediction time)
and training_code/features.py (used by train.py). the files are identical,
so training and prediction build the features with the same code.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"
EXOG = ["sst_c", "par_umol_m2_s", "nitrate_umol_l", "wind_speed_m_s","upwelling_index", "mixed_layer_depth_m", "salinity_psu",
"current_speed_m_s", "river_discharge_index", "cloud_fraction","surface_pressure_hpa", "turbidity_ntu",]
WARMUP = 14 
MIN_CHL = 1e-3     


def clean_chlorophyll(y, k: float, w: int):
    """spike filter for the chlorophyll
    day t is flagged when it is more than k times larger (or k times smaller)
    than the median of the previous w *cleaned* values. Flagged values are
    replaced by that median. Only days before t are used, so the filter can be
    applied identically to a growing history at prediction time.

    Returns (cleaned values, boolean flags).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = y.tolist()
    flag = [False] * n
    for t in range(n):
        if t < 3:
            if not np.isfinite(out[t]):
                out[t] = out[t - 1] if t > 0 else 1.0
            continue
        window = sorted(out[max(0, t - w):t])
        m = len(window)
        med = window[m // 2] if m % 2 else 0.5 * (window[m // 2 - 1] + window[m // 2])
        v = out[t]
        if not np.isfinite(v):
            out[t] = med
            flag[t] = True
        elif v > k * med or v < med / k:
            out[t] = med
            flag[t] = True
    return np.asarray(out, dtype=float), np.asarray(flag, dtype=bool)


def feature_names(p: int, q: int) -> list[str]:
    names = [f"{TARGET}__loglag_{i}" for i in range(1, p + 1)]
    for col in EXOG:
        names += [f"{col}__lag_{j}" for j in range(0, q + 1)]
    return names + ["annual_sin", "annual_cos"]


def feature_matrix(dates, chl_clean, exog: pd.DataFrame, p: int, q: int) -> np.ndarray: #phivektorn byggs här
    """regressor vectors phi(t) for every row (rows without enough past are NaN).
    - chlorophyll enters only through lags 1..p (never lag 0), as log of the cleaned value;
    - every exogenous variable enters through lags 0..q;
    - the annual cycle enters through sin/cos of the day of year.
    """
    n = len(dates)
    logy = np.log(np.maximum(np.asarray(chl_clean, dtype=float), MIN_CHL))
    cols = []
    for i in range(1, p + 1): #får inte ha med 0 (samma dag)
        lagged = np.full(n, np.nan)
        lagged[i:] = logy[:-i]
        cols.append(lagged)
    for col in EXOG:
        v = exog[col].to_numpy(dtype=float)
        for j in range(0, q + 1):
            lagged = np.full(n, np.nan)
            lagged[j:] = v[: n - j]
            cols.append(lagged)
    day = pd.to_datetime(pd.Series(dates)).dt.dayofyear.to_numpy()
    cols.append(np.sin(2 * np.pi * day / 365.25)) #tar hänsyn till skottår
    cols.append(np.cos(2 * np.pi * day / 365.25))
    return np.column_stack(cols)

def standardize(X, mean, scale):
    """explicit z-score with stored parameters."""
    return (np.asarray(X, dtype=float) - mean) / scale

def fit_standardizer(X):
    """mean and standard deviation (ddof=0) learned from the training matrix only."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale

def predict_one(model: dict, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> float:
    """one-day-ahead prediction (mg m-3) from history strictly before the day plus today's exogenous data."""
    cfg = model["config"]
    p, q = int(cfg["p"]), int(cfg["q"])

    hist = history_df.copy()
    hist[DATE] = pd.to_datetime(hist[DATE])
    hist = hist.sort_values(DATE).reset_index(drop=True)
    today = day_features_df.copy()
    today[DATE] = pd.to_datetime(today[DATE])
    if len(today) != 1:
        raise ValueError("day_features_df must have exactly one row")
    if len(hist) and hist[DATE].iloc[-1] >= today[DATE].iloc[0]:
        hist = hist[hist[DATE] < today[DATE].iloc[0]].reset_index(drop=True)

    #clean the observed chlorophyll history (kausal, explicit)
    chl_hist, _ = clean_chlorophyll(hist[TARGET].to_numpy(dtype=float), cfg["k"], cfg["w"])

    #stack history and today; chlorophyll of today is unknown and never used (lags start at 1)
    exog = pd.concat([hist[EXOG], today[EXOG]], ignore_index=True)
    exog = exog.ffill()
    exog = exog.fillna(model["train_exog_mean"])
    dates = pd.concat([hist[DATE], today[DATE]], ignore_index=True)
    chl_all = np.append(chl_hist, np.nan)

    #features for the last row only (needs the last max(p, q) + 1 rows)
    tail = max(p, q) + 1
    x = feature_matrix(dates.iloc[-tail:].reset_index(drop=True), chl_all[-tail:],
                       exog.iloc[-tail:].reset_index(drop=True), p, q)[-1]
    if not np.isfinite(x).all():
        raise ValueError("insufficient history to construct the requested lags")

    #stored normalisation, linear model, explicit output transformation
    z = standardize(x, model["mean"], model["scale"])
    log_pred = float(model["estimator"].predict(z.reshape(1, -1))[0])
    log_pred = min(max(log_pred, model["log_min"]), model["log_max"])   #keep extrapolation sane
    pred = float(np.exp(log_pred))
    if not np.isfinite(pred):
        pred = float(chl_hist[-1])
    return max(pred, 0.0)
