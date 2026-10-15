"""Replace this template with the code used to train and save best_model/model.pkl.
 
The final script should reproduce raw-data loading, chronological splitting,
lag construction, preprocessing, model selection, final fitting, and model
serialization. sklearn.preprocessing tools may be fitted on input features
using training data only and then saved for prediction. Keep all supporting
Python modules in training_code/. Do not use library pipelines, composition
wrappers, imputers, feature selectors, or output transformers. Target
transformation and output post-processing must remain explicit.
"""
 
from __future__ import annotations
 
import argparse
import pickle
from pathlib import Path
 
import numpy as np
from sklearn.linear_model import Ridge
 
from build_lagged_features import DEFAULT_LAGS, TARGET, build_lagged_frame, read_table
 
 
def clean_spikes(df, mad, k=6.0, window=15):
    """Replace chlorophyll retrieval errors (> k robust std from rolling median) by that median."""
    df = df.copy()
    med = df[TARGET].rolling(window, center=True, min_periods=1).median()
    bad = (df[TARGET] - med).abs() > k * mad
    df.loc[bad, TARGET] = med[bad]
    return df
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
 
    raw = read_table(args.data)  # reads and sorts by date
    y = raw[TARGET]
    resid = y - y.rolling(15, center=True, min_periods=1).median()
    mad = float(1.4826 * np.median(np.abs(resid - resid.median())))
    lagged = build_lagged_frame(clean_spikes(raw, mad))
    X = lagged.drop(columns=["date", TARGET]).to_numpy(float)
    t = lagged[TARGET].to_numpy(float)
    truth = build_lagged_frame(raw)[TARGET].to_numpy(float)  # uncleaned, for honest validation
 
    # Chronological split: train 2015-2022, validate on 2023.
    tr = (lagged["date"] < "2023-01-01").to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0)
    best = None
    for alpha in [0.1, 1, 10, 100, 1000]:
        pred = np.maximum(Ridge(alpha=alpha).fit((X[tr] - mu) / sd, t[tr]).predict((X[~tr] - mu) / sd), 0)
        err = truth[~tr] - pred
        r2 = 1 - (err ** 2).sum() / ((truth[~tr] - truth[~tr].mean()) ** 2).sum()
        print(f"alpha={alpha:<6} R2={r2:.4f} RMSE={np.sqrt((err ** 2).mean()):.4f} MAE={np.abs(err).mean():.4f}")
        if best is None or r2 > best[1]:
            best = (alpha, r2)
    print(f"selected alpha={best[0]}")
 
    # Final fit on all released data; save only plain numbers (no sklearn objects).
    mu, sd = X.mean(0), X.std(0)
    final = Ridge(alpha=best[0]).fit((X - mu) / sd, t)
    model = {"lags": DEFAULT_LAGS, "mad": mad, "mu": mu.tolist(), "sd": sd.tolist(),
             "coef": final.coef_.tolist(), "intercept": float(final.intercept_)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle)
    print(f"saved {args.output}")
 
 
if __name__ == "__main__":
    main()
 
