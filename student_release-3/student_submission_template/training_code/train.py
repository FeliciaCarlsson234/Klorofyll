"""Script to reproduce model training and save best_model/model.pkl."""

from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

best_model_dir = Path(__file__).resolve().parent.parent / "best_model"
if str(best_model_dir) not in sys.path:
    sys.path.insert(0, str(best_model_dir))

from utils import (DEFAULT_LAGS, TARGET, DATE, clean_outliers, remove_noise_features,
remove_correlated_features, build_lagged_frame)

def train_and_save(data_path: Path, output_path: Path) -> None:
    """Train the model on training data and serialize artifact dictionary."""
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(data_path, sheet_name="daily_data")
    else:
        raw = pd.read_csv(data_path)
    raw[DATE] = pd.to_datetime(raw[DATE])
    raw = raw.sort_values(DATE).reset_index(drop=True)

    #training period (2015-01-15 to 2017-12-31)
    train_start = '2015-01-15'
    train_end = '2017-12-31'

    #create lagged variables
    df_lagged = build_lagged_frame(raw)
    train_dates = (df_lagged[DATE] >= train_start) & (df_lagged[DATE] <= train_end)

    #data cleaning
    df_clean = clean_outliers(remove_correlated_features(remove_noise_features(df_lagged, TARGET, train_dates),
        TARGET, train_dates),TARGET, train_dates)

    df_clean[DATE] = pd.to_datetime(df_clean[DATE])
    df_clean = df_clean.sort_values(DATE).set_index(DATE)

    Y = df_clean[TARGET]
    X = df_clean.drop(columns=[TARGET])
    feature_names = X.columns.tolist()

    X_train = X.loc[train_start:train_end]
    Y_train = Y.loc[train_start:train_end]

    #standardscaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #train model (Ridge with alpha=50.0, chosen by notebook validation)
    model = Ridge(alpha=50.0)
    model.fit(X_train_scaled, Y_train)

    #save model-object
    artifact = {"estimator": model,"scaler": scaler,"feature_names": feature_names,"lag_spec": DEFAULT_LAGS,}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Modell tränad på {len(X_train)} rader med {len(feature_names)} features.")
    print(f"Sparad till {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train_and_save(args.data, args.output)


if __name__ == "__main__":
    main()
