"""Auxiliary prediction and data processing functions."""

from __future__ import annotations
import numpy as np
import pandas as pd
from build_lagged_features import DATE


def clean_outliers(df: pd.DataFrame, target: str, training_data: pd.Series, k: int = 6, window: int = 10) -> pd.DataFrame:
    """Clean outliers using rolling median and MAD on the training window."""
    df_cleaned = df.copy()
    train_mad = np.abs(df_cleaned.loc[training_data, target] - df_cleaned.loc[training_data, target].median()).median()
    rolling_median = df_cleaned[target].rolling(window=window, center=True, min_periods=1).median()
    outlier = (np.abs(df_cleaned[target] - rolling_median) > k * train_mad) & training_data
    df_cleaned.loc[outlier, target] = rolling_median[outlier]
    return df_cleaned


def remove_noise_features(df: pd.DataFrame, target: str, training_data: pd.Series, min_corr: float = 0.1) -> pd.DataFrame:
    """Remove features with absolute correlation to target below min_corr."""
    df_no_noise = df.copy()
    corr_matrix = df_no_noise.loc[training_data].corr(numeric_only=True).abs()
    noise_features = corr_matrix.index[corr_matrix[target] < min_corr].tolist()
    
    if target in noise_features:
        noise_features.remove(target)
    return df_no_noise.drop(columns=noise_features)


def remove_correlated_features(df: pd.DataFrame, target: str, training_data: pd.Series, max_corr: float = 0.9) -> pd.DataFrame:
    """Remove multicollinear features with pairwise correlation above max_corr."""
    df_no_correlation = df.copy()
    feature_cols = [c for c in df_no_correlation.columns if c not in [DATE, target]]
    corr_matrix = df_no_correlation.loc[training_data, feature_cols].corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features = [col for col in upper.columns if any(upper[col] > max_corr)]
    
    if target in correlated_features:
        correlated_features.remove(target)
    return df_no_correlation.drop(columns=correlated_features)
