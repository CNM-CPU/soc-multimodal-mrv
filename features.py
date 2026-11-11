"""
features.py
-----------
Feature engineering: collapse long-format sensor data into
per-farm feature vectors for model training.
"""

from __future__ import annotations

import pandas as pd


def aggregate_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sensor time series per (farm_id, variable, depth).

    Returns one row per farm with columns like:
        moisture_5_mean, moisture_5_std, temperature_15_mean, ...
    """
    df = df.dropna(subset=["value"])
    wide = (
        df.groupby(["farm_id", "variable", "depth"])["value"]
        .agg(["mean", "std", "min", "max"])
        .unstack(["variable", "depth"])
    )
    wide.columns = [f"{var}_{depth}_{stat}" for stat, var, depth in wide.columns]
    return wide.reset_index()


def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns (crop, tillage)."""
    return pd.get_dummies(df, columns=cols, drop_first=False)
