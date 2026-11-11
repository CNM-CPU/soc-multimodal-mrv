"""
data_pipeline.py
----------------
Data ingestion, quality control, and synthetic data generation
for the SOC multimodal MRV project.

Implements the standard data schema from the proposal (§6.2):

    time, sensor_id, farm_id, depth, variable, value

and the QC rules from §6.3:
  - range checks
  - IQR outlier detection
  - gap filling (cubic interpolation, <= 48 h)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Standard schema
# ---------------------------------------------------------------------------

SENSOR_SCHEMA = ["time", "sensor_id", "farm_id", "depth", "variable", "value"]
VARIABLES = ["moisture", "temperature", "ph", "ec"]
DEPTHS_CM = [5, 15, 30]

# Plausible physical ranges for QC (adjust to your sensor specs)
VALID_RANGES = {
    "moisture":    (0.0, 60.0),    # volumetric %
    "temperature": (-20.0, 50.0),  # deg C
    "ph":          (3.0, 10.0),
    "ec":          (0.0, 10.0),    # dS/m
}


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_sensor_data(
    n_farms: int = 30,
    days: int = 365,
    step_hours: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic IoT sensor dataset that mimics the structure
    of real field data (30 farms, 3 depths, 4 variables, 6-hour cadence).

    Returns a long-format dataframe matching SENSOR_SCHEMA.
    """
    rng = np.random.default_rng(seed)
    steps_per_day = 24 // step_hours
    n_steps = days * steps_per_day
    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=n_steps, freq=f"{step_hours}h")

    rows = []
    for farm in range(n_farms):
        farm_bias = rng.normal(0, 2.0)
        for depth in DEPTHS_CM:
            sensor_id = f"F{farm:02d}_D{depth}"
            for var in VARIABLES:
                lo, hi = VALID_RANGES[var]
                mid = (lo + hi) / 2
                amp = (hi - lo) / 4

                # Seasonal cycle + noise + farm offset
                seasonal = amp * np.sin(2 * np.pi * np.arange(n_steps) / n_steps)
                noise = rng.normal(0, amp * 0.2, n_steps)
                values = mid + seasonal + noise + farm_bias
                values = np.clip(values, lo, hi)

                # Random sensor dropouts (~2 % of points)
                dropout = rng.random(n_steps) < 0.02
                values[dropout] = np.nan

                for t, v in zip(times, values):
                    rows.append(
                        {
                            "time": t,
                            "sensor_id": sensor_id,
                            "farm_id": farm,
                            "depth": depth,
                            "variable": var,
                            "value": v,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QC functions
# ---------------------------------------------------------------------------

def range_check(df: pd.DataFrame) -> pd.DataFrame:
    """Set physically impossible values to NaN."""
    mask = pd.Series(False, index=df.index)
    for var, (lo, hi) in VALID_RANGES.items():
        var_mask = (df["variable"] == var) & (
            (df["value"] < lo) | (df["value"] > hi)
        )
        mask |= var_mask
    df = df.copy()
    df.loc[mask, "value"] = np.nan
    return df


def iqr_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Flag IQR outliers per (sensor_id, variable) and set to NaN."""
    df = df.copy()
    grouped = df.groupby(["sensor_id", "variable"])["value"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (df["value"] < lower) | (df["value"] > upper)
    df.loc[mask, "value"] = np.nan
    return df


def fill_gaps(df: pd.DataFrame, max_gap_hours: int = 48) -> pd.DataFrame:
    """
    Cubic interpolation for gaps up to `max_gap_hours`.
    Larger gaps are left as NaN (do not fabricate long missing periods).
    """
    df = df.sort_values("time").copy()
    filled = []
    for (sensor_id, var), group in df.groupby(["sensor_id", "variable"]):
        group = group.set_index("time")
        # Only interpolate if the whole gap is short enough
        group["value"] = group["value"].interpolate(
            method="cubic", limit=max_gap_hours // 6
        )
        filled.append(group.reset_index().assign(sensor_id=sensor_id, variable=var))
    return pd.concat(filled, ignore_index=True)[SENSOR_SCHEMA]


def run_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Full QC chain: range -> IQR -> gap fill."""
    df = range_check(df)
    df = iqr_outliers(df)
    df = fill_gaps(df)
    return df


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic sensor data...")
    raw = generate_synthetic_sensor_data(n_farms=5, days=30)
    print(f"  raw: {len(raw)} rows, "
          f"{raw['value'].isna().sum()} missing values")

    clean = run_qc(raw)
    print(f"  after QC: {len(clean)} rows, "
          f"{clean['value'].isna().sum()} missing values")
    print(clean.head())
