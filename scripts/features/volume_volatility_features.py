"""
volume_volatility_features.py

Generates volume- and volatility-based features from the base SEP OHLCV dataset.
This script builds on the gold-standard price and volume data to compute indicators
of crowd activity (volume surges), liquidity, and risk (volatility) over short
and medium time frames.

Features engineered include:
- Volume z-score (21-day)
- Dollar volume (close * volume)
- Volume % change from previous day
- Rolling return standard deviations (5d, 21d)
- Average True Range (ATR 14d)

Input:
    data/base/sep_base.parquet

Output:
    scripts/feature_matrices/<YYYY-MM-DD>_volume_volatility_features.parquet

To run:
    python scripts/features/volume_volatility_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
INPUT_PATH = Path("data/base/sep_base.parquet")
OUTPUT_DIR = Path("scripts/feature_matrices/")
TODAY = datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = OUTPUT_DIR / f"{TODAY}_volume_volatility_features.parquet"

# Load and sort
df = pd.read_parquet(INPUT_PATH)
df = df.sort_values(["ticker", "date"])

# Dollar volume
df["dollar_volume"] = df["close"] * df["volume"]

# Volume z-score (21d rolling)
df["vol_zscore_21d"] = (
    df.groupby("ticker")["volume"]
    .transform(lambda x: (x - x.rolling(21).mean()) / x.rolling(21).std())
)

# Volume % change from previous day
df["volume_pct_change_1d"] = df.groupby("ticker")["volume"].pct_change()

# Daily returns for volatility calcs
df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

# Rolling return standard deviations (volatility)
df["ret_std_5d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(5).std())
df["ret_std_21d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(21).std())

# True range: high - low
df["true_range"] = df["high"] - df["low"]

# ATR (Average True Range) over 14 days
df["atr_14d"] = df.groupby("ticker")["true_range"].transform(lambda x: x.rolling(14).mean())

# Final cleanup
df = df.dropna()

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Volume/volatility features saved to {OUTPUT_PATH}")
