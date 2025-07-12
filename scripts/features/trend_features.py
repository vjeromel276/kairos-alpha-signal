"""
trend_features.py

Generates trend-based features using moving averages and momentum indicators.
Focuses on smoothing and slope dynamics over short and medium horizons.

Features:
- Simple and exponential moving averages (5, 12, 21, 26 days)
- Price position vs SMA
- Slope of SMA (5-day difference)
- MACD, MACD signal, MACD histogram

Input:
    data/base/sep_base.parquet

Output:
    scripts/feature_matrices/<YYYY-MM-DD>_trend_features.parquet

To run:
    python scripts/features/trend_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
INPUT_PATH = Path("data/base/sep_base.parquet")
OUTPUT_DIR = Path("scripts/feature_matrices/")
TODAY = datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = OUTPUT_DIR / f"{TODAY}_trend_features.parquet"

# Load and sort
df = pd.read_parquet(INPUT_PATH)
df = df.sort_values(["ticker", "date"])

# Simple moving averages
df["sma_5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())
df["sma_21"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(21).mean())

# EMA for MACD
df["ema_12"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df["ema_26"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())

# Price position vs trend
df["price_vs_sma_21"] = (df["close"] - df["sma_21"]) / df["sma_21"]

# Slope of SMA (how steep is the trend?)
df["sma_21_slope"] = df.groupby("ticker")["sma_21"].transform(lambda x: x - x.shift(5))

# MACD features
df["macd"] = df["ema_12"] - df["ema_26"]
df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
df["macd_hist"] = df["macd"] - df["macd_signal"]

# Clean
df = df.dropna()

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Trend features saved to {OUTPUT_PATH}")
