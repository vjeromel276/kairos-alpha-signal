"""
price_shape_features.py

Extracts candlestick shape and gap-related features from OHLCV data to capture
short-term market psychology and price pressure.

Features:
- Candle body, wick sizes
- Candle body and wick % of total range
- Gap up/down (absolute and %)

Input:
    data/base/sep_base.parquet

Output:
    scripts/feature_matrices/<YYYY-MM-DD>_price_shape_features.parquet

To run:
    python scripts/features/price_shape_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
INPUT_PATH = Path("data/base/sep_base.parquet")
OUTPUT_DIR = Path("scripts/feature_matrices/")
TODAY = datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = OUTPUT_DIR / f"{TODAY}_price_shape_features.parquet"

# Load and sort
df = pd.read_parquet(INPUT_PATH)
df = df.sort_values(["ticker", "date"])

# Candlestick components
df["body_size"] = (df["close"] - df["open"]).abs()
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
df["candle_range"] = df["high"] - df["low"]

# Ratios to range
df["body_pct_of_range"] = df["body_size"] / df["candle_range"]
df["upper_wick_pct"] = df["upper_wick"] / df["candle_range"]
df["lower_wick_pct"] = df["lower_wick"] / df["candle_range"]

# Previous close for gap analysis
df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["gap_open"] = df["open"] - df["prev_close"]
df["gap_pct"] = df["gap_open"] / df["prev_close"]

# Clean
df = df.dropna()

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Price shape features saved to {OUTPUT_PATH}")
