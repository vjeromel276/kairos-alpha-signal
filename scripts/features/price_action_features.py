"""
price_action_features.py

Generates price action and momentum-based features from the base SEP OHLCV dataset.
This script loads the gold-standard price dataset (1998–present), computes a
set of canonical derived features based on price action, and saves the resulting
feature matrix as a dated `.parquet` file.

Features engineered include:
- Daily returns
- Rolling N-day returns (e.g., 5d, 21d)
- Price ratios (high/low, close/open)
- True range and price range percentage

Input:
    data/base/sep_base.parquet (filtered for 100/100 coverage, ~1850 tickers)

Output:
    scripts/feature_matrices/<YYYY-MM-DD>_feature_matrix.parquet

To run:
    python scripts/features/price_action_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
INPUT_PATH = Path("data/base/sep_base.parquet")
OUTPUT_DIR = Path("scripts/feature_matrices/")
TODAY = datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = OUTPUT_DIR / f"{TODAY}_price_action_features.parquet"

# Load gold dataset
df = pd.read_parquet(INPUT_PATH)

# Sort for rolling ops
df = df.sort_values(["ticker", "date"])

# Daily return
df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

# Rolling returns
df["ret_5d"] = df.groupby("ticker")["close"].pct_change(5)
df["ret_21d"] = df.groupby("ticker")["close"].pct_change(21)

# Price ratios
df["hl_ratio"] = df["high"] / df["low"]
df["co_ratio"] = df["close"] / df["open"]

# True range and price range %
df["true_range"] = df["high"] - df["low"]
df["range_pct"] = (df["high"] - df["low"]) / df["open"]

# Drop rows with nulls from pct_change
df = df.dropna()

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Feature matrix saved to {OUTPUT_PATH}")
