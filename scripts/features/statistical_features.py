"""
statistical_features.py

Computes z-score and percentile-based indicators to assess price stretch,
statistical outliers, and mean reversion potential.

Features:
- Z-scores of close price and returns
- Return percentile rank vs rolling window
- Price distance from rolling max
- Simple mean-reversion flag

Input:
    data/base/sep_base.parquet

Output:
    scripts/feature_matrices/<YYYY-MM-DD>_statistical_features.parquet

To run:
    python scripts/features/statistical_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Optional numba speed-up
try:
    from numba import njit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

# Paths
INPUT_PATH = Path("data/base/sep_base.parquet")
OUTPUT_DIR = Path("scripts/feature_matrices/")
TODAY = datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = OUTPUT_DIR / f"{TODAY}_statistical_features.parquet"

# Load and sort
df = pd.read_parquet(INPUT_PATH)
df = df.sort_values(["ticker", "date"])

# Compute daily return
df["ret_1d"] = df.groupby("ticker", group_keys=False)["close"].pct_change()

# Compute rolling mean, std, and max in a single pass per group
def compute_group_stats(group):
    group["close_mean_21d"] = group["close"].rolling(21).mean()
    group["close_std_21d"] = group["close"].rolling(21).std()
    group["ret_1d_std_21d"] = group["ret_1d"].rolling(21).std()
    group["rolling_max_21d"] = group["close"].rolling(21).max()
    return group

df = df.groupby("ticker", group_keys=False).apply(compute_group_stats)

# Z-scores
df["close_zscore_21d"] = (df["close"] - df["close_mean_21d"]) / df["close_std_21d"]
df["ret_1d_zscore_21d"] = df["ret_1d"] / df["ret_1d_std_21d"]

# Efficient rolling percentile rank
if USE_NUMBA:
    @njit
    def fast_percentile_rank(arr):
        temp = np.argsort(arr)
        ranks = np.empty(len(arr), dtype=np.int64)
        for i in range(len(arr)):
            ranks[temp[i]] = i
        return ranks[-1] / (len(arr) - 1) if len(arr) > 1 else 0.5
else:
    def fast_percentile_rank(arr):
        temp = arr.argsort()
        ranks = temp.argsort()
        return ranks[-1] / (len(arr) - 1) if len(arr) > 1 else 0.5

df["ret_1d_rank_21d"] = (
    df.groupby("ticker")["ret_1d"]
    .transform(lambda x: x.rolling(21).apply(fast_percentile_rank, raw=True))
)

# Price % from recent high
df["price_pct_from_rolling_max_21d"] = (
    (df["close"] - df["rolling_max_21d"]) / df["rolling_max_21d"]
)

# Mean reversion flag
df["mean_reversion_flag"] = (df["close_zscore_21d"] < -2.0).astype(int)

# Drop any rows with nulls from rolling ops
df = df.dropna()

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Statistical features saved to {OUTPUT_PATH}")
