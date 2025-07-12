"""
generate_targets.py

Generates forward-looking return-based targets for supervised model training.
Includes both regression (e.g. next-day return) and classification labels (e.g. 5-day up/down).

Targets:
- ret_1d_f: Next day return
- ret_5d_f: 5-day forward return
- label_5d_up: 1 if ret_5d_f > 0, else 0

Input:
    data/base/sep_base.parquet

Output:
    scripts/feature_matrices/<YYYY-MM-DD>_targets.parquet

To run:
    python scripts/features/generate_targets.py --date 2025-07-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=True, help="Date label for output file (YYYY-MM-DD)")
args = parser.parse_args()

try:
    datetime.strptime(args.date, "%Y-%m-%d")
except ValueError:
    raise ValueError("Date must be in YYYY-MM-DD format")

# ---- Paths ----
INPUT_PATH = Path("data/base/sep_base.parquet")
OUTPUT_DIR = Path("scripts/feature_matrices/")
OUTPUT_PATH = OUTPUT_DIR / f"{args.date}_targets.parquet"

# ---- Load and Sort ----
df = pd.read_parquet(INPUT_PATH)
df = df.sort_values(["ticker", "date"])

# ---- Forward Returns ----
df["ret_1d_f"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["ret_5d_f"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["label_5d_up"] = np.where(df["ret_5d_f"].notna(), (df["ret_5d_f"] > 0).astype(int), np.nan)

print(f"🔍 Targets generated with {df['ret_5d_f'].isna().sum():,} rows lacking 5-day forward return")

# ---- Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Target matrix saved to {OUTPUT_PATH}")
