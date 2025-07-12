"""
build_full_feature_matrix.py

Merges all modular feature blocks and targets into a single feature matrix
filtered by a midcap+ ticker universe.

Usage:
    python scripts/features/build_full_feature_matrix.py --date 2025-07-03
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import argparse

# ---- CLI
parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, help="Date string (YYYY-MM-DD) to load feature files")
args = parser.parse_args()
date_str = args.date

# ---- Paths
DATA_DIR = Path("scripts/feature_matrices/")
UNIVERSE_PATH = Path(f"scripts/sep_dataset/feature_sets/midcap_and_up_ticker_universe_{date_str}.csv")
OUTPUT_PATH = DATA_DIR / f"{date_str}_full_feature_matrix.parquet"

# ---- Feature blocks to merge
feature_files = [
    f"{date_str}_price_action_features.parquet",
    f"{date_str}_volume_volatility_features.parquet",
    f"{date_str}_trend_features.parquet",
    f"{date_str}_price_shape_features.parquet",
    f"{date_str}_statistical_features.parquet",
    f"{date_str}_targets.parquet",
]

# ---- Load universe
print(f"📂 Loading ticker universe: {UNIVERSE_PATH}")
tickers = pl.read_csv(UNIVERSE_PATH).select("ticker")

# ---- Merge feature blocks
print("🔗 Merging feature blocks with Polars...")
df = pl.read_parquet(DATA_DIR / feature_files[0])
start_dates = df.select("date").unique().sort("date").tail(1)
start_row_count = df.height

for f in feature_files[1:]:
    next_df = pl.read_parquet(DATA_DIR / f)

    # Drop duplicate columns except for ticker/date
    dupes = [col for col in next_df.columns if col in df.columns and col not in ("ticker", "date")]
    if dupes:
        print(f"⚠️ Dropping duplicate columns from {f}: {dupes}")
        next_df = next_df.drop(dupes)

    df = df.join(next_df, on=["ticker", "date"], how="left")

# ---- Filter to ticker universe
df = df.join(tickers, on="ticker", how="inner")

# ---- Optional cleaning (retain rows with essential fields only)
df = df.filter(pl.col("date").is_not_null() & pl.col("ticker").is_not_null())

# ---- Report live date preservation
final_dates = df.select("date").unique().sort("date").tail(1)
if start_dates.item() != final_dates.item():
    start_dates = start_dates.item()
    final_dates = final_dates.item()
    print(f"⚠️ Most recent date dropped: had {start_dates}, now {final_dates}")

print(f"✅ Final matrix: {df.height:,} rows × {df.width} columns")

# ---- Save
df.write_parquet(OUTPUT_PATH)
print(f"💾 Saved full matrix to: {OUTPUT_PATH}")
