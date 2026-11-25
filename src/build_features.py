"""Step 1: raw transaction CSV -> customer feature table."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from preprocessing import FEATURE_COLS, build_customer_features, load_transactions

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/bank_transactions.csv"
print(f"reading {RAW}")
df = load_transactions(RAW)
print("usable transactions:", f"{len(df):,}")
feat = build_customer_features(df)
print("customers:", f"{len(feat):,}")
feat.to_parquet("data/customer_features.parquet", index=False)
print(feat[FEATURE_COLS].describe().T[["mean", "50%", "max"]])
