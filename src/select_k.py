"""Step 2 (optional): silhouette sweep used to justify k. Writes reports/k_selection.csv."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from preprocessing import FEATURE_COLS, log_transform
from segmentation import select_k

feat = pd.read_parquet("data/customer_features.parquet")
print("sweep on the full 9-feature set (the one that scored poorly):")
sc = select_k(log_transform(feat[FEATURE_COLS]))
sc.to_csv("reports/k_selection.csv", index=False)

print("\nsweep on candidate feature subsets:")
for name, cols in {
    "age+spend+balance (chosen)": ["age", "monetary_avg", "balance_avg"],
    "+ balance/spend ratio": ["age", "monetary_avg", "balance_avg", "balance_to_spend"],
    "+ recency": ["age", "monetary_avg", "balance_avg", "recency_days"],
}.items():
    r = select_k(log_transform(feat[cols]), k_range=range(3, 7), sample=20_000)
    print(f"  {name:28s}", [(int(x.k), round(x.silhouette, 3)) for x in r.itertuples()])
