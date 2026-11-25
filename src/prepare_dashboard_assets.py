"""Step 4: pre-compute the small artefacts the Streamlit app reads, so it starts fast."""
import json, sys
sys.path.insert(0, "src")
import pandas as pd

seg = pd.read_parquet("data/customers_segmented.parquet")
names = json.loads(open("models/segmentation_meta.json").read())["segment_names"]

samp = seg.sample(min(25_000, len(seg)), random_state=42).copy()
samp["label"] = samp["segment"].astype(str).map(lambda s: names[s]["label"])
samp[["customer_id", "age", "monetary_avg", "balance_avg", "recency_days", "frequency",
      "monetary_total", "balance_to_spend", "gender", "location", "segment", "label"]] \
    .to_parquet("data/segment_sample.parquet", index=False)

(seg.groupby(["segment", "location"]).size().rename("customers").reset_index()
   .sort_values(["segment", "customers"], ascending=[True, False])
   .groupby("segment").head(8)).to_csv("reports/segment_top_locations.csv", index=False)
(seg.groupby(["segment", "gender"]).size().rename("customers").reset_index()) \
    .to_csv("reports/segment_gender_mix.csv", index=False)
print("dashboard assets written:", samp.shape)
