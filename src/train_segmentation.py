"""Fit the final K-Means model on all customers and persist artefacts."""
import json, sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from preprocessing import log_transform
from segmentation import CHOSEN_K, CLUSTER_COLS, make_pipeline, name_segments, profile_segments

feat = pd.read_parquet("data/customer_features.parquet")
X = log_transform(feat[CLUSTER_COLS])

print(f"fitting KMeans(k={CHOSEN_K}) on {len(X):,} customers, features={CLUSTER_COLS}")
pipe = make_pipeline(CHOSEN_K).fit(X)
labels = pipe.predict(X)

profile = profile_segments(feat, labels)
names = name_segments(profile)
profile["label"] = profile["segment"].map(lambda s: names[s]["label"])
profile["action"] = profile["segment"].map(lambda s: names[s]["action"])

# KMeans keeps labels_ for every fitted row (883k int32 = 3.4 MB); the app only
# needs the centroids, so drop it before persisting
if hasattr(pipe.named_steps["kmeans"], "labels_"):
    del pipe.named_steps["kmeans"].labels_
joblib.dump(pipe, "models/segmentation_pipeline.pkl", compress=3)
profile.to_csv("reports/segment_profile.csv", index=False)
feat.assign(segment=labels).to_parquet("data/customers_segmented.parquet", index=False)

meta = {
    "cluster_cols": CLUSTER_COLS,
    "k": CHOSEN_K,
    "n_customers": int(len(feat)),
    "segment_names": names,
    "feature_ranges": {c: [float(feat[c].quantile(0.01)), float(feat[c].quantile(0.99))] for c in CLUSTER_COLS},
    "feature_medians": {c: float(feat[c].median()) for c in CLUSTER_COLS},
}
with open("models/segmentation_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(profile[["segment", "label", "customers", "share_pct", "age", "monetary_avg", "balance_avg", "frequency"]].to_string(index=False))
