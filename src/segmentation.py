"""
Module 1: K-Means customer segmentation.

Pipeline: log1p -> StandardScaler -> KMeans.
k is chosen by silhouette score on a stratified sample (the full 883k-row
silhouette is O(n^2) and not tractable), cross-checked against the inertia elbow.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_COLS, log_transform


def make_pipeline(k: int, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def select_k(X: pd.DataFrame, k_range=range(2, 9), sample=25_000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X.iloc[idx]
    rows = []
    for k in k_range:
        pipe = make_pipeline(k, seed).fit(Xs)
        labels = pipe.predict(Xs)
        Z = pipe.named_steps["scaler"].transform(Xs)
        rows.append({
            "k": k,
            "inertia": pipe.named_steps["kmeans"].inertia_,
            "silhouette": silhouette_score(Z, labels, sample_size=10_000, random_state=seed),
            "calinski_harabasz": calinski_harabasz_score(Z, labels),
            "davies_bouldin": davies_bouldin_score(Z, labels),
        })
        print(f"  k={k} sil={rows[-1]['silhouette']:.4f} db={rows[-1]['davies_bouldin']:.3f}", flush=True)
    return pd.DataFrame(rows)


def profile_segments(feat: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean/median of each raw (un-logged) feature per cluster, for interpretation."""
    d = feat.copy()
    d["segment"] = labels
    agg = d.groupby("segment").agg(
        customers=("customer_id", "size"),
        age=("age", "mean"),
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary_total=("monetary_total", "median"),
        monetary_avg=("monetary_avg", "median"),
        balance_avg=("balance_avg", "median"),
        balance_to_spend=("balance_to_spend", "median"),
        txn_hour_avg=("txn_hour_avg", "mean"),
    ).reset_index()
    agg["share_pct"] = (100 * agg["customers"] / len(d)).round(2)
    return agg


def name_segments(profile: pd.DataFrame) -> dict:
    """
    Label clusters from their own position in the fitted profile rather than by
    hand, so the names stay correct if the model is refit on new data.

    Names describe *what separates the clusters in this data* - balance tier and
    life stage - and nothing more. They deliberately avoid wealth vocabulary
    ("affluent", "premium", "high-value"): on this base the top cluster's median
    balance is only the 72nd percentile, and the second cluster covers 40% of all
    customers. Calling either group affluent would import an industry meaning
    (investable assets in the tens of lakhs) that these numbers do not support.

    Clusters are ranked by median balance, with median spend breaking ties, so
    the four labels are always distinct. An age qualifier is attached only where
    a cluster is genuinely older than the base.
    """
    p = profile.set_index("segment")
    order = p.sort_values(["balance_avg", "monetary_avg"], ascending=False).index.tolist()
    base_age = p["age"].median()

    names = {}
    for rank, seg in enumerate(order):
        age, bal = p.loc[seg, "age"], p.loc[seg, "balance_avg"]
        older = age >= base_age + 5          # distinctly older than the rest
        if rank == 0:
            label = ("Established Mid-Balance" if older else "Young Mid-Balance")
            action = ("Longest tenure and the highest balances in this base. Term deposits, "
                      "portfolio review, retirement and protection products.")
        elif rank == 1:
            label = ("Established Mid-Balance" if older else "Young Mid-Balance")
            if label in {v["label"] for v in names.values()}:
                label = "Mainstream Mid-Balance"
            action = ("The volume segment. Recurring deposits, auto-debit, card upgrades - "
                      "test offers here before rolling them out.")
        elif rank == 2:
            label = "Young Low-Balance"
            action = ("Balance-building first: step-up recurring deposits, salary account "
                      "pitch, digital engagement.")
        else:
            label = ("Near-Zero Balance" if bal <= 1_000 else "Minimal-Balance")
            action = ("Spends more than it holds. Retention and activation before any "
                      "credit pitch: balance-building offers, fee waivers, re-onboarding.")
        names[int(seg)] = {"label": label, "action": action, "balance_rank": rank}
    return names


# Features actually used to fit the model. Chosen empirically: adding recency,
# frequency or the balance/spend ratio lowered silhouette (recency is near-uniform
# because every customer transacts inside the same 82-day window, and frequency
# is degenerate at a median of 1 transaction per customer). Those three are still
# reported per segment as descriptive attributes.
CLUSTER_COLS = ["age", "monetary_avg", "balance_avg"]
CHOSEN_K = 4
