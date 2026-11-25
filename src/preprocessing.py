"""
Module 1 data preparation: raw transaction log -> customer-level feature table.

Source: bank_transactions.csv (Indian retail bank, 1,048,567 transactions,
884,265 customers, 2016-08-01 to 2016-10-21).

Known data-quality defects in this dataset, handled explicitly below:
  1. CustomerDOB contains the placeholder '1/1/1800' for 57,339 rows.
  2. Dates use 2-digit years. pandas pivots '57' to 2057, so ~54k customers
     born in the 1950s-60s appear to be born in the future.
  3. CustAccountBalance and TransactionAmount are extremely right-skewed
     (balance max 115,035,495 INR vs median 16,792) -> log1p transform.
"""
import numpy as np
import pandas as pd

RAW_COLS = {
    "TransactionID": "txn_id",
    "CustomerID": "customer_id",
    "CustomerDOB": "dob",
    "CustGender": "gender",
    "CustLocation": "location",
    "CustAccountBalance": "balance",
    "TransactionDate": "txn_date",
    "TransactionTime": "txn_time",
    "TransactionAmount (INR)": "amount",
}

DOB_PLACEHOLDER = "1/1/1800"
SNAPSHOT_DATE = pd.Timestamp("2016-10-22")   # one day after last transaction

FEATURE_COLS = [
    "age", "recency_days", "frequency", "monetary_total", "monetary_avg",
    "balance_avg", "balance_to_spend", "txn_hour_avg", "tenure_days_active",
]


def _fix_two_digit_year(s: pd.Series, not_after: pd.Timestamp) -> pd.Series:
    """pandas maps '57' -> 2057. Pull anything after `not_after` back a century."""
    future = s > not_after
    s = s.copy()
    s[future] = s[future] - pd.DateOffset(years=100)
    return s


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLS)

    # --- defect 1: placeholder DOB -> missing ---
    df["dob"] = df["dob"].replace(DOB_PLACEHOLDER, np.nan)

    df["txn_date"] = pd.to_datetime(df["txn_date"], format="%d/%m/%y", errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], format="%d/%m/%y", errors="coerce")

    # --- defect 2: century pivot ---
    df["dob"] = _fix_two_digit_year(df["dob"], not_after=SNAPSHOT_DATE)

    df["age"] = ((SNAPSHOT_DATE - df["dob"]).dt.days / 365.25).round(1)
    # plausibility window for a banking customer
    df.loc[(df["age"] < 18) | (df["age"] > 95), "age"] = np.nan

    # TransactionTime is HHMMSS as an integer
    df["txn_hour"] = (df["txn_time"] // 10000).clip(0, 23)

    df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]))
    df = df[df["amount"] > 0]
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the transaction log to one row per customer (RFM + behaviour)."""
    g = df.groupby("customer_id")
    feat = pd.DataFrame({
        "recency_days": (SNAPSHOT_DATE - g["txn_date"].max()).dt.days,
        "frequency": g.size(),
        "monetary_total": g["amount"].sum(),
        "monetary_avg": g["amount"].mean(),
        "balance_avg": g["balance"].mean(),
        "txn_hour_avg": g["txn_hour"].mean(),
        "age": g["age"].max(),
        "tenure_days_active": (g["txn_date"].max() - g["txn_date"].min()).dt.days,
        "gender": g["gender"].first(),
        "location": g["location"].first(),
    }).reset_index()

    # headroom: how much balance backs each rupee of spend
    feat["balance_to_spend"] = feat["balance_avg"] / feat["monetary_avg"].replace(0, np.nan)

    # median-impute the numeric features that carry real nulls
    for c in ["age", "balance_avg", "balance_to_spend", "txn_hour_avg"]:
        feat[c] = feat[c].fillna(feat[c].median())

    feat["balance_to_spend"] = feat["balance_to_spend"].clip(upper=feat["balance_to_spend"].quantile(0.99))
    return feat


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """log1p the heavy-tailed money/count columns so K-Means sees comparable scales."""
    X = X.copy()
    for c in ["monetary_total", "monetary_avg", "balance_avg", "balance_to_spend", "frequency"]:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X
