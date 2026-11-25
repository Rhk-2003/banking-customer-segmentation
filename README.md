# Banking Customer Segmentation

K-Means segmentation of **883,660 retail banking customers**, built from **1,047,732 raw
transactions** of an Indian retail bank (Aug–Oct 2016), served through an interactive
Streamlit app that assigns any customer profile to a segment in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

---

## What it does

Set three attributes — age, average transaction value, average account balance — and the
app returns the customer's segment, how clear-cut that assignment is, and the sales
action that fits the segment. A scatter view plots the customer against a 25,000-customer
sample so you can see *where* they land, not just which label they get.

## The four segments

| Segment | Share | Avg age | Median balance | Median txn | Sales action |
|---|---|---|---|---|---|
| **Established Mid-Balance** | 13.1% | 49.6 | ₹53,523 | ₹912 | Term deposits, portfolio review, retirement and protection |
| **Young Mid-Balance** | 40.8% | 30.7 | ₹34,682 | ₹1,105 | Volume segment: recurring deposits, auto-debit, card upgrades |
| **Young Low-Balance** | 35.4% | 28.0 | ₹12,356 | ₹160 | Balance-building: step-up RDs, salary account pitch |
| **Near-Zero Balance** | 10.7% | 28.6 | ₹118 | ₹285 | Retention first: balance-building, fee waivers, re-onboarding |

The Near-Zero group is the only sharply defined one — median ₹118, max ₹1,813 — and the
only genuinely actionable segment: a balance-to-spend ratio of **0.5** means their typical
transaction is twice the balance backing it.

### Why no "affluent" or "premium" in these names

An earlier version of this project ranked the clusters and labelled them *Affluent
Established* / *Core Mass Affluent*. The data does not support that vocabulary:

- The top cluster's median balance of ₹53,523 is only the **72nd percentile** of this base.
- "Core Mass Affluent" covered **40.8% of all customers** — a plurality cannot be affluent.
- Of the 15,722 customers actually holding over ₹10 lakh, **52.8% sat in the *second*
  tier**, not the top one — the "affluent" cluster did not even contain most affluent customers.
- The real separator between the top two clusters is **age** (median 47 vs 30), not wealth.
  Because balance is log-transformed before clustering, ₹53k and ₹11 crore land close
  together and age splits them instead.

The names now describe balance tier and life stage, which is what the clusters actually
encode. Absolute wealth is not something this dataset can speak to — it records average
account balance from a transaction log, not total relationship value.

## Pipeline

```
bank_transactions.csv  (1,048,567 rows)
      │
      ├─ clean        placeholder DOB, century pivot, invalid ages, zero-value txns
      ├─ aggregate    transaction log → one row per customer (RFM + behaviour)
      ├─ transform    log1p on money columns → StandardScaler
      └─ cluster      K-Means (k=4) → segment + recommended action
```

## Two decisions worth defending in an interview

**1. Why only three features?**
The first model used all nine engineered features and scored a silhouette of **0.19**.
The cause was in the data, not the algorithm: the median customer appears **once** in the
82-day window, so `frequency` is degenerate and `recency` is near-uniform — the "R" and
"F" of RFM carry almost no variance here. Dropping them lifted the silhouette to
**0.279**. Adding the balance-to-spend ratio lowered it again (0.273), because it is
collinear with the two features it is derived from.

**2. Why k=4 when the silhouette peaks at k=2?**
k=2 splits the base into "rich" and "not rich" — statistically cleaner, commercially
useless. k=4 trades a little cohesion for four tiers that map to four different product
conversations. This is an operational partition, chosen on business value, and the app
says so on its own analytics page rather than quietly reporting the best number.

## Data quality defects found and handled

| Defect | Rows | Handling |
|---|---|---|
| `CustomerDOB` = `1/1/1800` placeholder | 57,339 | Treated as missing |
| 2-digit years pivot 1957 → **2057** | 54,427 | DOB after snapshot pulled back one century |
| Ages outside 18–95 after parsing | — | Null, then median-imputed |
| Balance skew: median ₹16,792 vs max ₹115,035,495 | all | `log1p` before scaling |
| Zero-value transactions | 835 | Dropped |

Left unhandled, the century bug alone would have put 6% of customers at a negative age.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained model and dashboard assets are committed, so the app runs immediately.

### Rebuilding from the raw data

Download `bank_transactions.csv` from Kaggle into `data/`, then:

```bash
python src/build_features.py data/bank_transactions.csv   # clean + aggregate
python src/select_k.py                                    # silhouette sweep (optional)
python src/train_segmentation.py                          # fit + persist
python src/prepare_dashboard_assets.py                    # app assets
```

## Layout

```
├── app.py                            Streamlit app
├── src/
│   ├── preprocessing.py              cleaning + customer feature engineering
│   ├── segmentation.py               pipeline, k selection, segment naming
│   ├── build_features.py             step 1
│   ├── select_k.py                   step 2
│   ├── train_segmentation.py         step 3
│   ├── prepare_dashboard_assets.py   step 4
│   ├── inference.py                  shared by app + scripts
│   └── viz.py                        chart theme
├── models/                           fitted pipeline + metadata
├── reports/                          segment profiles, k-selection results
└── data/                             dashboard sample (raw CSV gitignored)
```

## Limitations

- **The median customer appears once.** This is closer to a cross-sectional snapshot than
  a behavioural time series; real RFM segmentation needs a longer observation window.
- **Silhouette 0.279 is weak.** The customer base is closer to a continuum than to four
  natural clumps. These are useful partitions, not discovered customer species.
- **No ground truth.** Segment names are interpretations of centroid positions.
- **Recommended actions are illustrative** product mappings, not credit decisions — real
  offers need consent, suitability and underwriting checks.

## Data source

[Bank Customer Segmentation (Kaggle)](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation) — 1,048,567 transactions from an Indian retail bank.

## Push to GitHub

```bash
git remote add origin https://github.com/Rhk-2003/banking-customer-segmentation.git
git branch -M main
git push -u origin main
```

The raw Kaggle CSV and the large regenerable intermediates are gitignored; the trained
model and dashboard assets are committed, so anyone cloning the repo can run the app
without downloading the dataset.
