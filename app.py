"""
Banking Customer Segmentation - Streamlit app.

K-Means segmentation of 883,660 retail banking customers, built from 1,047,732
raw transactions (Indian retail bank, Aug-Oct 2016).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import inference as inf
import viz

st.set_page_config(page_title="Banking Customer Segmentation",
                   page_icon="▦", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  .note {color:#666666; font-size:13px; line-height:1.5;}
  .card {border:1px solid #e8e8e8; border-radius:10px; padding:16px 18px; background:#ffffff;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    return inf.load_segmentation()


@st.cache_data(show_spinner=False)
def load_tables():
    return {
        "segment_sample": pd.read_parquet(ROOT / "data/segment_sample.parquet"),
        "k_selection": pd.read_csv(ROOT / "reports/k_selection.csv"),
        "top_loc": pd.read_csv(ROOT / "reports/segment_top_locations.csv"),
    }


try:
    SEG_PIPE, SEG_META, SEG_PROFILE = load_models()
    T = load_tables()
except FileNotFoundError as e:
    st.error(f"Model artefacts missing: {e}\n\nRun the pipeline first - see README.md.")
    st.stop()

RANKED = SEG_PROFILE.sort_values("balance_avg", ascending=False)["label"].tolist()
SEG_COLORS = viz.segment_colors(RANKED)

st.sidebar.markdown("### Customer Segmentation")
PAGE = st.sidebar.radio("Page", ["Segment a customer", "Segment analytics", "About the data"],
                        label_visibility="collapsed", key="nav_page")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div class='note'>K-Means, k={SEG_META['k']}<br>"
    f"{SEG_META['n_customers']:,} customers<br>"
    f"1,047,732 transactions<br>Aug-Oct 2016</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page 1: interactive segment assignment
# --------------------------------------------------------------------------- #
if PAGE == "Segment a customer":
    st.title("Which segment does this customer belong to?")
    st.markdown("<div class='note'>Set the three attributes the model was fitted on. "
                "The segment, the nearest-centroid margin and the recommended sales "
                "action update live.</div>", unsafe_allow_html=True)
    st.write("")

    controls, result = st.columns([1, 1.5], gap="large")

    with controls:
        st.subheader("Customer parameters")
        preset = st.selectbox(
            "Start from a preset",
            ["Custom", "Young salaried professional", "Senior high-balance saver",
             "Low-balance high-spender", "Dormant low-activity customer"])
        presets = {
            "Young salaried professional": (28.0, 1200.0, 45_000.0),
            "Senior high-balance saver": (55.0, 900.0, 250_000.0),
            "Low-balance high-spender": (31.0, 3000.0, 800.0),
            "Dormant low-activity customer": (34.0, 150.0, 6_000.0),
        }
        if preset in presets:
            d_age, d_amt, d_bal = presets[preset]
        else:
            med = SEG_META["feature_medians"]
            d_age, d_amt, d_bal = med["age"], med["monetary_avg"], med["balance_avg"]

        age = st.slider("Age (years)", 18.0, 95.0, float(d_age), 0.5)
        amt = st.number_input("Average transaction amount (₹)", 1.0, 1_500_000.0,
                              float(d_amt), step=50.0,
                              help="Mean value of the customer's transactions.")
        bal = st.number_input("Average account balance (₹)", 0.0, 50_000_000.0,
                              float(d_bal), step=1000.0)
        st.caption("Money inputs are log-transformed before scaling, so the model "
                   "reads them on a ratio scale — ₹1k→₹10k moves the customer as "
                   "much as ₹10k→₹100k.")

    res = inf.predict_segment(SEG_PIPE, SEG_META,
                              {"age": age, "monetary_avg": amt, "balance_avg": bal})
    row = SEG_PROFILE[SEG_PROFILE.segment == res["segment"]].iloc[0]

    with result:
        st.subheader("Assignment")
        k1, k2, k3 = st.columns([2, 1, 1])
        k1.metric("Segment", res["label"])
        k2.metric("Cluster ID", res["segment"])
        k3.metric("Share of base", f"{row.share_pct:.1f}%")

        st.markdown(f"<div class='card'><b>Recommended action</b><br>"
                    f"<span class='note'>{res['action']}</span></div>",
                    unsafe_allow_html=True)
        st.write("")
        if res["margin"] >= 0.35:
            st.success(f"Clear assignment — the nearest two centroids are "
                       f"{res['margin']:.2f} apart in scaled space.")
        else:
            st.warning(f"Borderline customer — the two nearest centroids are only "
                       f"{res['margin']:.2f} apart, so this profile sits near a "
                       f"segment boundary. Treat the second-ranked segment's action "
                       f"as a valid alternative.")

        order = np.argsort(res["distances"])
        names = SEG_META["segment_names"]
        prox = pd.DataFrame({
            "Segment": [names[str(i)]["label"] for i in order],
            "Relative proximity": (res["confidence"][order] * 100).round(1),
        })
        st.plotly_chart(
            viz.hbar(prox["Segment"], prox["Relative proximity"],
                     "Proximity to each segment centroid", "Relative proximity (%)",
                     color=[SEG_COLORS.get(s, viz.SERIES_1) for s in prox["Segment"]],
                     texts=[f"{v:.1f}%" for v in prox["Relative proximity"]]),
            width="stretch")
        st.caption("Inverse-distance proximity, not a calibrated probability — "
                   "K-Means is a hard partition and assigns every point to exactly "
                   "one centroid.")

    st.write("")
    st.subheader("Where this customer sits in the base")
    s = T["segment_sample"]
    tab1, tab2 = st.tabs(["Balance vs spend", "Age vs balance"])

    def scatter(xcol, ycol, xlab, ylab, logx, logy):
        here = {"balance_avg": bal, "age": age, "monetary_avg": amt}
        fig = go.Figure()
        for lab in RANKED:
            sub = s[s.label == lab]
            fig.add_trace(go.Scattergl(
                x=sub[xcol], y=sub[ycol], mode="markers", name=lab,
                marker=dict(size=4, color=SEG_COLORS[lab], opacity=0.5, line=dict(width=0)),
                hovertemplate=f"{lab}<br>{xlab}: %{{x:,.0f}}<br>{ylab}: %{{y:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[here[xcol]], y=[here[ycol]], mode="markers", name="This customer",
            marker=dict(size=17, color="#dc3c3d", symbol="x-thin",
                        line=dict(width=3, color="#dc3c3d")),
            hovertemplate="This customer<extra></extra>"))
        fig.update_xaxes(title=xlab, type="log" if logx else "linear")
        fig.update_yaxes(title=ylab, type="log" if logy else "linear")
        fig.update_layout(title=f"{ylab} vs {xlab} — 25,000-customer sample")
        return viz.style(fig, height=460)

    with tab1:
        st.plotly_chart(scatter("balance_avg", "monetary_avg", "Average balance (₹, log)",
                                "Average transaction (₹, log)", True, True), width="stretch")
    with tab2:
        st.plotly_chart(scatter("age", "balance_avg", "Age (years)",
                                "Average balance (₹, log)", False, True), width="stretch")


# --------------------------------------------------------------------------- #
# Page 2: segment analytics
# --------------------------------------------------------------------------- #
elif PAGE == "Segment analytics":
    st.title("Segment analytics")
    prof = SEG_PROFILE.sort_values("balance_avg", ascending=False).reset_index(drop=True)

    cols = st.columns(len(prof))
    for i, r in prof.iterrows():
        cols[i].metric(r.label, f"{r.share_pct:.1f}%", f"{int(r.customers):,} customers",
                       delta_color="off")

    st.write("")
    st.subheader("Segment profiles")
    show = prof[["label", "customers", "share_pct", "age", "monetary_avg", "balance_avg",
                 "balance_to_spend", "frequency", "recency_days", "action"]].copy()
    show.columns = ["Segment", "Customers", "Share %", "Avg age", "Median txn ₹",
                    "Median balance ₹", "Balance ÷ spend", "Avg txns", "Avg recency (days)",
                    "Recommended action"]
    st.dataframe(
        show.style.format({"Customers": "{:,.0f}", "Share %": "{:.1f}", "Avg age": "{:.1f}",
                           "Median txn ₹": "{:,.0f}", "Median balance ₹": "{:,.0f}",
                           "Balance ÷ spend": "{:.1f}", "Avg txns": "{:.2f}",
                           "Avg recency (days)": "{:.1f}"}),
        width="stretch", hide_index=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.hbar(prof.label, prof.balance_avg.round(0),
                                 "Median account balance by segment", "Median balance (₹)",
                                 color=[SEG_COLORS[l] for l in prof.label],
                                 texts=[f"₹{v:,.0f}" for v in prof.balance_avg]), width="stretch")
    with c2:
        st.plotly_chart(viz.hbar(prof.label, prof.monetary_avg.round(0),
                                 "Median transaction value by segment", "Median transaction (₹)",
                                 color=[SEG_COLORS[l] for l in prof.label],
                                 texts=[f"₹{v:,.0f}" for v in prof.monetary_avg]), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(viz.hbar(prof.label, prof.age.round(1), "Average age by segment",
                                 "Average age (years)",
                                 color=[SEG_COLORS[l] for l in prof.label],
                                 texts=[f"{v:.1f}" for v in prof.age]), width="stretch")
    with c4:
        st.plotly_chart(viz.hbar(prof.label, prof.balance_to_spend.round(1),
                                 "Balance-to-spend ratio by segment",
                                 "Median balance ÷ median transaction",
                                 color=[SEG_COLORS[l] for l in prof.label],
                                 texts=[f"{v:.1f}×" for v in prof.balance_to_spend]), width="stretch")
    st.caption("A balance-to-spend ratio below 1 means the customer's typical "
               "transaction is larger than the balance backing it.")

    st.write("")
    st.subheader("How k was chosen")
    ks = T["k_selection"]
    fig = go.Figure(go.Scatter(x=ks.k, y=ks.silhouette, mode="lines+markers",
                               line=dict(color=viz.SERIES_1, width=2),
                               marker=dict(size=9, color=viz.SERIES_1),
                               hovertemplate="k=%{x}<br>silhouette %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="k (number of clusters)", dtick=1)
    fig.update_yaxes(title="Silhouette score")
    fig.update_layout(title="Silhouette across k — initial 9-feature set")
    st.plotly_chart(viz.style(fig, height=320, showlegend=False), width="stretch")

    st.markdown(f"""
**The honest version of this chart.** On the first 9-feature set the silhouette
*peaked at k=2* ({ks.silhouette.max():.3f}) and sat near 0.16–0.19 everywhere else — weak
separation. Two features were the cause:

- `frequency` is degenerate — the median customer appears **once** in the 82-day window, so the RFM "F" axis carries almost no variance.
- `recency` is near-uniform for the same reason: every customer transacted inside the same 82 days.

Dropping them and fitting on **age, average transaction value and average balance**
raised the silhouette to **0.279 at k=4**. k=2 scores higher still, but splits the base
into "rich" and "not rich", which no sales team can act on. So k=4 is a deliberate
trade of a little cohesion for four tiers that map to different products.

**What this model is not.** A silhouette of 0.28 means these are *operational
partitions*, not well-separated natural populations — the customer base is closer to
a continuum than to four clumps. It is a defensible way to route a sales list; it is
not evidence that four distinct customer species exist.
""")

    st.write("")
    st.subheader("Geographic concentration")
    loc = T["top_loc"].copy()
    names = SEG_META["segment_names"]
    loc["Segment"] = loc.segment.astype(str).map(lambda s: names[s]["label"])
    pick = st.selectbox("Segment", RANKED, key="loc_seg")
    sub = loc[loc.Segment == pick].head(8)
    st.plotly_chart(viz.hbar(sub.location, sub.customers, f"Top locations — {pick}",
                             "Customers", color=SEG_COLORS[pick],
                             texts=[f"{v:,}" for v in sub.customers]), width="stretch")


# --------------------------------------------------------------------------- #
# Page 3: about the data
# --------------------------------------------------------------------------- #
else:
    st.title("About this project")
    c = st.columns(4)
    c[0].metric("Transactions processed", "1,047,732")
    c[1].metric("Customers segmented", f"{SEG_META['n_customers']:,}")
    c[2].metric("Segments", SEG_META["k"])
    c[3].metric("Period", "Aug–Oct 2016")

    st.markdown("""
### Pipeline

```
bank_transactions.csv  (1,048,567 rows)
        │
        ├─ clean        placeholder DOB, century pivot, invalid ages, zero-value txns
        ├─ aggregate    transaction log → one row per customer (RFM + behaviour)
        ├─ transform    log1p on money columns → StandardScaler
        └─ cluster      K-Means (k=4) → segment + recommended sales action
```

### Features the model is fitted on

| Feature | Why |
|---|---|
| `age` | Life stage drives which products fit |
| `monetary_avg` | Average transaction value — spending power |
| `balance_avg` | Average account balance — the wealth axis |

`recency`, `frequency` and the balance-to-spend ratio are computed and reported per
segment, but are **not** fitted on — see *Segment analytics* for the silhouette
evidence behind that decision.

### Data quality defects handled

| Defect | Rows affected | Handling |
|---|---|---|
| `CustomerDOB` = `1/1/1800` placeholder | 57,339 | Treated as missing |
| 2-digit years pivot 1957 → 2057 | 54,427 | DOB after the snapshot pulled back one century |
| Ages outside 18–95 after parsing | — | Set to null, then median-imputed |
| Balance skew: median ₹16,792 vs max ₹115,035,495 | all | `log1p` before scaling |
| Zero-value transactions | 835 | Dropped |

### Honest limitations

- **The median customer appears once.** With an 82-day window and 1.19 transactions
  per customer on average, this is close to a cross-sectional snapshot, not a
  behavioural time series. True RFM segmentation needs a longer window.
- **Silhouette 0.279 is weak.** These are operational partitions, not natural clusters.
- **No ground truth.** Segmentation is unsupervised; the segment names are
  interpretations of the centroids, not validated customer types.
- **Recommended actions are illustrative** product mappings, not credit decisions.
  Real product eligibility needs consent, suitability and underwriting checks.
""")
    st.caption("Source: bank_transactions.csv (Kaggle) — Indian retail bank transaction log.")
