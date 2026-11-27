"""APL Logistics delivery assistant: late rate, outcome mix, shipping mode
and region breakdown, with filters.

Run from the project root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

from src import config
from src.data import delivered_only
from src.kpis import group_summary

st.set_page_config(page_title="APL Logistics", page_icon="🚚", layout="wide")

# ----------------------------------------------------------------- constants
CLASS_ORDER = ["Early", "On-time", "Delayed"]
CLASS_COLORS = {"Early": "#2ca02c", "On-time": "#1f77b4", "Delayed": "#d62728"}

FILTERS = {                      # key -> (label, column)
    "modes": ("Shipping mode", config.SHIP_MODE),
    "markets": ("Market", config.MARKET),
    "segments": ("Customer segment", config.SEGMENT),
    "regions": ("Order region", config.REGION),
}


# ------------------------------------------------------------------- loaders
@st.cache_data
def load_data() -> pd.DataFrame:
    return delivered_only(pd.read_parquet(config.PROCESSED_PATH)).reset_index(drop=True)


try:
    d_all = load_data()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}\n\nRun the data preparation notebooks "
             "first (they create the processed parquet file).")
    st.stop()

OPTIONS = {k: sorted(d_all[col].unique()) for k, (_, col) in FILTERS.items()}

# ------------------------------------------------------------- session state
for k in FILTERS:
    st.session_state.setdefault(f"f_{k}", OPTIONS[k])


def on_reset():
    """Select everything again."""
    for k in FILTERS:
        st.session_state[f"f_{k}"] = OPTIONS[k]


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Dataset")
    st.metric("Delivered orders in file", f"{len(d_all):,}")
    st.info(
        "Data note: in this dataset, lateness follows the chosen shipping mode very "
        "regularly (for example, First Class always takes 2 days against a 1-day promise). "
        "This looks like a simplified or generated dataset, so treat the patterns as "
        "properties of this file, not as proven real-world operating facts. "
        "Canceled shipments (7,754 orders) are excluded."
    )
    st.header("Filters")
    st.button("Reset all filters", on_click=on_reset, use_container_width=True)
    st.caption("Late-rate bands (High ≥ 50%, Medium 25–50%) are a simple display choice.")

st.title("🚚 Delivery performance assistant")
st.caption("Shows how often orders arrive late, why, and where the delays concentrate.")

# --------------------------------------------------------------------- form
col_in, col_out = st.columns([1, 1.25], gap="large")

with col_in:
    st.subheader("Order filters")
    a, b = st.columns(2)
    with a:
        st.multiselect(FILTERS["modes"][0], OPTIONS["modes"], key="f_modes")
        st.multiselect(FILTERS["segments"][0], OPTIONS["segments"], key="f_segments")
    with b:
        st.multiselect(FILTERS["markets"][0], OPTIONS["markets"], key="f_markets")
        st.multiselect(FILTERS["regions"][0], OPTIONS["regions"], key="f_regions")

# ------------------------------------------------------------- filter the data
mask = pd.Series(True, index=d_all.index)
for k, (_, col) in FILTERS.items():
    mask &= d_all[col].isin(st.session_state[f"f_{k}"])
d = d_all[mask]

if d.empty:
    with col_out:
        st.warning("No orders match these filters.")
    st.stop()

# ------------------------------------------------------------------ summary
late_rate = float(d[config.TARGET].mean())
on_time_rate = float((d[config.GAP] <= 0).mean())
avg_gap = float(d[config.GAP].mean())

if late_rate >= 0.50:
    level, icon = "High", "🔴"
elif late_rate >= 0.25:
    level, icon = "Medium", "🟠"
else:
    level, icon = "Low", "🟢"

m = group_summary(d, config.SHIP_MODE).sort_values("late_rate").reset_index()
r = group_summary(d, config.REGION).sort_values("late_rate").reset_index()

# ------------------------------------------------------------------ results
with col_out:
    st.subheader("Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Delivered orders", f"{len(d):,}")
    m2.metric("Late rate", f"{late_rate:.1%}")
    m3.metric("Late-rate level", f"{icon} {level}")
    n1, n2 = st.columns(2)
    n1.metric("On-time rate (early + on-time)", f"{on_time_rate:.1%}")
    n2.metric("Average delay gap (days)", f"{avg_gap:.2f}")

    st.markdown("#### Headline")
    worst = m.iloc[-1]
    msg = (f"Highest late rate: **{worst[config.SHIP_MODE]}** at {worst['late_rate']:.1%} "
           f"({int(worst['orders']):,} orders) · "
           f"Selection late rate: **{late_rate:.1%}**.")
    (st.warning if late_rate >= 0.25 else st.success)(msg)

    # ---- outcome mix
    st.markdown("#### Delivery outcome mix")
    mix = (d[config.DELIVERY_CLASS].value_counts(normalize=True)
             .reindex(CLASS_ORDER).fillna(0))
    fig, ax = plt.subplots(figsize=(6, 3.0))
    bars = ax.bar(mix.index, mix.values, color=[CLASS_COLORS[c] for c in mix.index])
    for bar, v in zip(bars, mix.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1%}",
                ha="center", va="bottom", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(mix.max() * 1.15, 0.05))
    ax.set_ylabel("Share of orders")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- gap distribution
    st.markdown("#### Delay gap (real minus scheduled days)")
    gap = d[config.GAP].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.bar(gap.index, gap.values,
           color=["#d62728" if g > 0 else "#1f77b4" for g in gap.index])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xticks(list(gap.index))
    ax.set_xlabel("Gap in days:  red = late, blue = on time or early")
    ax.set_ylabel("Orders")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- shipping mode
    st.markdown("#### By shipping mode")
    fig, ax = plt.subplots(figsize=(6, max(2.6, 0.55 * len(m))))
    ax.barh(m[config.SHIP_MODE], m["late_rate"],
            color=["#d62728" if v >= 0.5 else "#1f77b4" for v in m["late_rate"]])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Late rate:  red = 50% or more")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    mode_class = pd.crosstab(d[config.SHIP_MODE], d[config.DELIVERY_CLASS],
                             normalize="index").reindex(columns=CLASS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, max(2.6, 0.55 * len(mode_class))))
    left = pd.Series(0.0, index=mode_class.index)
    for c in CLASS_ORDER:
        ax.barh(mode_class.index, mode_class[c], left=left,
                color=CLASS_COLORS[c], label=c)
        left += mode_class[c]
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Outcome mix within each shipping mode")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(m.set_index(config.SHIP_MODE), use_container_width=True)

    # ---- region
    st.markdown("#### By order region")
    fig, ax = plt.subplots(figsize=(6, max(3.6, 0.32 * len(r))))
    ax.barh(r[config.REGION], r["late_rate"], color="#1f77b4")
    ax.axvline(late_rate, color="black", lw=0.9, ls="--")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Late rate  (dashed line = selection average)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Regions differ by only a few percentage points, and most of that is explained by "
        "each region's shipping-mode mix. Small regions are noisy, so check the order "
        "counts in the table below."
    )
    st.dataframe(r.set_index(config.REGION), use_container_width=True)
