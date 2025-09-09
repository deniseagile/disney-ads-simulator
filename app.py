# app.py — Disney+ Current Ads vs Merch (Amazon, ShopDisney, QR)
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Disney+ Ads vs Merch (Amazon / ShopDisney / QR)", layout="wide")
st.title("🎬 Disney+ Revenue: Current Ads vs Merch by Channel")
st.caption(
    "Loads CSVs from the repo (no uploads). Compare status-quo external ads vs merchandising via "
    "Amazon, ShopDisney, and Disney+ QR."
)

# ───────────────────────── Sidebar controls ─────────────────────────
with st.sidebar:
    st.header("⚙️ Global Controls")

    # Allocation sliders (we will normalize to 100% automatically)
    alloc_amz = st.slider("Alloc to Amazon (%)", 0, 100, 30, 5)
    alloc_shop = st.slider("Alloc to ShopDisney (%)", 0, 100, 35, 5)
    alloc_qr = st.slider("Alloc to Disney+ QR (%)", 0, 100, 35, 5)
    total_alloc_pct = alloc_amz + alloc_shop + alloc_qr
    total_alloc = max(total_alloc_pct, 1)  # avoid divide-by-zero
    st.caption(f"Total merch allocation: **{total_alloc_pct}%** (auto-normalized)")

    conv_mult_all = st.slider("Conversion multiplier (global)", 0.25, 2.5, 1.0, 0.05)
    qr_uplift = st.slider("QR uplift × (applied to QR only)", 1.0, 3.0, 1.5, 0.1)

    st.markdown("---")
    apply_amazon_fee_override = st.checkbox("Force Amazon fee = 30%", value=True)
    affiliate_on = st.checkbox("Apply Amazon affiliate rebate", value=True)
    respect_inventory = st.checkbox("Respect inventory caps (per channel)", value=True)

    st.markdown("---")
    cpm = st.slider("Status-Quo CPM ($ / 1,000 impressions)", 10, 120, 30, 1)

# Normalize allocation weights to sum to 1.0
alloc = {
    "Amazon": alloc_amz / total_alloc,
    "ShopDisney": alloc_shop / total_alloc,
    "Disney+ QR": alloc_qr / total_alloc,
}

# ───────────────────────── Data loading (repo-only) ─────────────────────────
def read_csv_or_none(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

files_needed = [
    "products.csv",
    "content_titles.csv",
    "content_product_map.csv",
    "viewership.csv",
    "channels.csv",
    "conversion_assumptions.csv",
]
dfs = {name: read_csv_or_none(name) for name in files_needed}
missing = [k for k, v in dfs.items() if v is None]
if missing:
    st.error("Missing required files in repo root:\n\n- " + "\n- ".join(missing))
    st.stop()

df_products = dfs["products.csv"]
df_titles = dfs["content_titles.csv"]
df_cpmap = dfs["content_product_map.csv"]
df_views = dfs["viewership.csv"]
df_channels = dfs["channels.csv"]
df_conv = dfs["conversion_assumptions.csv"]

# ───────────────────────── Type safety ─────────────────────────
for c in ["Price", "Cost", "Inventory_Level"]:
    df_products[c] = pd.to_numeric(df_products[c], errors="coerce").fillna(0.0)
df_views["Viewers_Exposed"] = pd.to_numeric(df_views["Viewers_Exposed"], errors="coerce").fillna(0.0)
df_views["Ad_Impressions"] = pd.to_numeric(df_views["Ad_Impressions"], errors="coerce").fillna(0.0)
for c in ["Amazon_Fee_Percent", "Affiliate_Commission_Percent"]:
    if c in df_channels.columns:
        df_channels[c] = pd.to_numeric(df_channels[c], errors="coerce").fillna(0.0)
df_conv["Base_Conversion_Rate"] = pd.to_numeric(df_conv.get("Base_Conversion_Rate", 0.0), errors="coerce").fillna(0.0)
df_conv["QR_Uplift_X"] = pd.to_numeric(df_conv.get("QR_Uplift_X", 1.0), errors="coerce").fillna(1.0)

# ───────────────────────── Build modeling frame ─────────────────────────
# Title × SKU base
base = (
    df_cpmap.merge(df_titles, on="Title_ID", how="left")
            .merge(df_products, on="SKU", how="left", suffixes=("", "_prod"))
)

# Attach viewership totals per title
views_agg = df_views.groupby("Title_ID", as_index=False).agg(
    Viewers_Exposed=("Viewers_Exposed", "sum"),
    Ad_Impressions=("Ad_Impressions", "sum"),
)
base = base.merge(views_agg, on="Title_ID", how="left")

# Cross-join channels to get Title × SKU × Channel rows
df_channels = df_channels.copy()
df_channels["_k"] = 1
base["_k"] = 1
model = base.merge(df_channels, on="_k").drop(columns="_k")

# Attach conversion assumptions
model = model.merge(df_conv, on="Channel", how="left")

# ───────────────────────── Per-channel allocation & conversion ─────────────────────────
# Effective viewers per channel = title viewers × allocation(channel)
model["Alloc"] = model["Channel"].map(alloc).fillna(0.0)
model["Effective_Viewers"] = model["Viewers_Exposed"].fillna(0.0) * model["Alloc"]

# Effective conversion: base × global multiplier × (QR uplift if QR)
conv = model["Base_Conversion_Rate"].astype(float).fillna(0.0) * conv_mult_all
is_qr = model["Channel"].str.lower().eq("disney+ qr")
conv = np.where(is_qr, conv * qr_uplift, conv)
conv_series = pd.to_numeric(pd.Series(conv, index=model.index), errors="coerce").fillna(0.0)
model["Conversion_Effective"] = np.maximum(conv_series.values, 0.0)

# ───────────────────────── Units & revenue per channel ─────────────────────────
price = model["Price"].fillna(0.0)
cost = model["Cost"].fillna(0.0)

# Units
model["Units_Raw"] = model["Effective_Viewers"] * model["Conversion_Effective"]
if respect_inventory:
    inv = model["Inventory_Level"].fillna(0.0)
    model["Units_Sold"] = np.minimum(model["Units_Raw"], inv)
else:
    model["Units_Sold"] = model["Units_Raw"]

# Amazon fee (override or CSV)
model["Amazon_Fee_Effective"] = np.where(
    model["Channel"].str.lower().eq("amazon"),
    0.30 if apply_amazon_fee_override else model.get("Amazon_Fee_Percent", pd.Series(0, index=model.index)).fillna(0.0),
    0.0,
)

# Revenue pieces
model["Revenue_Gross"] = model["Units_Sold"] * price
model["COGS"] = model["Units_Sold"] * cost
model["Amazon_Fees"] = model["Units_Sold"] * price * model["Amazon_Fee_Effective"]
model["Affiliate_Rebate"] = np.where(
    (model["Channel"].str.lower().eq("amazon")) & affiliate_on,
    model["Units_Sold"] * price * model.get("Affiliate_Commission_Percent", pd.Series(0, index=model.index)).fillna(0.0),
    0.0,
)
model["Revenue_Net"] = model["Revenue_Gross"] - model["COGS"] - model["Amazon_Fees"] + model["Affiliate_Rebate"]

# ───────────────────────── Scenario totals ─────────────────────────
# Status-quo: CPM × total impressions across titles
statusquo_net = float((df_views["Ad_Impressions"].sum() / 1000.0) * cpm)

by_channel = (model.groupby("Channel", as_index=False)
              .agg(Net_Revenue=("Revenue_Net", "sum"),
                   Units=("Units_Sold", "sum")))

def ch_total(name: str) -> float:
    row = by_channel[by_channel["Channel"].str.lower() == name.lower()]
    return float(row["Net_Revenue"].iloc[0]) if not row.empty else 0.0

net_amazon = ch_total("Amazon")
net_shop = ch_total("ShopDisney")
net_qr = ch_total("Disney+ QR")

# ───────────────────────── KPIs ─────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Ads (CPM) — Net", f"{statusquo_net:,.0f}")
c2.metric("Merch via Amazon — Net", f"{net_amazon:,.0f}")
c3.metric("Merch via ShopDisney — Net", f"{net_shop:,.0f}")
c4.metric("Merch via Disney+ QR — Net", f"{net_qr:,.0f}")

# ───────────────────────── Comparison chart ─────────────────────────
compare_df = pd.DataFrame({
    "Scenario": ["Current Ads", "Amazon Merch", "ShopDisney Merch", "Disney+ QR Merch"],
    "Net_Revenue": [statusquo_net, net_amazon, net_shop, net_qr],
}).set_index("Scenario")

st.markdown("### Impact: Current Ads vs Merch by Channel")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = ["#9aa0a6", "#1e88e5", "#2e7d32", "#fbc02d"]  # gray, blue, green, gold
bars = ax.bar(compare_df.index, compare_df["Net_Revenue"], color=colors, width=0.62)

ax.set_ylabel("Net Revenue (USD)")
ax.set_title("Revenue per 1M Viewers (or selected viewership)", fontsize=14, pad=12)

ymax = float(compare_df["Net_Revenue"].max())
ax.set_ylim(0, max(1.0, ymax) * 1.35)
ax.margins(x=0.08)

# Dollar labels inside bars
for bar, val in zip(bars, compare_df["Net_Revenue"]):
    label = f"${val/1_000:.0f}K" if val < 1_000_000 else f"${val/1_000_000:.1f}M"
    ax.text(bar.get_x() + bar.get_width()/2, val * 0.97, label,
            ha="center", va="top", fontsize=11, fontweight="bold", color="black")

# Uplift vs Current for each merch channel
current = float(compare_df.loc["Current Ads", "Net_Revenue"])
x_positions = [b.get_x() + b.get_width()/2 for b in bars]
heights = [b.get_height() for b in bars]

def annotate_uplift(ix: int, color: str):
    val = heights[ix]
    if current > 0:
        pct = (val - current) / current
        factor = val / current
        ax.annotate(
            f"+{pct*100:.0f}% vs Current  •  {factor:.1f}×",
            xy=(x_positions[ix], heights[ix]), xycoords="data",
            xytext=(0, 14), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=1)
        )

# annotate Amazon/Shop/QR (bars 1,2,3)
annotate_uplift(1, "#1e88e5")
annotate_uplift(2, "#2e7d32")
annotate_uplift(3, "#fbc02d")

ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10)
plt.subplots_adjust(top=0.86)
st.pyplot(fig)

# ───────────────────────── Channel detail ─────────────────────────
st.markdown("### Channel Detail (Net & Units)")
st.dataframe(by_channel.rename(columns={"Net_Revenue": "Net_Revenue_USD"}), use_container_width=True)

# ───────────────────────── SKU × Title × Channel table ─────────────────────────
st.markdown("### SKU × Title × Channel Detail")
cols = ["SKU", "Product_Name", "Title_Name", "Channel",
        "Price", "Cost", "Inventory_Level",
        "Alloc", "Effective_Viewers", "Conversion_Effective",
        "Units_Sold", "Revenue_Net"]
have = [c for c in cols if c in model.columns]
st.dataframe(
    model[have].sort_values(["Title_Name", "SKU", "Channel"]).reset_index(drop=True),
    use_container_width=True
)

st.markdown("---")
st.caption(
    "Status-quo = CPM × total impressions (viewership.csv). Merch scenarios allocate title viewers to "
    "Amazon, ShopDisney, and Disney+ QR per sliders, apply per-channel conversions & fees, and compute "
    "net revenue. Inventory caps are per-channel for simplicity."
)
