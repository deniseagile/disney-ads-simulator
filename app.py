# app.py
import os, ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Disney+ Contextual Ads with Scene Triggers", layout="wide")
st.title("🎬 Disney+ Contextual Merch — Normalized Model + Scene Triggers")
st.caption("Loads CSVs from the repository (no uploads). Compares Current Ads (CPM) vs Contextual Merch (Base & Scene-Triggered).")

# ---------------- Sidebar Controls (no uploaders) ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    ad_alloc = st.slider("% ad inventory allocated to merch", 0, 100, 40, 5) / 100.0
    conv_mult_all = st.slider("Global conversion multiplier", 0.25, 2.5, 1.0, 0.05)
    qr_uplift = st.slider("Disney+ QR uplift (x)", 1.0, 3.0, 1.5, 0.1)
    dsp_uplift = st.slider("DSP/Context relevance uplift (x)", 1.0, 2.0, 1.2, 0.05)
    apply_amazon_fee_override = st.checkbox("Force 30% Amazon fee", value=True)
    affiliate_on = st.checkbox("Include Amazon affiliate rebate", value=True)
    respect_inventory = st.checkbox("Respect inventory caps", value=True)
    # Status-quo baseline
    cpm = st.slider("CPM for Status Quo ($ per 1,000 impressions)", 10, 80, 30, 1)

# ---------------- Data loading (repo-only) ----------------
def must_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return None

files_needed = [
    "products.csv",
    "content_titles.csv",
    "content_product_map.csv",
    "viewership.csv",
    "channels.csv",
    "conversion_assumptions.csv",
]

loaded = {name: must_read_csv(name) for name in files_needed}
missing = [k for k, v in loaded.items() if v is None]

if missing:
    st.error(
        "Missing required files in repo root:\n\n- " +
        "\n- ".join(missing) +
        "\n\nCommit these CSVs (exact names) alongside app.py and reload."
    )
    st.stop()

df_products = loaded["products.csv"]
df_titles = loaded["content_titles.csv"]
df_cpmap = loaded["content_product_map.csv"]
df_views = loaded["viewership.csv"]
df_channels = loaded["channels.csv"]
df_conv = loaded["conversion_assumptions.csv"]

# Optional scenes file (no uploads; read only if present)
scenes_path = "scenes_normalized.csv"
df_scenes = must_read_csv(scenes_path) if os.path.exists(scenes_path) else None

# ---------------- Type safety ----------------
for c in ["Price","Cost","Inventory_Level"]:
    df_products[c] = pd.to_numeric(df_products[c], errors="coerce")
df_views["Viewers_Exposed"] = pd.to_numeric(df_views["Viewers_Exposed"], errors="coerce")
df_views["Ad_Impressions"] = pd.to_numeric(df_views["Ad_Impressions"], errors="coerce")
for c in ["Amazon_Fee_Percent","Affiliate_Commission_Percent"]:
    if c in df_channels.columns:
        df_channels[c] = pd.to_numeric(df_channels[c], errors="coerce").fillna(0.0)
if "Base_Conversion_Rate" in df_conv.columns:
    df_conv["Base_Conversion_Rate"] = pd.to_numeric(df_conv["Base_Conversion_Rate"], errors="coerce").fillna(0.0)
else:
    df_conv["Base_Conversion_Rate"] = 0.0
if "QR_Uplift_X" not in df_conv.columns:
    df_conv["QR_Uplift_X"] = 1.0

# ---------------- Build modeling frame ----------------
base = (df_cpmap
        .merge(df_titles, on="Title_ID", how="left", suffixes=("", "_title"))
        .merge(df_products, on="SKU", how="left", suffixes=("", "_prod")))

# unify Franchise and Title_Name columns (prefer product for merch grouping)
if "Franchise" in base.columns:
    base["Franchise_unified"] = base["Franchise"]
elif "Franchise_prod" in base.columns:
    base["Franchise_unified"] = base["Franchise_prod"]
elif "Franchise_title" in base.columns:
    base["Franchise_unified"] = base["Franchise_title"]
else:
    base["Franchise_unified"] = ""

if "Title_Name" not in base.columns and "Title_Name_title" in base.columns:
    base["Title_Name"] = base["Title_Name_title"]

views_agg = df_views.groupby("Title_ID", as_index=False).agg(
    Viewers_Exposed=("Viewers_Exposed","sum"),
    Ad_Impressions=("Ad_Impressions","sum")
)
base = base.merge(views_agg, on="Title_ID", how="left")

# Cross-join channels
df_channels = df_channels.copy()
df_channels["_k"] = 1
base["_k"] = 1
model = base.merge(df_channels, on="_k").drop(columns="_k")

# Attach conversion assumptions
model = model.merge(df_conv, on="Channel", how="left", suffixes=("","_conv"))

# ---- Effective conversion (safe) ----
conv = (model["Base_Conversion_Rate"].astype(float).fillna(0.0) * conv_mult_all)
is_qr = model["Channel"].str.lower().eq("disney+ qr")
conv = np.where(is_qr, conv * qr_uplift, conv)
rel = model.get("Relevance_Score")
if rel is not None:
    rel = pd.to_numeric(rel, errors="coerce").fillna(0.0)
    conv = np.where(rel >= 0.9, conv * dsp_uplift, conv)
conv = pd.to_numeric(pd.Series(conv), errors="coerce").fillna(0.0)
model["Conversion_Effective_Base"] = np.maximum(conv.values, 0.0)

# ---- Base exposure & units ----
model["Effective_Viewers_Base"] = model["Viewers_Exposed"].fillna(0) * ad_alloc
model["Units_Sold_Raw_Base"] = model["Effective_Viewers_Base"] * model["Conversion_Effective_Base"]
if respect_inventory:
    model["Units_Sold_Base"] = np.minimum(model["Units_Sold_Raw_Base"], model["Inventory_Level"].fillna(0))
else:
    model["Units_Sold_Base"] = model["Units_Sold_Raw_Base"]

# ---- Fees & revenue (base) ----
if apply_amazon_fee_override:
    amazon_fee_eff = np.where(model["Channel"].str.lower().eq("amazon"), 0.30, 0.0)
else:
    amazon_fee_eff = np.where(model["Channel"].str.lower().eq("amazon"),
                              model.get("Amazon_Fee_Percent", pd.Series(0, index=model.index)).fillna(0.0),
                              0.0)
model["Amazon_Fee_Effective"] = amazon_fee_eff
price = model["Price"].fillna(0.0)
cost = model["Cost"].fillna(0.0)
model["Revenue_Gross_Base"] = model["Units_Sold_Base"] * price
model["COGS_Base"] = model["Units_Sold_Base"] * cost
model["Amazon_Fees_Base"] = model["Units_Sold_Base"] * price * model["Amazon_Fee_Effective"]
model["Affiliate_Rebate_Base"] = np.where(
    (model["Channel"].str.lower().eq("amazon")) & affiliate_on,
    model["Units_Sold_Base"] * price * model.get("Affiliate_Commission_Percent", pd.Series(0, index=model.index)).fillna(0.0),
    0.0
)
model["Revenue_Net_Base"] = model["Revenue_Gross_Base"] - model["COGS_Base"] - model["Amazon_Fees_Base"] + model["Affiliate_Rebate_Base"]

# ---- Status Quo (external ads) from CPM × impressions ----
total_impressions = float(df_views["Ad_Impressions"].fillna(0).sum())
statusquo_net = (total_impressions / 1000.0) * cpm

# ---------------- Scene triggers (optional file) ----------------
if df_scenes is not None and not df_scenes.empty:
    st.markdown("### 🎯 Scene Triggers Applied (from scenes_normalized.csv)")
    for c in ["Extra_Minutes","Conv_Uplift_X","Slot_Fill_Percent"]:
        if c in df_scenes.columns:
            df_scenes[c] = pd.to_numeric(df_scenes[c], errors="coerce").fillna(0.0)

    def parse_channels(val):
        if pd.isna(val): return ["Amazon","ShopDisney","Disney+ QR"]
        if isinstance(val, list): return val
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list): return parsed
        except Exception:
            pass
        return [x.strip() for x in str(val).split(",") if x.strip()]

    if "Apply_To_Channels" in df_scenes.columns:
        df_scenes["Apply_To_List"] = df_scenes["Apply_To_Channels"].apply(parse_channels)
    else:
        df_scenes["Apply_To_List"] = [["Amazon","ShopDisney","Disney+ QR"]] * len(df_scenes)

    # Compute increments row-wise
    incr_rows = []
    for _, s in df_scenes.iterrows():
        franch = s.get("Franchise", None)
        extra_minutes = float(s.get("Extra_Minutes", 0.0))
        uplift = float(s.get("Conv_Uplift_X", 1.0))
        fill = float(s.get("Slot_Fill_Percent", 1.0))
        apply_list = s.get("Apply_To_List", ["Amazon","ShopDisney","Disney+ QR"])

        extra_slots = max(extra_minutes, 0.0) * 2.0  # 30s spots
        mask = model["Channel"].isin(apply_list)
        if franch:
            # match unified franchise
            if "Franchise_unified" in model.columns:
                mask &= model["Franchise_unified"].astype(str).str.contains(franch, case=False, na=False)
            elif "Franchise" in model.columns:
                mask &= model["Franchise"].astype(str).str.contains(franch, case=False, na=False)

        extra_exposed = model.loc[mask, "Viewers_Exposed"].fillna(0) * ad_alloc * fill * extra_slots
        incr_units = extra_exposed * model.loc[mask, "Conversion_Effective_Base"] * uplift

        inv_left = np.maximum(model.loc[mask, "Inventory_Level"].fillna(0) - model.loc[mask, "Units_Sold_Base"], 0)
        incr_units_capped = np.minimum(incr_units, inv_left) if respect_inventory else incr_units

        pr = model.loc[mask, "Price"].fillna(0.0)
        co = model.loc[mask, "Cost"].fillna(0.0)
        fees = np.where(model.loc[mask, "Channel"].str.lower()=="amazon",
                        incr_units_capped * pr * model.loc[mask, "Amazon_Fee_Effective"], 0.0)
        rebate = np.where(model.loc[mask, "Channel"].str.lower()=="amazon",
                          incr_units_capped * pr * model.get("Affiliate_Commission_Percent", pd.Series(0, index=model.index)).loc[mask].fillna(0.0) if affiliate_on else 0.0,
                          0.0)
        rev_n = incr_units_capped * pr - incr_units_capped * co - fees + rebate

        tmp = pd.DataFrame({
            "idx": model.loc[mask].index,
            "Incremental_Units": incr_units_capped,
            "Revenue_Net_Incr": rev_n
        })
        incr_rows.append(tmp)

    incr_df = pd.concat(incr_rows, ignore_index=True) if incr_rows else pd.DataFrame(columns=["idx","Incremental_Units","Revenue_Net_Incr"])
    model["Incremental_Units"] = 0.0
    model["Revenue_Net_Incr"] = 0.0
    if not incr_df.empty:
        inc_sum = incr_df.groupby("idx").sum(numeric_only=True)
        model.loc[inc_sum.index, "Incremental_Units"] = inc_sum["Incremental_Units"]
        model.loc[inc_sum.index, "Revenue_Net_Incr"] = inc_sum["Revenue_Net_Incr"]

    model["Units_Sold_Final"] = model["Units_Sold_Base"] + model["Incremental_Units"]
    model["Revenue_Net_Final"] = model["Revenue_Net_Base"] + model["Revenue_Net_Incr"]
else:
    model["Incremental_Units"] = 0.0
    model["Units_Sold_Final"] = model["Units_Sold_Base"]
    model["Revenue_Net_Final"] = model["Revenue_Net_Base"]

# ---------------- KPIs ----------------
total_net_base = float(model["Revenue_Net_Base"].sum())
total_net_final = float(model["Revenue_Net_Final"].sum())
incr_units_total = float(model["Incremental_Units"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Contextual Merch — Base ($)", f"{total_net_base:,.0f}")
c2.metric("Contextual Merch — With Scenes ($)", f"{total_net_final:,.0f}", delta=f"{(total_net_final-total_net_base):,.0f}")
c3.metric("Extra Units from Scene Triggers", f"{incr_units_total:,.0f}")

# ---------------- Comparison Chart (with labels & uplift) ----------------
compare_df = pd.DataFrame({
    "Scenario": ["Current Ads", "Contextual Ads", "Scene-Triggered Ads"],
    "Net_Revenue": [float(statusquo_net), float(total_net_base), float(total_net_final)]
}).set_index("Scenario")

st.markdown("### Impact: Current vs Contextual vs Scene-Triggered")

fig, ax = plt.subplots(figsize=(7.5, 4.8))

bars = ax.bar(
    compare_df.index,
    compare_df["Net_Revenue"],
    color=["#9aa0a6", "#4169e1", "#f4c430"],  # gray, royal blue, gold
    width=0.62,
)

ax.set_ylabel("Net Revenue (USD)")
ax.set_title("Revenue per 1M Viewers", fontsize=14, pad=12)

# headroom so annotations don’t collide
ymax = float(compare_df["Net_Revenue"].max())
ax.set_ylim(0, ymax * 1.35)   # add ~35% headroom
ax.margins(x=0.10)             # a bit of side padding

# $ labels INSIDE bars (near the top, centered)
for bar, val in zip(bars, compare_df["Net_Revenue"]):
    label = f"${val/1_000:.0f}K" if val < 1_000_000 else f"${val/1_000_000:.1f}M"
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val * 0.97,                   # inside the bar
        label,
        ha="center", va="top",
        fontsize=11, fontweight="bold", color="black"
    )

# Uplift annotations vs Current (placed ABOVE bars with arrows)
current = float(compare_df.loc["Current Ads", "Net_Revenue"])
contextual = float(compare_df.loc["Contextual Ads", "Net_Revenue"])
scene = float(compare_df.loc["Scene-Triggered Ads", "Net_Revenue"])

if current > 0:
    uplift_ctx_pct = (contextual - current) / current
    uplift_scene_pct = (scene - current) / current
    scene_factor = scene / current

    x_positions = [b.get_x() + b.get_width()/2 for b in bars]
    heights    = [b.get_height() for b in bars]

    # Contextual callout
    ax.annotate(
        f"+{uplift_ctx_pct*100:.0f}% vs Current",
        xy=(x_positions[1], heights[1]), xycoords="data",
        xytext=(0, 14), textcoords="offset points",
        ha="center", va="bottom", fontsize=10, color="#4169e1",
        arrowprops=dict(arrowstyle="-", color="#4169e1", lw=1)
    )

    # Scene callout (percent + factor)
    ax.annotate(
        f"+{uplift_scene_pct*100:.0f}% vs Current  •  {scene_factor:.1f}×",
        xy=(x_positions[2], heights[2]), xycoords="data",
        xytext=(0, 16), textcoords="offset points",
        ha="center", va="bottom", fontsize=10, color="#b8860b",
        arrowprops=dict(arrowstyle="-", color="#b8860b", lw=1)
    )

# Tidy up ticks/fonts
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10)

plt.subplots_adjust(top=0.86)  # extra top padding for title + callouts
st.pyplot(fig)

# ---------------- Channel summary ----------------
st.markdown("### Channel Performance")
chan = (model.groupby("Channel", as_index=False)
        .agg(Net_Base=("Revenue_Net_Base","sum"),
             Net_Final=("Revenue_Net_Final","sum"),
             Units_Base=("Units_Sold_Base","sum"),
             Units_Final=("Units_Sold_Final","sum")))
st.dataframe(chan, use_container_width=True)

# ---------------- Drilldowns ----------------
col1, col2 = st.columns(2)
with col1:
    titles_list = sorted(model.get("Title_Name", pd.Series(dtype=str)).dropna().unique().tolist())
    if titles_list:
        pick_title = st.selectbox("🎬 Drill by Title", titles_list)
        df_t = model[model.get("Title_Name", "") == pick_title]
        if not df_t.empty:
            st.bar_chart(df_t.set_index("Channel")[["Revenue_Net_Base","Revenue_Net_Final"]])

with col2:
    # Use unified franchise if present
    fr_col = "Franchise_unified" if "Franchise_unified" in model.columns else ("Franchise" if "Franchise" in model.columns else None)
    if fr_col:
        fr_list = sorted(model[fr_col].dropna().astype(str).unique().tolist())
        if fr_list:
            pick_fr = st.selectbox("🧸 Drill by Franchise", fr_list)
            df_f = model[model[fr_col].astype(str) == str(pick_fr)]
            if not df_f.empty:
                st.bar_chart(df_f.set_index("Channel")[["Revenue_Net_Base","Revenue_Net_Final"]])

# ---------------- Detail table ----------------
st.markdown("### SKU × Title × Channel Detail")
cols = ["SKU","Title_Name","Channel",
        "Price","Cost","Inventory_Level",
        "Conversion_Effective_Base",
        "Units_Sold_Base","Incremental_Units","Units_Sold_Final",
        "Revenue_Net_Base","Revenue_Net_Final",
        "Franchise_unified"]
have = [c for c in cols if c in model.columns]
st.dataframe(model[have].sort_values(["Title_Name","SKU","Channel"]).reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.caption(
    "All CSVs are read from the repo. Status quo uses CPM × total impressions. "
    "Contextual paths compute conversions by channel, fees, and inventory. "
    "Optional scenes_normalized.csv adds extra minutes → ad slots (2 × 30s) with uplift and fill, capped by inventory."
)
