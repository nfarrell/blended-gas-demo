"""
Gas Blend Optimizer – NG + H₂ + Biomethane
Mock Digital Twin for Gas Network Blending Optimization

Run with:  streamlit run app.py
Requires:  pip install streamlit plotly pandas pulp
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import warnings
from datetime import datetime, timedelta

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
import streamlit as st


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour to an rgba() string accepted by Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gas Blend Optimizer",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (energy-sector blue / green theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d2137 0%, #103a5e 100%);
        border: 1px solid #1a5276;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: #85c1e9 !important; font-size: 0.8rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.6rem !important; font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #0d2137; border-radius: 6px 6px 0 0;
        color: #85c1e9; padding: 8px 22px;
    }
    .stTabs [aria-selected="true"] { background: #1a5276 !important; color: white !important; }

    [data-testid="stSidebar"] { background-color: #0a1929; }
    [data-testid="stSidebar"] .stSlider label { color: #85c1e9 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (realistic industry values)
# ─────────────────────────────────────────────────────────────────────────────
GAS = {
    "NG": {
        "cv":      39.5,   # Gross calorific value  MJ/m³
        "wobbe":   51.2,   # Wobbe Index            MJ/m³
        "density": 0.800,  # kg/m³  (at STP)
        "co2":     0.202,  # tCO₂ per MWh delivered
        "color":   "#1565C0",
        "label":   "Natural Gas",
    },
    "H2": {
        "cv":      10.8,
        "wobbe":   48.2,
        "density": 0.090,
        "co2":     0.000,  # green hydrogen assumed
        "color":   "#2E7D32",
        "label":   "Hydrogen",
    },
    "Bio": {
        "cv":      38.5,   # biomethane – within NG quality band
        "wobbe":   49.5,
        "density": 0.770,
        "co2":     0.000,  # carbon-neutral biomethane
        "color":   "#558B2F",
        "label":   "Biomethane",
    },
}

# Gas quality specification limits
# Lower CV bound is relaxed to 30 MJ/m³ to reflect H₂-ready / industrial networks;
# Wobbe band follows EN 16726 Group H guidance.
WOBBE_MIN, WOBBE_MAX = 46.0, 52.0   # MJ/m³
CV_MIN,    CV_MAX    = 30.0, 45.0   # MJ/m³
NG_CO2               = 0.202        # tCO₂/MWh – baseline comparison only


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def optimise_blend(
    n_hours:       int,
    hourly_demand: float,   # MWh/h (average; load shape applied internally)
    max_h2_pct:    float,   # vol%
    ren_pct:       float,   # energy-basis renewable target %
    cost_ng:       float,   # $/GJ
    cost_h2:       float,
    cost_bio:      float,
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Solve an LP for every hour of the horizon using PuLP / CBC.

    Decision variables per hour
    ─────────────────────────────
        q_ng, q_h2, q_bio  [m³/h]  – volumetric injection rate

    Objective
    ─────────
        Minimise total gas procurement cost

    Constraints
    ───────────
        C1  Energy balance  : blended energy ≥ demand
        C2  H₂ volume cap   : q_h2 / q_total ≤ max_h2_pct / 100
        C3  Renewable quota : (E_h2 + E_bio) / E_total ≥ ren_pct / 100
        C4  Wobbe Index     : WOBBE_MIN ≤ volume-weighted Wobbe ≤ WOBBE_MAX
        C5  Calorific Value : blended CV ≥ CV_MIN
    """
    np.random.seed(seed)
    cv_ng,  cv_h2,  cv_bio = GAS["NG"]["cv"],    GAS["H2"]["cv"],    GAS["Bio"]["cv"]
    w_ng,   w_h2,   w_bio  = GAS["NG"]["wobbe"],  GAS["H2"]["wobbe"],  GAS["Bio"]["wobbe"]

    # Sinusoidal load profile + small random noise
    hod     = np.arange(n_hours) % 24
    shape   = 0.65 + 0.35 * np.sin(np.pi * (hod - 6) / 12)
    shape   = np.clip(shape, 0.45, 1.40)
    demands = hourly_demand * shape * (1.0 + 0.04 * np.random.randn(n_hours))
    demands = np.maximum(demands, 5.0)

    rows = []
    for i in range(n_hours):
        d_mj = float(demands[i]) * 3_600.0   # MWh → MJ  (1 MWh = 3 600 MJ)

        prob  = pulp.LpProblem(f"blend_h{i}", pulp.LpMinimize)
        q_ng  = pulp.LpVariable("q_ng",  lowBound=0)
        q_h2  = pulp.LpVariable("q_h2",  lowBound=0)
        q_bio = pulp.LpVariable("q_bio", lowBound=0)

        # Objective – cost_X in $/GJ; q_X * cv_X / 1 000 = GJ delivered by source X
        prob += (
            cost_ng  * (q_ng  * cv_ng  / 1_000.0) +
            cost_h2  * (q_h2  * cv_h2  / 1_000.0) +
            cost_bio * (q_bio * cv_bio / 1_000.0)
        )

        # C1: meet demand
        prob += (q_ng * cv_ng + q_h2 * cv_h2 + q_bio * cv_bio >= d_mj)

        # C2: H₂ volume fraction  q_h2/(q_ng+q_h2+q_bio) ≤ max_h2_pct/100
        #   → q_h2*(100-max_h2_pct) ≤ max_h2_pct*(q_ng+q_bio)
        if max_h2_pct < 100:
            prob += (q_h2 * (100 - max_h2_pct) <= max_h2_pct * (q_ng + q_bio))

        # C3: renewable energy fraction  (E_h2+E_bio)/E_total ≥ ren_pct/100
        #   → (E_h2+E_bio)*(100-ren_pct) ≥ ren_pct*E_ng
        if 0 < ren_pct < 100:
            prob += (
                (q_h2 * cv_h2 + q_bio * cv_bio) * (100 - ren_pct) >=
                ren_pct * (q_ng * cv_ng)
            )
        elif ren_pct >= 100:
            prob += (q_ng == 0)

        # C4: Wobbe Index – linear volume-weighted approximation
        q_tot = q_ng + q_h2 + q_bio
        prob += (q_ng * w_ng + q_h2 * w_h2 + q_bio * w_bio >= WOBBE_MIN * q_tot)
        prob += (q_ng * w_ng + q_h2 * w_h2 + q_bio * w_bio <= WOBBE_MAX * q_tot)

        # C5: Blended CV lower bound
        prob += (q_ng * cv_ng + q_h2 * cv_h2 + q_bio * cv_bio >= CV_MIN * q_tot)

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        qng  = max(pulp.value(q_ng)  or 0.0, 0.0)
        qh2  = max(pulp.value(q_h2)  or 0.0, 0.0)
        qbio = max(pulp.value(q_bio) or 0.0, 0.0)
        qtot = qng + qh2 + qbio + 1e-12

        e_ng  = qng  * cv_ng  / 3_600.0   # MWh
        e_h2  = qh2  * cv_h2  / 3_600.0
        e_bio = qbio * cv_bio / 3_600.0
        etot  = e_ng + e_h2 + e_bio + 1e-12

        blend_cv    = (qng * cv_ng  + qh2 * cv_h2  + qbio * cv_bio) / qtot
        blend_wobbe = (qng * w_ng   + qh2 * w_h2   + qbio * w_bio)  / qtot

        cost = (
            cost_ng  * (qng  * cv_ng  / 1_000.0) +
            cost_h2  * (qh2  * cv_h2  / 1_000.0) +
            cost_bio * (qbio * cv_bio / 1_000.0)
        )

        rows.append({
            "hour":           i,
            "timestamp":      None,
            "demand_mwh":     float(demands[i]),
            "q_ng_m3h":       qng,
            "q_h2_m3h":       qh2,
            "q_bio_m3h":      qbio,
            "e_ng_mwh":       e_ng,
            "e_h2_mwh":       e_h2,
            "e_bio_mwh":      e_bio,
            "cost_usd":       cost,
            "ren_pct":        (e_h2 + e_bio) / etot * 100,
            "h2_vol_pct":     qh2 / qtot * 100,
            "blended_cv":     blend_cv,
            "blended_wobbe":  blend_wobbe,
            "emissions_tco2": e_ng * GAS["NG"]["co2"],
            "wobbe_ok":       WOBBE_MIN <= blend_wobbe <= WOBBE_MAX,
            "cv_ok":          blend_cv >= CV_MIN,
            "lp_status":      pulp.LpStatus[status],
        })

    return pd.DataFrame(rows)


def compute_baseline(df_opt: pd.DataFrame, cost_ng: float) -> pd.DataFrame:
    """100 % NG baseline – same demand profile, no blending."""
    rows = []
    for _, r in df_opt.iterrows():
        q_ng = r["demand_mwh"] * 3_600.0 / GAS["NG"]["cv"]
        cost = cost_ng * (q_ng * GAS["NG"]["cv"] / 1_000.0)
        rows.append({
            "cost_usd":       cost,
            "emissions_tco2": r["demand_mwh"] * NG_CO2,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sensitivity_sweep(hourly_demand: float, cost_ng: float, cost_h2: float, cost_bio: float):
    """
    24-hour proxy runs across H₂ limit and renewable target ranges.
    Cached on cost inputs; re-runs only when costs change.
    """
    h2_range  = list(range(0, 21, 2))
    ren_range = list(range(0, 101, 10))

    h2_costs, h2_ren_ach = [], []
    for h2_max in h2_range:
        try:
            d    = optimise_blend(24, hourly_demand, h2_max, 30, cost_ng, cost_h2, cost_bio, seed=0)
            etot = d["e_ng_mwh"].sum() + d["e_h2_mwh"].sum() + d["e_bio_mwh"].sum()
            h2_costs.append(d["cost_usd"].sum())
            h2_ren_ach.append((d["e_h2_mwh"].sum() + d["e_bio_mwh"].sum()) / etot * 100)
        except Exception:
            h2_costs.append(float("nan"))
            h2_ren_ach.append(float("nan"))

    ren_costs, ren_co2 = [], []
    for ren in ren_range:
        try:
            d = optimise_blend(24, hourly_demand, 15, ren, cost_ng, cost_h2, cost_bio, seed=0)
            ren_costs.append(d["cost_usd"].sum())
            ren_co2.append(d["emissions_tco2"].sum())
        except Exception:
            ren_costs.append(float("nan"))
            ren_co2.append(float("nan"))

    return h2_range, h2_costs, h2_ren_ach, ren_range, ren_costs, ren_co2


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("⛽ Gas Blend Optimizer – NG + H₂ + Biomethane")
st.markdown("### Mock Digital Twin for Gas Network Blending Optimization")
st.markdown(
    """
    Gas distribution networks are under growing pressure to decarbonise by blending
    **Hydrogen (H₂)** and **Biomethane (Bio / RNG)** with conventional **Natural Gas (NG)**.
    This dashboard runs an **hourly Linear Programme (LP)** that minimises procurement cost
    while meeting customer energy demand, respecting H₂ safety limits and gas-quality standards
    (Wobbe Index, Calorific Value), and achieving a configurable renewable-content target.
    Adjust the sidebar controls and press **Run Optimization** to explore the solution space.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Scenario Controls")

    st.subheader("⏱ Time Horizon")
    days       = st.slider("Forecast window (days)", 1, 14, 7)
    n_hours    = days * 24
    start_date = st.date_input("Start date", value=datetime.today())

    st.subheader("🎯 Blend Targets")
    ren_target = st.slider(
        "Target renewable content (%)", 0, 100, 30, step=5,
        help="Minimum share of total energy that must come from H₂ or Biomethane.",
    )
    max_h2 = st.slider(
        "Maximum H₂ blend (vol%)", 0, 20, 15, step=1,
        help="Upper bound on hydrogen volume fraction (safety / appliance compatibility limit).",
    )

    st.subheader("⚡ Energy Demand")
    daily_demand  = st.slider(
        "Daily energy demand (MWh/day)", 500, 5000, 2500, step=100,
        help="Average daily energy consumption of the customers served by this network.",
    )
    hourly_demand = daily_demand / 24.0

    st.subheader("💰 Gas Costs ($/GJ)")
    cost_ng  = st.slider("Natural Gas cost",  1.0, 20.0,  8.0, step=0.5)
    cost_h2  = st.slider("Hydrogen cost",     1.0, 40.0, 18.0, step=0.5)
    cost_bio = st.slider("Biomethane cost",   1.0, 30.0, 14.0, step=0.5)

    st.divider()
    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN / CACHE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
cache_key = (n_hours, daily_demand, max_h2, ren_target, cost_ng, cost_h2, cost_bio, str(start_date))

if run_btn or "cache_key" not in st.session_state or st.session_state.cache_key != cache_key:
    with st.spinner("⚙️  Solving LP for all hours…"):
        df_opt = optimise_blend(n_hours, hourly_demand, max_h2, ren_target, cost_ng, cost_h2, cost_bio)
        timestamps = [
            datetime.combine(start_date, datetime.min.time()) + timedelta(hours=i)
            for i in range(n_hours)
        ]
        df_opt["timestamp"] = timestamps
        df_base = compute_baseline(df_opt, cost_ng)
        st.session_state.df_opt    = df_opt
        st.session_state.df_base   = df_base
        st.session_state.cache_key = cache_key

df   = st.session_state.df_opt
df_b = st.session_state.df_base

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview & Blend",
    "📋  Optimisation Results",
    "🔬  Gas Quality & Network",
    "📈  Sensitivity Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW & BLEND
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── KPI aggregates ────────────────────────────────────────────────────
    total_cost  = df["cost_usd"].sum()
    base_cost   = df_b["cost_usd"].sum()
    cost_saving = (base_cost - total_cost) / base_cost * 100

    e_ng_tot  = df["e_ng_mwh"].sum()
    e_h2_tot  = df["e_h2_mwh"].sum()
    e_bio_tot = df["e_bio_mwh"].sum()
    etot      = e_ng_tot + e_h2_tot + e_bio_tot + 1e-12

    ren_achieved = (e_h2_tot + e_bio_tot) / etot * 100
    co2_base     = df_b["emissions_tco2"].sum()
    co2_opt      = df["emissions_tco2"].sum()
    co2_avoided  = co2_base - co2_opt
    avg_cv       = df["blended_cv"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "💰 Total Cost (period)", f"${total_cost:,.0f}",
        delta=f"{cost_saving:+.1f}% vs 100% NG",
        help="Total gas procurement cost for the optimised blend over the forecast period.",
    )
    c2.metric(
        "♻️ Renewable Content", f"{ren_achieved:.1f}%",
        delta=f"Target: {ren_target}%",
        help="Share of total energy delivered that came from Hydrogen or Biomethane.",
    )
    c3.metric(
        "🌿 CO₂ Avoided", f"{co2_avoided:,.1f} tCO₂",
        delta=f"Baseline: {co2_base:,.0f} t",
        help="CO₂ reduction vs a pure-NG baseline, assuming green H₂ and carbon-neutral Biomethane.",
    )
    c4.metric(
        "🔥 Avg Calorific Value", f"{avg_cv:.2f} MJ/m³",
        delta=f"Spec: {CV_MIN}–{CV_MAX}",
        help="Volume-weighted average calorific value of the blended gas at injection.",
    )

    st.divider()

    # ── Stacked area – hourly blend ────────────────────────────────────────
    st.subheader("Hourly Blend Contribution (MWh/h)")
    st.caption(
        "Stacked bands show the energy contribution from each gas source per hour. "
        "The dotted white line is the actual demand profile."
    )

    fig_area = go.Figure()
    for src, col in [("NG", "e_ng_mwh"), ("H2", "e_h2_mwh"), ("Bio", "e_bio_mwh")]:
        fig_area.add_trace(go.Scatter(
            x=df["timestamp"], y=df[col],
            name=GAS[src]["label"],
            stackgroup="one",
            mode="none",
            fillcolor=hex_to_rgba(GAS[src]["color"], 0.8),
            hovertemplate=f"{GAS[src]['label']}: %{{y:.1f}} MWh/h<extra></extra>",
        ))
    fig_area.add_trace(go.Scatter(
        x=df["timestamp"], y=df["demand_mwh"],
        name="Demand", mode="lines",
        line=dict(color="white", width=1.5, dash="dot"),
        hovertemplate="Demand: %{y:.1f} MWh/h<extra></extra>",
    ))
    fig_area.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="Time", yaxis_title="Energy (MWh/h)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # ── Pie + summary table ────────────────────────────────────────────────
    col_pie, col_tbl = st.columns([1, 1])

    with col_pie:
        st.subheader("Overall Blend Composition")
        st.caption("Share of total energy delivered over the selected period.")
        fig_pie = go.Figure(go.Pie(
            labels=[GAS[k]["label"] for k in ("NG", "H2", "Bio")],
            values=[e_ng_tot, e_h2_tot, e_bio_tot],
            hole=0.45,
            marker_colors=[GAS[k]["color"] for k in ("NG", "H2", "Bio")],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,.0f} MWh (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            template="plotly_dark", height=300,
            margin=dict(t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_tbl:
        st.subheader("📌 Period Summary")
        summary = {
            "Forecast period":           f"{days} day(s)  /  {n_hours} hours",
            "Total energy delivered":    f"{etot:,.0f} MWh",
            "NG share (energy)":         f"{e_ng_tot  / etot * 100:.1f}%",
            "H₂ share (energy)":         f"{e_h2_tot  / etot * 100:.1f}%",
            "Biomethane share (energy)": f"{e_bio_tot / etot * 100:.1f}%",
            "Avg H₂ vol%":               f"{df['h2_vol_pct'].mean():.1f}%",
            "Cost saving vs baseline":   f"${base_cost - total_cost:,.0f}  ({cost_saving:.1f}%)",
            "CO₂ avoided":               f"{co2_avoided:,.1f} tCO₂",
            "Wobbe compliance":          "✅ 100%" if df["wobbe_ok"].all()
                                         else f"⚠️ {df['wobbe_ok'].mean()*100:.0f}%",
            "CV compliance":             "✅ 100%" if df["cv_ok"].all()
                                         else f"⚠️ {df['cv_ok'].mean()*100:.0f}%",
        }
        df_sum = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
        st.dataframe(df_sum, hide_index=True, use_container_width=True, height=360)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – OPTIMISATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Hourly Optimised Injection Schedule")
    st.caption(
        "Each row is one hour of optimised operation. "
        "Green/red shading in the compliance columns flags gas-quality adherence."
    )

    display_df = df[[
        "timestamp", "demand_mwh",
        "q_ng_m3h", "q_h2_m3h", "q_bio_m3h",
        "h2_vol_pct", "ren_pct",
        "blended_cv", "blended_wobbe",
        "cost_usd", "wobbe_ok", "cv_ok",
    ]].copy()
    display_df.columns = [
        "Timestamp", "Demand (MWh/h)",
        "NG (m³/h)", "H₂ (m³/h)", "Bio (m³/h)",
        "H₂ vol%", "Renewable %",
        "CV (MJ/m³)", "Wobbe (MJ/m³)",
        "Cost ($)", "Wobbe ✓", "CV ✓",
    ]
    for c in ["Demand (MWh/h)", "NG (m³/h)", "H₂ (m³/h)", "Bio (m³/h)",
              "H₂ vol%", "Renewable %", "CV (MJ/m³)", "Wobbe (MJ/m³)", "Cost ($)"]:
        display_df[c] = display_df[c].round(2)

    def _bool_style(val):
        if val is True:
            return "background-color:#1b5e20; color:white"
        if val is False:
            return "background-color:#b71c1c; color:white"
        return ""

    styled = (
        display_df.style
        .background_gradient(subset=["H₂ vol%", "Renewable %"], cmap="YlGn")
        .background_gradient(subset=["Cost ($)"],                cmap="YlOrRd")
        .map(_bool_style, subset=["Wobbe ✓", "CV ✓"])
    )
    st.dataframe(styled, use_container_width=True, height=340)

    csv_buf = io.StringIO()
    display_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️  Download Optimised Schedule (CSV)",
        csv_buf.getvalue(),
        file_name=f"blend_optimised_{start_date}.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Optimised vs Baseline comparison ──────────────────────────────────
    st.subheader("Optimised vs Baseline (100% Natural Gas)")
    st.caption(
        "The baseline assumes all energy is supplied from Natural Gas at the same demand profile. "
        "The optimised blend reduces cost (when renewables are cheaper) and always cuts emissions."
    )

    df_cmp = df.copy()
    df_cmp["base_cost"]      = df_b["cost_usd"].values
    df_cmp["base_emissions"] = df_b["emissions_tco2"].values
    df_cmp["base_wobbe"]     = GAS["NG"]["wobbe"]

    fig_cmp = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Hourly Cost ($)", "CO₂ Emissions (tCO₂/h)", "Wobbe Index (MJ/m³)"],
        vertical_spacing=0.09,
    )
    pairs = [
        (1, "cost_usd",       "base_cost",      "Optimised",     "Baseline"),
        (2, "emissions_tco2", "base_emissions",  "Optimised",     "Baseline"),
        (3, "blended_wobbe",  "base_wobbe",      "Blended Wobbe", "Pure NG Wobbe"),
    ]
    for row, col_opt, col_bas, lbl_opt, lbl_bas in pairs:
        fig_cmp.add_trace(go.Scatter(
            x=df_cmp["timestamp"], y=df_cmp[col_opt],
            name=lbl_opt, line=dict(color="#4CAF50", width=2),
            hovertemplate=f"{lbl_opt}: %{{y:.2f}}<extra></extra>",
            showlegend=(row == 1),
        ), row=row, col=1)
        fig_cmp.add_trace(go.Scatter(
            x=df_cmp["timestamp"], y=df_cmp[col_bas],
            name=lbl_bas, line=dict(color="#1565C0", width=1.5, dash="dash"),
            hovertemplate=f"{lbl_bas}: %{{y:.2f}}<extra></extra>",
            showlegend=(row == 1),
        ), row=row, col=1)

    fig_cmp.add_hrect(
        y0=WOBBE_MIN, y1=WOBBE_MAX,
        fillcolor="rgba(76,175,80,0.1)", line_width=0,
        annotation_text=f"Spec {WOBBE_MIN}–{WOBBE_MAX} MJ/m³",
        annotation_position="top right", row=3, col=1,
    )
    fig_cmp.update_layout(
        template="plotly_dark", height=600,
        legend=dict(orientation="h", y=-0.04),
        margin=dict(t=40, b=60), hovermode="x unified",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Auto-generated logic explanation ──────────────────────────────────
    st.subheader("💡 Why Did the Optimiser Make These Choices?")
    h2_bind_pct  = (df["h2_vol_pct"] >= max_h2 * 0.95).mean() * 100
    ren_bind_pct = (df["ren_pct"]    <= ren_target * 1.05).mean() * 100
    cheap_ren    = "Hydrogen" if cost_h2 < cost_bio else "Biomethane"
    cheap_sym    = "H₂" if cost_h2 < cost_bio else "Bio"

    st.info(
        f"""
**Optimiser Logic Summary  ({n_hours}-hour window)**

- The LP minimised total procurement cost across **{n_hours} hourly intervals** with a sinusoidal demand profile.
- **Renewable source preference:** {cheap_ren} was cheaper at the current settings  \
(${cost_h2:.1f}/GJ for H₂ vs ${cost_bio:.1f}/GJ for Bio), so the renewable quota was filled primarily with {cheap_sym}.
- **H₂ volume cap ({max_h2}% vol):** Binding in **{h2_bind_pct:.0f}%** of hours — in those hours the \
optimizer wanted more H₂ but was constrained by the safety limit.
- **Renewable target ({ren_target}%):** Approximately binding in **{ren_bind_pct:.0f}%** of hours.
- **Peak demand** hours (midday): all three sources ramp up proportionally to meet higher load.
- **Off-peak** hours (night): volumes fall but the blend ratio stays stable because the renewable \
and H₂ constraints are relative (%), not absolute.
- **Gas quality:** Wobbe Index and CV remained within spec in \
{'all' if (df['wobbe_ok'] & df['cv_ok']).all() else 'most'} hours, \
confirming safe injection and appliance compatibility.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – GAS QUALITY & NETWORK
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # ── 5-node network diagram ─────────────────────────────────────────────
    st.subheader("🗺️ Simulated 5-Node Gas Network")
    st.caption(
        "Node colour indicates local H₂ concentration (vol%). "
        "Green border = within H₂ limit; red border = over limit."
    )

    NODE_POS = {
        "Injection\nPoint A":  (0.08, 0.75),
        "Junction 1":          (0.32, 0.60),
        "City Gate B":         (0.58, 0.78),
        "Industrial\nZone":    (0.55, 0.35),
        "Residential\nHub":    (0.82, 0.52),
    }
    EDGES = [
        ("Injection\nPoint A", "Junction 1"),
        ("Junction 1",         "City Gate B"),
        ("Junction 1",         "Industrial\nZone"),
        ("City Gate B",        "Residential\nHub"),
        ("Industrial\nZone",   "Residential\nHub"),
    ]

    h2_avg = float(df["h2_vol_pct"].mean())
    # Slight downstream attenuation (realistic linepack dilution)
    NODE_H2 = {
        "Injection\nPoint A":  h2_avg * 1.00,
        "Junction 1":          h2_avg * 0.97,
        "City Gate B":         h2_avg * 0.93,
        "Industrial\nZone":    h2_avg * 0.88,
        "Residential\nHub":    h2_avg * 0.84,
    }

    fig_net = go.Figure()

    for n1, n2 in EDGES:
        x0, y0 = NODE_POS[n1]
        x1, y1 = NODE_POS[n2]
        fig_net.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color="#37474F", width=8),
            hoverinfo="skip", showlegend=False,
        ))

    for name, (x, y) in NODE_POS.items():
        h2v = NODE_H2[name]
        ok  = h2v <= max_h2
        fig_net.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=52,
                color=h2v,
                colorscale="YlGn",
                cmin=0,
                cmax=max(max_h2, h2_avg * 1.1, 1.0),
                colorbar=dict(title="H₂ vol%", thickness=14, x=1.04),
                line=dict(color="#4CAF50" if ok else "#F44336", width=4),
                showscale=True,
            ),
            text=[f"<b>{name}</b><br>{h2v:.1f}%"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"H₂: {h2v:.1f}%  ({'✅ OK' if ok else '❌ Over limit'})"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_net.update_layout(
        template="plotly_dark", height=420,
        xaxis=dict(visible=False, range=[-0.05, 1.1]),
        yaxis=dict(visible=False, range=[0.15,  0.95]),
        margin=dict(t=10, b=10, l=10, r=100),
    )
    st.plotly_chart(fig_net, use_container_width=True)

    st.divider()

    # ── Quality time-series ────────────────────────────────────────────────
    st.subheader("Gas Quality Parameters over Time")
    st.caption(
        "Shaded green bands show the acceptable operating range for each parameter. "
        "Exceedances would trigger a blend re-dispatch in a live system."
    )

    fig_q = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            f"Calorific Value (MJ/m³)  │  spec: {CV_MIN}–{CV_MAX}",
            f"Wobbe Index (MJ/m³)  │  spec: {WOBBE_MIN}–{WOBBE_MAX}",
            f"H₂ Concentration (vol%)  │  limit: {max_h2}%",
        ],
        vertical_spacing=0.10,
    )

    fig_q.add_trace(go.Scatter(
        x=df["timestamp"], y=df["blended_cv"],
        name="Blended CV", line=dict(color="#42A5F5", width=2),
        hovertemplate="CV: %{y:.2f} MJ/m³<extra></extra>",
    ), row=1, col=1)
    fig_q.add_hrect(y0=CV_MIN, y1=CV_MAX, fillcolor="rgba(76,175,80,0.12)",
                    line_width=0, row=1, col=1)

    fig_q.add_trace(go.Scatter(
        x=df["timestamp"], y=df["blended_wobbe"],
        name="Wobbe Index", line=dict(color="#66BB6A", width=2),
        hovertemplate="Wobbe: %{y:.2f} MJ/m³<extra></extra>",
    ), row=2, col=1)
    fig_q.add_hrect(y0=WOBBE_MIN, y1=WOBBE_MAX, fillcolor="rgba(76,175,80,0.12)",
                    line_width=0, row=2, col=1)

    fig_q.add_trace(go.Scatter(
        x=df["timestamp"], y=df["h2_vol_pct"],
        name="H₂ vol%", line=dict(color="#AED581", width=2),
        fill="tozeroy", fillcolor="rgba(174,213,129,0.15)",
        hovertemplate="H₂: %{y:.2f} vol%<extra></extra>",
    ), row=3, col=1)
    fig_q.add_hline(
        y=max_h2, line_dash="dash", line_color="#EF5350",
        annotation_text=f"Limit {max_h2}%", annotation_position="top right",
        row=3, col=1,
    )

    fig_q.update_layout(
        template="plotly_dark", height=560,
        showlegend=False,
        margin=dict(t=40, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_q, use_container_width=True)

    # ── Compliance summary ─────────────────────────────────────────────────
    st.subheader("Compliance Summary")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(
        "CV Compliance",
        f"{df['cv_ok'].mean()*100:.1f}%",
        delta=f"{df['cv_ok'].sum()}/{len(df)} hours in-spec",
    )
    cc2.metric(
        "Wobbe Compliance",
        f"{df['wobbe_ok'].mean()*100:.1f}%",
        delta=f"{df['wobbe_ok'].sum()}/{len(df)} hours in-spec",
    )
    cc3.metric(
        "H₂ Limit Compliance",
        f"{(df['h2_vol_pct'] <= max_h2).mean()*100:.1f}%",
        delta=f"Avg H₂: {df['h2_vol_pct'].mean():.1f} vol%",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Sensitivity Analysis")
    st.caption(
        "Each point represents a 24-hour proxy optimisation at a different parameter value. "
        "These curves reveal how cost and renewable content trade off against controllable levers."
    )

    with st.spinner("Computing sensitivity curves…"):
        h2_range, h2_costs, h2_ren_ach, ren_range, ren_costs, ren_co2 = \
            sensitivity_sweep(hourly_demand, cost_ng, cost_h2, cost_bio)

    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("**Cost vs H₂ Volume Limit** *(renewable target fixed at 30%)*")
        st.caption("Easing the H₂ cap reduces cost only when H₂ is cheaper than NG — otherwise the constraint is non-binding.")
        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Scatter(
            x=h2_range, y=h2_costs,
            mode="lines+markers",
            line=dict(color="#42A5F5", width=2.5), marker=dict(size=7),
            hovertemplate="H₂ max: %{x}%<br>Daily cost: $%{y:,.0f}<extra></extra>",
        ))
        fig_s1.add_vline(x=max_h2, line_dash="dash", line_color="#EF5350",
                         annotation_text=f"Current: {max_h2}%", annotation_position="top left")
        fig_s1.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Max H₂ vol%", yaxis_title="Daily Cost ($)",
            margin=dict(t=10, b=40), showlegend=False,
        )
        st.plotly_chart(fig_s1, use_container_width=True)

    with sc2:
        st.markdown("**Cost vs Renewable Target** *(H₂ limit fixed at 15%)*")
        st.caption("Higher renewable requirements force more expensive H₂/Bio, raising total cost.")
        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Scatter(
            x=ren_range, y=ren_costs,
            mode="lines+markers",
            line=dict(color="#66BB6A", width=2.5), marker=dict(size=7),
            hovertemplate="Renewable target: %{x}%<br>Daily cost: $%{y:,.0f}<extra></extra>",
        ))
        fig_s2.add_vline(x=ren_target, line_dash="dash", line_color="#EF5350",
                         annotation_text=f"Current: {ren_target}%", annotation_position="top left")
        fig_s2.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Renewable Target (%)", yaxis_title="Daily Cost ($)",
            margin=dict(t=10, b=40), showlegend=False,
        )
        st.plotly_chart(fig_s2, use_container_width=True)

    st.divider()

    # ── Pareto front ───────────────────────────────────────────────────────
    st.subheader("Pareto Front: Cost vs Renewable Content")
    st.caption(
        "The curve sweeps the renewable target from 0 → 100%, tracing every cost–renewable trade-off. "
        "The ★ marks your current settings. Bubble colour encodes CO₂ emissions."
    )

    valid = [(r, c, e) for r, c, e in zip(ren_range, ren_costs, ren_co2)
             if not (np.isnan(c) or np.isnan(e))]

    if valid:
        pr, pc, pe = zip(*valid)
        fig_par = go.Figure()
        fig_par.add_trace(go.Scatter(
            x=list(pr), y=list(pc),
            mode="lines+markers",
            marker=dict(
                size=10,
                color=list(pe),
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="CO₂ (t/day)", thickness=14),
                line=dict(color="white", width=0.5),
            ),
            line=dict(color="#546E7A", width=1.5),
            text=[f"Ren: {r}%<br>Cost: ${c:,.0f}<br>CO₂: {e:.1f} t"
                  for r, c, e in zip(pr, pc, pe)],
            hovertemplate="%{text}<extra></extra>",
            name="Feasible solutions",
        ))
        cur_idx = min(range(len(pr)), key=lambda i: abs(pr[i] - ren_target))
        fig_par.add_trace(go.Scatter(
            x=[pr[cur_idx]], y=[pc[cur_idx]],
            mode="markers",
            marker=dict(size=18, color="#F44336", symbol="star",
                        line=dict(color="white", width=2)),
            name="Current setting",
            hoverinfo="skip",
        ))
        fig_par.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Renewable Content (%)", yaxis_title="Daily Cost ($)",
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_par, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️  About This Mock Dashboard", expanded=False):
    st.markdown(
        f"""
### About This Mock Digital Twin

This dashboard is a **simplified demonstration** of the real-world optimisation challenge faced by
gas distribution network operators as they transition away from 100% Natural Gas.

**What is modelled (simplified)**
- A **Linear Programme (LP)** using [PuLP](https://coin-or.github.io/pulp/) (CBC solver) minimises
  hourly gas procurement cost subject to energy demand, H₂ volume limits, renewable-content targets,
  and Wobbe/CV constraints.
- Gas properties (CV, Wobbe Index, density, CO₂ intensity) use realistic industry values:
  NG {GAS["NG"]["cv"]} MJ/m³ / {GAS["NG"]["wobbe"]} MJ/m³ Wobbe,
  H₂ {GAS["H2"]["cv"]} MJ/m³ / {GAS["H2"]["wobbe"]} MJ/m³ Wobbe,
  Bio {GAS["Bio"]["cv"]} MJ/m³ / {GAS["Bio"]["wobbe"]} MJ/m³ Wobbe.
- Hourly demand follows a sinusoidal day/night load profile with ±4% random noise.
- The 5-node network is **topological only** — pressures and directional flows are not simulated.
- The CV lower bound is relaxed to {CV_MIN} MJ/m³ to represent a transitional, H₂-ready network
  where customer appliances have been upgraded or are in industrial applications.

**What is NOT modelled (required in a production system)**
- Hydraulic pressure/flow simulation (Weymouth / Panhandle equations)
- Transient gas mixing, linepack, and residence-time effects
- Gas chromatograph measurement uncertainty and lag
- Per-appliance compatibility limits for H₂ blends
- Market price forecasts, forward contracts, and balancing costs
- Regulatory permit constraints by geographic zone or pipeline segment
- Compressor fuel consumption and station-level losses

**Technologies used:** Python · Streamlit · Plotly · Pandas · PuLP (CBC solver)

*This dashboard is a proof-of-concept for demonstration purposes only. It does not represent the
views, systems, or data of any specific gas network operator. All values are synthetic.*
        """
    )
