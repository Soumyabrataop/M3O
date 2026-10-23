"""
M3O Streamlit in Snowflake (SiS) Dashboard.

This script is designed to run natively inside Snowflake's Streamlit environment.
It connects directly to the active session via Snowpark, reads from
M3O_ENGINE.TELEMETRY.LAYER_METRICS, and renders real-time collapse analytics,
drift charts, and layer telemetry.
"""

import streamlit as st
import pandas as pd
import altair as alt

# Native Snowpark session inside Snowflake
from snowflake.snowpark.context import get_active_session

# Page setup
st.set_page_config(
    page_title="M3O • Model Surgery Cockpit (Snowflake)",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ M3O: Mechanistic Model Merger & Optimizer")
st.markdown("**Native Streamlit in Snowflake (SiS) Analytics Cockpit** • `M3O_ENGINE.TELEMETRY.LAYER_METRICS`")

session = get_active_session()

# Header controls
col_info, col_refresh = st.columns([5, 1])
with col_info:
    st.caption("Live streaming analytics tracking layer-by-layer task vector surgery on Gemma 4.")
with col_refresh:
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()


@st.cache_data(ttl=3)
def fetch_telemetry_data():
    sql = """
    SELECT 
        LAYER_NAME,
        ALPHA,
        FROBENIUS_DRIFT,
        COS_SIM,
        STATUS,
        TIMESTAMP
    FROM M3O_ENGINE.TELEMETRY.LAYER_METRICS
    ORDER BY TIMESTAMP DESC
    """
    return session.sql(sql).to_pandas()


try:
    df = fetch_telemetry_data()
except Exception as e:
    st.error(f"Error querying M3O_ENGINE.TELEMETRY.LAYER_METRICS: {e}")
    df = pd.DataFrame()

if not df.empty:
    # Metadata extraction
    def parse_metadata(row):
        name = row["LAYER_NAME"]
        parts = name.split(".")
        idx = -1
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                break
        
        proj = "other"
        if "self_attn.q_proj" in name:
            proj = "self_attn.q_proj"
        elif "self_attn.v_proj" in name:
            proj = "self_attn.v_proj"
        elif "mlp.gate_proj" in name:
            proj = "mlp.gate_proj"
        elif "mlp.down_proj" in name:
            proj = "mlp.down_proj"
        return pd.Series([idx, proj])

    df[["LAYER_INDEX", "PROJECTION"]] = df.apply(parse_metadata, axis=1)

    # Executive KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    total_records = len(df)
    mean_cos = df["COS_SIM"].mean()
    min_cos = df["COS_SIM"].min()
    mean_frob = df["FROBENIUS_DRIFT"].mean()
    latest_alpha = df["ALPHA"].iloc[0] if total_records > 0 else 0.5

    stable_cnt = (df["STATUS"] == "STABLE").sum()
    degraded_cnt = (df["STATUS"] == "DEGRADED").sum()
    critical_cnt = (df["STATUS"] == "CRITICAL").sum()

    overall_health = "STABLE" if min_cos >= 0.85 else ("DEGRADED" if min_cos >= 0.65 else "CRITICAL")
    health_color = "normal" if overall_health == "STABLE" else "inverse"

    with k1:
        st.metric("Model Health", overall_health, delta="Normal" if overall_health == "STABLE" else "Alert", delta_color=health_color)
    with k2:
        st.metric("Total Layer Telemetry", f"{total_records} rows", f"Latest α = {latest_alpha}")
    with k3:
        st.metric("Mean Cosine Sim", f"{mean_cos:.6f}", f"Min: {min_cos:.6f}")
    with k4:
        st.metric("Mean Frobenius Δ", f"{mean_frob:.6f}")
    with k5:
        st.metric("Health Breakdown", f"{stable_cnt} Stable", f"{degraded_cnt} Deg / {critical_cnt} Crit")

    st.markdown("---")

    # Analytics Tabs
    t_charts, t_table = st.tabs(["📈 Drift & Collapse Curves", "📋 Warehouse Explorer"])

    with t_charts:
        st.subheader("Cosine Similarity across Network Depth")
        chart_df = df[df["LAYER_INDEX"] >= 0].sort_values(by=["LAYER_INDEX", "PROJECTION"])

        if not chart_df.empty:
            cos_chart = alt.Chart(chart_df).mark_line(point=True).encode(
                x=alt.X("LAYER_INDEX:O", title="Layer Index (Depth 0 → 34)"),
                y=alt.Y("COS_SIM:Q", scale=alt.Scale(domain=[min(0.7, min_cos - 0.05), 1.002]), title="Cosine Similarity"),
                color=alt.Color("PROJECTION:N", title="Projection Target"),
                tooltip=["LAYER_NAME", "PROJECTION", "LAYER_INDEX", "COS_SIM", "FROBENIUS_DRIFT", "STATUS"],
            ).properties(height=360)

            rule_deg = alt.Chart(pd.DataFrame({'y': [0.85]})).mark_rule(color='#FFD600', strokeDash=[4, 4]).encode(y='y:Q')
            rule_crit = alt.Chart(pd.DataFrame({'y': [0.65]})).mark_rule(color='#FF5252', strokeDash=[4, 4]).encode(y='y:Q')

            st.altair_chart(cos_chart + rule_deg + rule_crit, use_container_width=True)
            st.caption("Thresholds: Yellow dashed = Degraded Boundary (0.85). Red dashed = Critical Collapse Boundary (0.65).")

            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Frobenius Drift Magnitude")
                frob_chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("LAYER_INDEX:O", title="Layer Depth"),
                    y=alt.Y("FROBENIUS_DRIFT:Q", title="Frobenius Norm Δ"),
                    color=alt.Color("PROJECTION:N"),
                    tooltip=["LAYER_INDEX", "PROJECTION", "FROBENIUS_DRIFT"],
                ).properties(height=280)
                st.altair_chart(frob_chart, use_container_width=True)

            with c_right:
                st.subheader("Integrity Distribution")
                stat_df = df["STATUS"].value_counts().reset_index()
                stat_df.columns = ["STATUS", "COUNT"]
                pie = alt.Chart(stat_df).mark_arc(innerRadius=45).encode(
                    theta=alt.Theta(field="COUNT", type="quantitative"),
                    color=alt.Color(field="STATUS", type="nominal", scale=alt.Scale(
                        domain=["STABLE", "DEGRADED", "CRITICAL"],
                        range=["#00E676", "#FFD600", "#FF5252"]
                    )),
                    tooltip=["STATUS", "COUNT"],
                ).properties(height=280)
                st.altair_chart(pie, use_container_width=True)

    with t_table:
        st.subheader("Live Telemetry Records in Snowflake")
        search = st.text_input("Filter by layer substring...", "")
        view_df = df
        if search:
            view_df = view_df[view_df["LAYER_NAME"].str.contains(search, case=False)]

        st.dataframe(
            view_df[["LAYER_NAME", "PROJECTION", "ALPHA", "FROBENIUS_DRIFT", "COS_SIM", "STATUS", "TIMESTAMP"]],
            use_container_width=True,
            height=400,
        )

else:
    st.info("No records found in M3O_ENGINE.TELEMETRY.LAYER_METRICS yet. Run `python ui/cli.py --alpha 0.5` on your machine to stream telemetry into this table!")
