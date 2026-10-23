"""
Streamlit Dashboard for M3O (Mechanistic Model Merger & Optimizer).

Provides interactive controls for lambda tuning, real-time streaming progress,
model collapse and integrity curves, layer-wise matrix analytics, and dual telemetry inspection.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import pandas as pd
import streamlit as st
import altair as alt

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from engine.merger import (
    WeightSurgeryEngine,
    STATUS_STABLE,
    STATUS_DEGRADED,
    STATUS_CRITICAL,
    classify_status,
)
from telemetry.logger import TelemetryLogger

# Page configuration
st.set_page_config(
    page_title="M3O • Mechanistic Model Merger & Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-tech styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E232A;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2E3642;
    }
    .status-badge-stable {
        color: #00E676;
        font-weight: bold;
        background-color: rgba(0, 230, 118, 0.15);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #00E676;
    }
    .status-badge-degraded {
        color: #FFD600;
        font-weight: bold;
        background-color: rgba(255, 214, 0, 0.15);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #FFD600;
    }
    .status-badge-critical {
        color: #FF5252;
        font-weight: bold;
        background-color: rgba(255, 82, 82, 0.15);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #FF5252;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_layer_metadata(layer_name: str):
    """Extract layer index and projection type from tensor name."""
    parts = layer_name.split(".")
    layer_idx = -1
    proj_type = "unknown"
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break

    if "self_attn.q_proj" in layer_name:
        proj_type = "self_attn.q_proj"
    elif "self_attn.v_proj" in layer_name:
        proj_type = "self_attn.v_proj"
    elif "mlp.gate_proj" in layer_name:
        proj_type = "mlp.gate_proj"
    elif "mlp.down_proj" in layer_name:
        proj_type = "mlp.down_proj"

    return layer_idx, proj_type


# -----------------------------------------------------------------------------
# Sidebar: Parameter & Surgery Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ M3O Controls")
st.sidebar.markdown("**Mechanistic Model Surgery & Optimizer**")

st.sidebar.subheader("Task Vector Scaling (λ / Alpha)")
lambda_preset = st.sidebar.selectbox(
    "Quick Presets",
    ["Custom", "Conservative (0.3)", "Balanced (0.5)", "Identity (1.0)", "Aggressive (1.8)", "Stress Test (2.5)"],
    index=2,
)

preset_map = {
    "Conservative (0.3)": 0.3,
    "Balanced (0.5)": 0.5,
    "Identity (1.0)": 1.0,
    "Aggressive (1.8)": 1.8,
    "Stress Test (2.5)": 2.5,
}

initial_lambda = preset_map.get(lambda_preset, 0.5)
lambda_val = st.sidebar.slider(
    "λ (Scaling Factor)",
    min_value=-1.0,
    max_value=3.0,
    value=float(initial_lambda),
    step=0.05,
    help="Coefficient multiplying the task vector τ = W_tuned - W_base. λ=1.0 is full fine-tuned; λ=0.5 is halfway merge.",
)

st.sidebar.subheader("Weights Checkpoints")
base_path_input = st.sidebar.text_input("Base Reference Weights", "./weights/gemma4_base.safetensors")
tuned_path_input = st.sidebar.text_input("Tuned Weights", "./weights/model.safetensors")

base_file = Path(base_path_input)
tuned_file = Path(tuned_path_input)

if not base_file.exists():
    st.sidebar.error(f"❌ Missing base file: {base_file.name}")
else:
    st.sidebar.success(f"✓ Base ready ({base_file.stat().st_size / (1024**3):.2f} GB)")

if not tuned_file.exists():
    st.sidebar.error(f"❌ Missing tuned file: {tuned_file.name}")
else:
    st.sidebar.success(f"✓ Tuned ready ({tuned_file.stat().st_size / (1024**3):.2f} GB)")

st.sidebar.subheader("Target Filters")
target_filter = st.sidebar.multiselect(
    "Filter Projections",
    ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.down_proj"],
    default=["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.down_proj"],
)

limit_layers = st.sidebar.number_input(
    "Limit Layers (0 for all)",
    min_value=0,
    max_value=140,
    value=0,
    help="Set to small number (e.g. 20) for lightning-fast preview.",
)

st.sidebar.subheader("Telemetry Routing")
has_live_creds = bool(os.getenv("MONGODB_URI") or os.getenv("SNOWFLAKE_ACCOUNT"))
mock_telemetry_toggle = st.sidebar.checkbox("Force Mock Telemetry", value=not has_live_creds)
if not mock_telemetry_toggle:
    if os.getenv("MONGODB_URI"):
        st.sidebar.success("✓ MongoDB Atlas Ready")
    if os.getenv("SNOWFLAKE_ACCOUNT"):
        st.sidebar.success("✓ Snowflake Warehouse Ready")

execute_btn = st.sidebar.button("🚀 Execute Weight Surgery", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.title("⚡ M3O: Mechanistic Model Merger & Optimizer")
st.markdown(
    "Streamingly compute task vectors $\\tau = W_{\\text{tuned}} - W_{\\text{base}}$, "
    "evaluate merged weights $W_{\\text{merged}} = W_{\\text{base}} + (\\lambda \\cdot \\tau)$, "
    "and monitor model collapse heuristics in real time."
)

# Persistent Session State
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "telemetry_records" not in st.session_state:
    st.session_state.telemetry_records = []

# Execution Trigger
if execute_btn:
    if not base_file.exists() or not tuned_file.exists():
        st.error("Weights safetensors files could not be found. Please verify the paths in the sidebar.")
    else:
        with st.spinner("Initializing streaming surgery engine and reading checkpoints..."):
            engine = WeightSurgeryEngine(base_path=str(base_file), tuned_path=str(tuned_file))
            telemetry = TelemetryLogger(mock_mode=mock_telemetry_toggle)

            import safetensors
            with safetensors.safe_open(str(base_file), framework="numpy") as f_base:
                all_keys = list(f_base.keys())

            # Apply target filters
            filtered_keys = [k for k in all_keys if any(tf in k for tf in target_filter)]
            if limit_layers > 0:
                filtered_keys = filtered_keys[:limit_layers]

            progress_bar = st.progress(0, text="Streaming layer-wise surgery...")
            results_list = []
            t_start = time.perf_counter()

            for idx, res in enumerate(engine.stream_merge(lambda_val=lambda_val, target_keys=filtered_keys)):
                results_list.append(res)
                telemetry.log_layer_metric(
                    layer_name=res.tensor_name,
                    alpha=res.lambda_val,
                    frobenius_drift=res.frob_drift,
                    cos_sim=res.cos_sim,
                    status=res.status,
                )
                progress_bar.progress((idx + 1) / len(filtered_keys), text=f"Processing [{idx + 1}/{len(filtered_keys)}]: {res.tensor_name}")

            t_elapsed = time.perf_counter() - t_start
            flushed_sf = telemetry.flush_to_snowflake()

            # Store in session state
            st.session_state.last_results = results_list
            st.session_state.last_summary = {
                "lambda_val": lambda_val,
                "elapsed": t_elapsed,
                "throughput": len(results_list) / t_elapsed if t_elapsed > 0 else 0,
                "total_layers": len(results_list),
                "flushed_sf": flushed_sf,
            }
            st.session_state.telemetry_records = list(telemetry.mongo_mock_records)
            telemetry.close()

            st.success(f"Successfully processed {len(results_list)} layers in {t_elapsed:.2f}s ({len(results_list)/t_elapsed:.2f} layers/sec)!")


# -----------------------------------------------------------------------------
# Results Presentation
# -----------------------------------------------------------------------------
if st.session_state.last_results is not None:
    results = st.session_state.last_results
    summary = st.session_state.last_summary

    # Prepare DataFrame
    data = []
    for r in results:
        l_idx, p_type = extract_layer_metadata(r.tensor_name)
        data.append({
            "Layer Name": r.tensor_name,
            "Layer Index": l_idx,
            "Projection": p_type,
            "Lambda": r.lambda_val,
            "Cosine Sim": r.cos_sim,
            "Frobenius Drift": r.frob_drift,
            "Status": r.status,
            "Shape": str(r.shape),
            "Dtype": r.dtype,
        })
    df = pd.DataFrame(data)

    # 1. Executive KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    mean_cos = df["Cosine Sim"].mean()
    min_cos = df["Cosine Sim"].min()
    mean_frob = df["Frobenius Drift"].mean()
    overall_status = classify_status(min_cos)

    status_color = "status-badge-stable" if overall_status == STATUS_STABLE else ("status-badge-degraded" if overall_status == STATUS_DEGRADED else "status-badge-critical")

    with col1:
        st.markdown(f"**Model Health**<br><span class='{status_color}'>{overall_status}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Layers Processed", f"{summary['total_layers']}", f"λ = {summary['lambda_val']}")
    with col3:
        st.metric("Mean Cosine Sim", f"{mean_cos:.6f}", f"Min: {min_cos:.6f}")
    with col4:
        st.metric("Mean Frob Drift", f"{mean_frob:.6f}")
    with col5:
        st.metric("Throughput", f"{summary['throughput']:.1f} lyr/s", f"{summary['elapsed']:.2f}s total")

    st.markdown("---")

    # 2. Interactive Analysis Tabs
    tab_analytics, tab_data, tab_telemetry, tab_inspector, tab_gemma = st.tabs([
        "📈 Drift & Collapse Analytics",
        "📋 Layer Telemetry Explorer",
        "🛰️ Dual Telemetry Stream",
        "🔬 Mechanistic Inspector",
        "🧠 Gemma 4 Diagnostician (gemma-4-31b-it)",
    ])

    with tab_analytics:
        st.subheader("Integrity Heuristics Across Network Depth")

        # Cosine Similarity Chart with Collapse Thresholds
        chart_df = df[df["Layer Index"] >= 0].sort_values(by=["Layer Index", "Projection"])

        base_chart = alt.Chart(chart_df).encode(
            x=alt.X("Layer Index:O", title="Layer Index (Text-Decoder Depth 0 → 34)"),
            y=alt.Y("Cosine Sim:Q", scale=alt.Scale(domain=[min(0.6, min_cos - 0.05), 1.01]), title="Cosine Similarity"),
            color=alt.Color("Projection:N", title="Projection Target"),
            tooltip=["Layer Name", "Projection", "Layer Index", "Cosine Sim", "Frobenius Drift", "Status"],
        )
        line_cos = base_chart.mark_line(point=True)

        # Collapse Threshold rule lines
        rule_stable = alt.Chart(pd.DataFrame({'y': [0.85]})).mark_rule(color='#FFD600', strokeDash=[4, 4]).encode(y='y:Q')
        rule_crit = alt.Chart(pd.DataFrame({'y': [0.65]})).mark_rule(color='#FF5252', strokeDash=[4, 4]).encode(y='y:Q')

        st.altair_chart((line_cos + rule_stable + rule_crit).properties(height=380), use_container_width=True)
        st.caption("Thresholds: Yellow dashed line = Degraded threshold (0.85). Red dashed line = Critical Collapse threshold (0.65).")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Frobenius Norm Drift by Depth")
            drift_chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Layer Index:O", title="Layer Depth"),
                y=alt.Y("Frobenius Drift:Q", title="Frobenius Drift ||W_merged - W_base||"),
                color=alt.Color("Projection:N"),
                tooltip=["Layer Index", "Projection", "Frobenius Drift"],
            ).properties(height=300)
            st.altair_chart(drift_chart, use_container_width=True)

        with col_b:
            st.subheader("Health Status Breakdown")
            status_counts = df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            status_pie = alt.Chart(status_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Status",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["STABLE", "DEGRADED", "CRITICAL"],
                        range=["#00E676", "#FFD600", "#FF5252"]
                    ),
                ),
                tooltip=["Status", "Count"],
            ).properties(height=300)
            st.altair_chart(status_pie, use_container_width=True)

    with tab_data:
        st.subheader("Layer-by-Layer Telemetry Table")
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            search_query = st.text_input("🔍 Search layer name...", "")
        with filter_col2:
            status_filter = st.multiselect("Filter Status", ["STABLE", "DEGRADED", "CRITICAL"], default=["STABLE", "DEGRADED", "CRITICAL"])

        filtered_df = df[df["Status"].isin(status_filter)]
        if search_query:
            filtered_df = filtered_df[filtered_df["Layer Name"].str.contains(search_query, case=False)]

        st.dataframe(
            filtered_df[["Layer Name", "Projection", "Shape", "Lambda", "Frobenius Drift", "Cosine Sim", "Status"]],
            use_container_width=True,
            height=400,
        )

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Telemetry as CSV",
            data=csv,
            file_name=f"m3o_telemetry_lambda_{summary['lambda_val']}.csv",
            mime="text/csv",
        )

    with tab_telemetry:
        st.subheader("Dual Telemetry Activity Stream")
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown("#### Hot Path: MongoDB Atlas Ingestion")
            st.markdown(f"**Logged Documents:** {len(st.session_state.telemetry_records)} records")
            if st.session_state.telemetry_records:
                st.json(st.session_state.telemetry_records[:3])
                st.caption("Displaying sample JSON payloads inserted per layer into MongoDB Atlas collection `layer_metrics`.")

        with t_col2:
            st.markdown("#### Cold Path: Snowflake Micro-Batches")
            st.markdown(f"**Target Table:** `M3O_ENGINE.TELEMETRY.LAYER_METRICS`")
            st.markdown(f"**Total Records Flushed:** {summary['flushed_sf']} rows")
            st.code(
                f"""INSERT INTO M3O_ENGINE.TELEMETRY.LAYER_METRICS (
    LAYER_NAME, ALPHA, FROBENIUS_DRIFT, COS_SIM, STATUS, TIMESTAMP
) VALUES (%s, %s, %s, %s, %s, %s); /* {summary['flushed_sf']} records */""",
                language="sql",
            )

    with tab_inspector:
        st.subheader("Single Layer Mechanistic Inspector")
        chosen_layer = st.selectbox("Select Layer to Inspect", df["Layer Name"].tolist())
        layer_row = df[df["Layer Name"] == chosen_layer].iloc[0]

        i_col1, i_col2, i_col3 = st.columns(3)
        with i_col1:
            st.metric("Matrix Shape", layer_row["Shape"])
            st.metric("Precision Dtype", layer_row["Dtype"])
        with i_col2:
            st.metric("Cosine Similarity", f"{layer_row['Cosine Sim']:.8f}")
            st.metric("Frobenius Drift", f"{layer_row['Frobenius Drift']:.8f}")
        with i_col3:
            st.metric("Layer Health", layer_row["Status"])
            st.metric("Task Vector Scaling λ", f"{layer_row['Lambda']}")

    with tab_gemma:
        st.subheader("Gemma 4 (`gemma-4-26b-a4b-it`) Mechanistic Diagnostician")
        st.markdown(
            "Calls Google's **`gemma-4-26b-a4b-it`** model directly via the Gemini API "
            "to perform an automated mechanistic interpretability assessment of the surgery pass."
        )

        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            placeholder="AQ... or AIzaSy... (pre-filled from .env)",
        )

        metrics_payload = {
            "alpha": summary["lambda_val"],
            "total_layers": summary["total_layers"],
            "mean_cos_sim": mean_cos,
            "min_cos_sim": min_cos,
            "mean_frob_drift": mean_frob,
            "stable_count": (df["Status"] == "STABLE").sum(),
            "degraded_count": (df["Status"] == "DEGRADED").sum(),
            "critical_count": (df["Status"] == "CRITICAL").sum(),
        }

        if st.button("🤖 Run gemma-4-26b-a4b-it Diagnosis", type="primary"):
            from telemetry.gemini_diagnostics import diagnose_model_surgery_with_gemini
            with st.spinner("Invoking gemma-4-26b-a4b-it via Gemini API..."):
                report = diagnose_model_surgery_with_gemini(metrics_payload, api_key=api_key_input)
                st.markdown(report)

else:
    st.info("👈 Click **Execute Weight Surgery** in the sidebar to stream layers, compute task vectors, and visualize model collapse heuristics.")
