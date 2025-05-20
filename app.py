import streamlit as st
import polars as pl
import plotly.graph_objects as go
import math
import joblib
import numpy as np
from io import BytesIO
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Shaker Dashboard", page_icon="📈", layout="wide")
st.set_option("server.maxUploadSize", 500)  # Increase file upload limit to 500MB

# --- Header with Branding ---
st.image("Prodigy_IQ_logo.png", width=200)
st.markdown("""
    <style>
        .main { padding-top: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- Theme Switcher ---
mode = st.radio("🎨 Select Theme Mode", ["Light", "Dark"], horizontal=True)
template = "plotly_white" if mode == "Light" else "plotly_dark"

# --- Title ---
st.markdown("""
<h1 style='text-align: center; font-size: 2.5rem;'>📊 Shaker Performance Dashboard</h1>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data(path):
    return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)

uploaded_file = st.file_uploader("📁 Upload Drilling CSV (up to 500MB)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    df = df.select([
        "Rate Of Penetration (ft_per_hr)",
        "Hole Diameter (in)",
        "Total Pump Output (gal_per_min)",
        "SHAKER #1 (Units)", "SHAKER #2 (Units)", "SHAKER #3 (PERCENT)",
        "On Bottom Hours (hrs)", "Mechanical Specific Energy (ksi)",
        "Time Of Penetration (min_per_ft)", "Circulating Hours (hrs)",
    ])

    df = df.with_columns([
        (pl.col("Hole Diameter (in)") / 12).alias("Hole Diameter (ft)"),
        ((math.pi * (pl.col("Hole Diameter (in)") / 24) ** 2) * pl.col("Rate Of Penetration (ft_per_hr)"))
            .alias("Solids Volume Rate (ft3/hr)"),
        ((pl.col("SHAKER #1 (Units)") + pl.col("SHAKER #2 (Units)") + pl.col("SHAKER #3 (PERCENT)") / 100) * 450)
            .alias("Max Screen Throughput (gpm)"),
        ((pl.col("Rate Of Penetration (ft_per_hr)")) * 0.6).alias("Screen Load Index"),
    ])

    df = df.with_columns([
        ((pl.col("Solids Volume Rate (ft3/hr)") / (pl.col("Max Screen Throughput (gpm)") + 1e-6)) * 100)
            .alias("Screen Utilization (%)")
    ])

    tab1, tab2, tab3, tab4 = st.tabs(["🧮 Utilization Meter", "⏳ Life Estimator", "📉 G-Force Detector", "📤 Export Reports"])

    with tab1:
        st.subheader("🧮 Real-Time Screen Utilization")
        col1, col2 = st.columns([3, 1])
        with col1:
            utilization = df["Screen Utilization (%)"].to_numpy()
            trendline = np.poly1d(np.polyfit(np.arange(len(utilization)), utilization, 2))(np.arange(len(utilization)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=utilization, name="Utilization (%)", mode="lines+markers", line=dict(color="#007ACC")))
            fig.add_trace(go.Scatter(y=trendline, name="Trendline", mode="lines", line=dict(color="red", dash="dash")))
            fig.update_layout(template=template, yaxis_title="% Utilization", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            current = utilization[-1]
            st.metric("Current Utilization", f"{current:.2f}%")
            if current > 90:
                st.error("❌ Alert: Screen load critical")
            elif current > 75:
                st.warning("⚠️ Warning: High load approaching")
            else:
                st.success("✅ Normal screen load")

    with tab2:
        st.subheader("⏳ Shaker Screen Life Estimator")

        def dummy_predict_life(rop, mse, hours):
            return max(0, 100 - (0.2 * rop + 0.5 * mse + 0.3 * hours))

        rop_np = df["Rate Of Penetration (ft_per_hr)"].to_numpy()
        mse_np = df["Mechanical Specific Energy (ksi)"].to_numpy()
        hours_np = df["On Bottom Hours (hrs)"].to_numpy()
        pred_life = [dummy_predict_life(r, m, h) for r, m, h in zip(rop_np, mse_np, hours_np)]

        fig_life = go.Figure(data=[go.Scatter(y=pred_life, mode="lines+markers", line=dict(color="magenta"))])
        fig_life.update_layout(template=template, height=300, title="Estimated Remaining Screen Life")
        st.plotly_chart(fig_life, use_container_width=True)

        if pred_life[-1] < 20:
            st.error("❌ Screen life critically low! Schedule screen change-out.")
        elif pred_life[-1] < 40:
            st.warning("⚠️ Screen life below 40%. Monitor closely.")
        else:
            st.success("✅ Screen life healthy.")

    with tab3:
        st.subheader("📉 Shaker G-Force Drop Detector (Heuristic)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["SHAKER #1 (Units)"].to_list(), name="Shaker #1", line=dict(color="green")))
        fig.add_trace(go.Scatter(y=df["SHAKER #2 (Units)"].to_list(), name="Shaker #2", line=dict(color="orange")))
        fig.add_trace(go.Scatter(y=df["SHAKER #3 (PERCENT)"].to_list(), name="Shaker #3 %", line=dict(color="blue")))
        fig.update_layout(template=template, height=400, yaxis_title="Shaker Output")
        st.plotly_chart(fig, use_container_width=True)

        shaker_drop = df[-1, "SHAKER #1 (Units)"] < 10 and df[-1, "SHAKER #2 (Units)"] < 10
        if shaker_drop:
            st.error("🔧 Possible G-force drop! Inspect shaker motors or tension.")
        else:
            st.success("✅ Shaker outputs nominal.")

    with tab4:
        st.subheader("📤 Export Data Report")
        df_pandas = df.to_pandas()

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_pandas.to_excel(writer, sheet_name="Dashboard Data", index=False)
        st.download_button("⬇️ Download as Excel", data=buffer.getvalue(), file_name="shaker_dashboard_export.xlsx")

        st.caption("PDF export feature coming soon...")

    st.success("📁 Dashboard successfully rendered from uploaded CSV.")
