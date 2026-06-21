import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. PAGE SETUP & THEME ---
st.set_page_config(
    page_title="VibroStats Enterprise", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load High-Tech Stylesheet
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# --- 2. SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("<h2 style='color:#00ffcc; text-align:center;'>🎛️ CONTROL PANEL</h2>", unsafe_allow_html=True)
st.sidebar.write("---")

st.sidebar.subheader("🎯 Algorithm Tuning")
sensitivity = st.sidebar.slider("Anomaly Threshold (Z-Score)", 1.0, 5.0, 3.0, 0.1)
window_size = st.sidebar.slider("Rolling Window Size", 5, 100, 20, 5)

st.sidebar.write("---")
st.sidebar.subheader("🏭 Asset Information")
asset_name = st.sidebar.text_input("Machine ID / Tag", "MOTOR_COMP_042")
location_tag = st.sidebar.text_input("Plant Location", "Zone A - Main Line")

# --- 3. MAIN DASHBOARD HEADER ---
st.markdown("<h1>⚡ VIBROSTATS ENTERPRISE <span style='font-size:16px; color:#888;'>v3.2</span></h1>", unsafe_allow_html=True)
st.markdown(f"**Asset:** `{asset_name}` | **Location:** `{location_tag}`")
st.markdown("---")

# --- 4. DATA INGESTION ENGINE ---
uploaded_file = st.file_uploader("📂 Drag & Drop Industrial Telemetry Log (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    vib_cols = [c for c in ['accX', 'accY', 'accZ'] if c in cols]
    
    if not vib_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            vib_cols = [numeric_cols[0]]
            
    if not vib_cols:
        st.error("❌ Critical Error: No numeric data streams detected.")
        st.stop()

    primary_axis = vib_cols[0]
    rolling_mean = df[primary_axis].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df[primary_axis].rolling(window=window_size, min_periods=1).std().fillna(1e-4)
    
    df['Z_Score'] = (df[primary_axis] - rolling_mean) / rolling_std
    df['Anomaly'] = df['Z_Score'].abs() > sensitivity
    
    max_z = float(df['Z_Score'].abs().max())
    total_points = len(df)
    anomaly_points = int(df['Anomaly'].sum())
    anomaly_percentage = (anomaly_points / total_points) * 100
    
    avg_temp = float(df['temperature'].mean()) if 'temperature' in df.columns else 24.5
    
    # --- 6. ENTERPRISE KPI CARDS ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        delta_val = "CRITICAL" if max_z > sensitivity else "NOMINAL"
        st.metric(label="Peak Anomaly Intensity", value=f"{max_z:.2f} σ", delta=delta_val, delta_color="inverse")
    with kpi2:
        delta_pct = "Action Required" if anomaly_percentage > 5 else "Acceptable"
        st.metric(label="Signal Deviation Rate", value=f"{anomaly_percentage:.1f}%", delta=delta_pct)
    with kpi3:
        if 'temperature' in df.columns:
            max_temp = float(df['temperature'].max())
            st.metric(label="Mean Thermal State", value=f"{avg_temp:.1f} °C", delta=f"Max: {max_temp:.1f}°C")
        else:
            st.metric(label="Mean Thermal State", value="N/A", delta="No Sensor Found")
    with kpi4:
        if anomaly_percentage > 5 or max_z > sensitivity:
            st.markdown("<div class='status-box status-alert'>🚨 SYSTEM FAULT DETECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-box status-ok'>✅ ASSET HEALTH NOMINAL</div>", unsafe_allow_html=True)

    st.write("##")

    # --- 7. INTERACTIVE INTERFACE TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Interactive Waveform Analysis", "🔍 Deep Analytics Engine", "📋 Automated Export Report"])
    
    with tab1:
        st.subheader("High-Frequency Waveform Visualization")
        fig = go.Figure()
        
        colors = {'accX': '#00ffcc', 'accY': '#ff007f', 'accZ': '#ffcc00'}
        for col in vib_cols:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode='lines',
                name=f"Vibration {col}",
                line=dict(color=colors.get(col, '#ffffff'), width=1.5)
            ))
            
        anomalies = df[df['Anomaly']]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies[primary_axis],
                mode='markers',
                name='Flagged Outlier',
                marker=dict(color='#ff4b4b', size=8, symbol='x')
            ))
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='#223'),
            yaxis=dict(showgrid=True, gridcolor='#223')
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Statistical Variance Distribution")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df[primary_axis],
                name='Density Distribution',
                marker_color='#00ffcc',
                opacity=0.75
            ))
            fig_hist.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_right:
            st.subheader("Thermal Baseline vs. Mechanical Stress")
            if 'temperature' in df.columns:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=df['temperature'], y=df[primary_axis],
                    mode='markers',
                    marker=dict(color=df['Z_Score'].abs(), colorscale='Viridis', showscale=True, size=6)
                ))
                fig_scatter.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Temperature (°C)", yaxis_title="Vibration Amplitude"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Additional diagnostic charts will unlock when temperature data is included.")

    with tab3:
        st.subheader("Automated Industrial Inspection Summary")
        anomaly_log = df[df['Anomaly']][vib_cols + ['Z_Score']].head(50)
        
        if not anomaly_log.empty:
            st.dataframe(anomaly_log.style.format(precision=3), use_container_width=True)
            st.download_button(
                label="📥 Download Certified CSV Diagnostic Log",
                data=df.to_csv(index=False),
                file_name=f"Diagnostic_Report_{asset_name}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 No anomalies detected within specified parameters.")

else:
    st.markdown(
        "<div style='text-align: center; padding: 50px; border: 2px dashed #334; border-radius: 10px; background: #111625;'><h3 style='color: #00ffcc;'>Awaiting Sensor Telemetry Input Pipeline</h3></div>", 
        unsafe_allow_html=True
    )
