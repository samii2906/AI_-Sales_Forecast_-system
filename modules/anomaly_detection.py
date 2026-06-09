import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from utils.theme import inject_css, PASTEL_COLORS, metric_card

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>🚨 Anomaly Detection</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df'].copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.error("No numeric columns found.")
        return

    sel_cols = st.multiselect("Select columns for anomaly detection", num_cols, default=num_cols[:3])
    contamination = st.slider("Expected anomaly rate (%)", 1, 20, 5) / 100

    if st.button("🔍 Detect Anomalies", use_container_width=True):
        X = df[sel_cols].fillna(0)
        model = IsolationForest(contamination=contamination, random_state=42)
        df['Anomaly'] = model.fit_predict(X)
        df['Anomaly_Score'] = model.score_samples(X)
        df['Is_Anomaly'] = df['Anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

        n_anomalies = (df['Anomaly'] == -1).sum()
        col1, col2 = st.columns(2)
        col1.markdown(metric_card("Anomalies Found", n_anomalies, color="#f48fb1"), unsafe_allow_html=True)
        col2.markdown(metric_card("Normal Points", len(df) - n_anomalies, color="#a5d6a7"), unsafe_allow_html=True)

        if len(sel_cols) >= 2:
            fig = go.Figure()
            normal = df[df['Anomaly'] == 1]
            anomalous = df[df['Anomaly'] == -1]
            fig.add_trace(go.Scatter(x=normal[sel_cols[0]], y=normal[sel_cols[1]],
                                     mode='markers', name='Normal',
                                     marker=dict(color=PASTEL_COLORS[1], size=5, opacity=0.6)))
            fig.add_trace(go.Scatter(x=anomalous[sel_cols[0]], y=anomalous[sel_cols[1]],
                                     mode='markers', name='Anomaly',
                                     marker=dict(color=PASTEL_COLORS[2], size=10,
                                                 symbol='x', line=dict(width=2))))
            fig.update_layout(title="Anomaly Scatter Plot", paper_bgcolor="#0e1117",
                              plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df[df['Anomaly'] == -1][sel_cols + ['Anomaly_Score']].head(50),
                     use_container_width=True)

        csv = df[df['Anomaly'] == -1].to_csv(index=False).encode()
        st.download_button("⬇️ Download Anomalies", csv, "anomalies.csv")