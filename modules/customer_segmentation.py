import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.theme import inject_css, PASTEL_COLORS, metric_card

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>👥 Customer Segmentation (RFM + KMeans)</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df'].copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    sel_cols = st.multiselect("Select features for clustering", num_cols, default=num_cols[:4])
    n_clusters = st.slider("Number of customer segments", 2, 8, 4)

    if st.button("🔬 Run Segmentation", use_container_width=True):
        X = df[sel_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Segment'] = model.fit_predict(X_scaled)
        df['Segment'] = 'Segment ' + (df['Segment'] + 1).astype(str)

        st.success(f"✅ {n_clusters} customer segments identified!")

        seg_summary = df.groupby('Segment')[sel_cols].mean().round(2)
        st.dataframe(seg_summary.style.background_gradient(cmap='PuBu'), use_container_width=True)

        fig_pie = px.pie(df, names='Segment', color_discrete_sequence=PASTEL_COLORS,
                         hole=0.4, title="Customer Segment Distribution")
        fig_pie.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0")
        st.plotly_chart(fig_pie, use_container_width=True)

        if len(sel_cols) >= 2:
            fig = px.scatter(df, x=sel_cols[0], y=sel_cols[1], color='Segment',
                             color_discrete_sequence=PASTEL_COLORS,
                             title="Segment Scatter", opacity=0.7)
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart per segment
        from plotly.graph_objects import Figure, Scatterpolar
        fig_radar = Figure()
        for seg in df['Segment'].unique():
            vals = df[df['Segment'] == seg][sel_cols].mean().values.tolist()
            vals += [vals[0]]
            fig_radar.add_trace(Scatterpolar(r=vals, theta=sel_cols + [sel_cols[0]],
                                             fill='toself', name=seg))
        fig_radar.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0",
                                 polar=dict(bgcolor="#1a1d2e"),
                                 title="Segment Radar Chart")
        st.plotly_chart(fig_radar, use_container_width=True)