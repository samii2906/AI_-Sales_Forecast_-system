import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.theme import inject_css, PASTEL_COLORS

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>🔬 Exploratory Data Analysis</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df']
    col_types = st.session_state.get('col_types', {})
    numeric_cols = col_types.get('numeric', df.select_dtypes(include=np.number).columns.tolist())
    cat_cols = col_types.get('categorical', df.select_dtypes(include='object').columns.tolist())

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📊 Categorical", "📉 Time Trends"])

    # ── TAB 1: Distributions
    with tab1:
        st.markdown("### Numerical Distributions")
        if numeric_cols:
            sel = st.selectbox("Select column", numeric_cols, key="dist_col")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=sel, nbins=40, title=f"Distribution of {sel}",
                                   color_discrete_sequence=[PASTEL_COLORS[0]])
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df, y=sel, title=f"Box Plot: {sel}",
                              color_discrete_sequence=[PASTEL_COLORS[1]])
                fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig2, use_container_width=True)

            # Stats summary
            st.markdown("#### 📋 Descriptive Statistics")
            stats = df[numeric_cols].describe().T.round(3)
            stats['skewness'] = df[numeric_cols].skew().round(3)
            stats['kurtosis'] = df[numeric_cols].kurtosis().round(3)
            st.dataframe(stats.style.background_gradient(cmap='PuBu'), use_container_width=True)

    # ── TAB 2: Correlations
    with tab2:
        st.markdown("### Correlation Heatmap")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                            title="Pearson Correlation Matrix")
            fig.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Scatter
            st.markdown("#### Scatter Explorer")
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X Axis", numeric_cols, key="sc_x")
            y_col = c2.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="sc_y")
            color_col = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")
            fig3 = px.scatter(df, x=x_col, y=y_col,
                              color=color_col if color_col != "None" else None,
                              color_discrete_sequence=PASTEL_COLORS,
                              trendline="ols", opacity=0.7,
                              title=f"{x_col} vs {y_col}")
            fig3.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3: Categorical
    with tab3:
        if cat_cols:
            sel_cat = st.selectbox("Select Categorical Column", cat_cols, key="cat_col")
            vc = df[sel_cat].value_counts().reset_index()
            vc.columns = [sel_cat, 'Count']
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(vc, x=sel_cat, y='Count', title=f"Frequency: {sel_cat}",
                             color='Count', color_continuous_scale='Purples')
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.pie(vc, names=sel_cat, values='Count',
                              color_discrete_sequence=PASTEL_COLORS,
                              title=f"Share: {sel_cat}", hole=0.4)
                fig2.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0")
                st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 4: Time Trends
    with tab4:
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if date_cols and numeric_cols:
            dc = st.selectbox("Date Column", date_cols, key="td_date")
            vc2 = st.selectbox("Value Column", numeric_cols, key="td_val")
            freq = st.radio("Frequency", ["Daily", "Weekly", "Monthly"], horizontal=True)
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
            ts = df.set_index(dc)[vc2].resample(freq_map[freq]).sum().reset_index()
            fig = px.line(ts, x=dc, y=vc2, title=f"{vc2} Over Time ({freq})",
                          color_discrete_sequence=[PASTEL_COLORS[0]])
            fig.update_traces(fill='tozeroy', fillcolor='rgba(179,157,219,0.1)')
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datetime columns detected for time trend analysis.")