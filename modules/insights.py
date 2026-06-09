import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import inject_css, PASTEL_COLORS, metric_card

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>💡 Business Insights with Filters</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df'].copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── Dynamic Filters Sidebar
    st.sidebar.markdown("### 🎛️ Insight Filters")
    filters = {}
    for col in cat_cols[:4]:
        opts = df[col].dropna().unique().tolist()
        sel = st.sidebar.multiselect(f"{col}", opts, default=opts)
        filters[col] = sel

    for col, sel in filters.items():
        if sel:
            df = df[df[col].isin(sel)]

    for col in num_cols[:2]:
        mn, mx = float(df[col].min()), float(df[col].max())
        if mn < mx:
            rng = st.sidebar.slider(f"{col} range", mn, mx, (mn, mx))
            df = df[(df[col] >= rng[0]) & (df[col] <= rng[1])]

    st.markdown(f"**Filtered rows: {len(df):,}**")

    if num_cols:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        kpis = num_cols[:4]
        colors = PASTEL_COLORS[:4]
        for i, (c, k) in enumerate(zip([col1, col2, col3, col4], kpis)):
            val = df[k].sum()
            c.markdown(metric_card(f"Total {k}", f"{val:,.0f}", color=colors[i]), unsafe_allow_html=True)

        # Group-by analysis
        st.markdown("### 📊 Group-by Analysis")
        if cat_cols and num_cols:
            c1, c2, c3 = st.columns(3)
            grp = c1.selectbox("Group by", cat_cols, key="ins_grp")
            agg_col = c2.selectbox("Aggregate", num_cols, key="ins_agg")
            agg_fn = c3.selectbox("Function", ["sum", "mean", "count", "max", "min"])

            grouped = df.groupby(grp)[agg_col].agg(agg_fn).reset_index().sort_values(agg_col, ascending=False)
            c1_, c2_ = st.columns(2)
            with c1_:
                fig = px.bar(grouped, x=grp, y=agg_col, title=f"{agg_fn.title()} of {agg_col} by {grp}",
                             color=agg_col, color_continuous_scale='Purples')
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
            with c2_:
                fig2 = px.pie(grouped, names=grp, values=agg_col,
                              color_discrete_sequence=PASTEL_COLORS, hole=0.45,
                              title=f"Share of {agg_col} by {grp}")
                fig2.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0")
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(grouped, use_container_width=True)

        # Top N
        st.markdown("### 🏆 Top / Bottom Performers")
        if cat_cols and num_cols:
            tc1, tc2, tc3 = st.columns(3)
            dim = tc1.selectbox("Dimension", cat_cols, key="top_dim")
            met = tc2.selectbox("Metric", num_cols, key="top_met")
            n = tc3.slider("Top N", 3, 20, 10)
            top_df = df.groupby(dim)[met].sum().nlargest(n).reset_index()
            bot_df = df.groupby(dim)[met].sum().nsmallest(n).reset_index()
            t1, t2 = st.tabs([f"Top {n}", f"Bottom {n}"])
            with t1:
                fig = px.bar(top_df, x=dim, y=met, color=met, color_continuous_scale='Greens',
                             title=f"Top {n} {dim} by {met}")
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                fig2 = px.bar(bot_df, x=dim, y=met, color=met, color_continuous_scale='Reds',
                              title=f"Bottom {n} {dim} by {met}")
                fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig2, use_container_width=True)