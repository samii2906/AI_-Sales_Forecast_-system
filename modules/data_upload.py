import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import preprocess_dataframe, detect_column_types
from utils.theme import inject_css, metric_card, PASTEL_COLORS
from utils.sample_data_generator import generate_sales_data, generate_churn_data
import io

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>📂 Data Upload & Preprocessing</h2></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload Data", "🧪 Use Sample Data"])

    with tab1:
        uploaded = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state['raw_df'] = df
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🛒 Load Sales Sample Data", use_container_width=True):
                st.session_state['raw_df'] = generate_sales_data()
                st.success("Sales data loaded!")
        with col2:
            if st.button("👥 Load Churn Sample Data", use_container_width=True):
                st.session_state['raw_df'] = generate_churn_data()
                st.success("Churn data loaded!")

    if 'raw_df' not in st.session_state:
        st.info("👆 Upload a file or load sample data to begin.")
        return

    df = st.session_state['raw_df'].copy()

    # ── Overview Cards
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("Total Rows", f"{len(df):,}"), unsafe_allow_html=True)
    col2.markdown(metric_card("Total Columns", len(df.columns), color="#80cbc4"), unsafe_allow_html=True)
    col3.markdown(metric_card("Missing Values", df.isnull().sum().sum(), color="#ffe082"), unsafe_allow_html=True)
    col4.markdown(metric_card("Duplicates", df.duplicated().sum(), color="#f48fb1"), unsafe_allow_html=True)

    # ── Raw Preview
    with st.expander("🔍 Raw Data Preview", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    # ── Data Types
    with st.expander("🏷️ Column Info & Data Types"):
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Dtype': df.dtypes.values,
            'Non-Null': df.notnull().sum().values,
            'Null %': (df.isnull().mean() * 100).round(2).values,
            'Unique': df.nunique().values,
            'Sample': [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "N/A" for c in df.columns]
        })
        st.dataframe(info_df, use_container_width=True)

    # ── Missing Value Heatmap
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        st.markdown("### 🟥 Missing Value Heatmap")
        miss_df = df[missing_cols].isnull().astype(int).head(50)
        fig = px.imshow(miss_df.T, color_continuous_scale=["#1a1d2e", "#f48fb1"],
                        labels=dict(color="Missing"),
                        title="Missing Values (Red = Missing)")
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0")
        st.plotly_chart(fig, use_container_width=True)

    # ── Preprocessing
    st.markdown("### ⚙️ Auto Preprocessing")
    if st.button("🚀 Run Auto Preprocessing", use_container_width=True):
        with st.spinner("Cleaning data..."):
            clean_df, report = preprocess_dataframe(df)
            st.session_state['df'] = clean_df
            st.session_state['col_types'] = detect_column_types(clean_df)

        st.success("✅ Preprocessing complete!")

        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Missing Fixed", len(report.get('missing_before', {})), color="#a5d6a7"), unsafe_allow_html=True)
        c2.markdown(metric_card("Duplicates Removed", report.get('duplicates_removed', 0), color="#ffe082"), unsafe_allow_html=True)
        c3.markdown(metric_card("Clean Rows", f"{len(clean_df):,}", color="#80cbc4"), unsafe_allow_html=True)

        st.dataframe(clean_df.head(50), use_container_width=True)

        # Download cleaned data
        csv = clean_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
    
    elif 'df' not in st.session_state:
        st.warning("Click 'Run Auto Preprocessing' to clean and proceed.")