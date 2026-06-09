import streamlit as st
from utils.theme import inject_css

st.set_page_config(
    page_title="Intelligent Sales AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <span style='font-size: 2.5rem;'>🧠</span><br>
        <span style='font-size: 1.2rem; font-weight: 700; color: #b39ddb;'>Intelligent Sales AI</span><br>
        <span style='font-size: 0.75rem; color: #9e9e9e;'>Analytics & Forecasting Platform</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigate", [
        "📂 Data Upload",
        "🔬 EDA Analysis",
        "📈 Sales Forecasting",
        "🔄 Churn Prediction",
        "👥 Customer Segmentation",
        "🚨 Anomaly Detection",
        "💡 Insights",
        "📋 Reports"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("<div style='color:#9e9e9e; font-size:0.75rem; text-align:center'>Built with Streamlit + ML</div>",
                unsafe_allow_html=True)

# Route pages
if "📂 Data Upload" in page:
    from modules import data_upload
    data_upload.show()

elif "🔬 EDA" in page:
    from modules import eda
    eda.show()

elif "📈 Sales Forecasting" in page:
    from modules import sales_forecasting
    sales_forecasting.show()

elif "🔄 Churn" in page:
    from modules import churn_prediction
    churn_prediction.show()

elif "👥 Customer Segmentation" in page:
    from modules import customer_segmentation
    customer_segmentation.show()

elif "🚨 Anomaly" in page:
    from modules import anomaly_detection
    anomaly_detection.show()

elif "💡 Insights" in page:
    from modules import insights
    insights.show()

elif "📋 Reports" in page:
    from modules import reports
    reports.show()