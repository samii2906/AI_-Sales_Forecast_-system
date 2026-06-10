import streamlit as st
from utils.theme import inject_css

st.set_page_config(
    page_title="Intelligent Sales AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <h2>🧠 Intelligent Sales AI</h2>
        <p>Analytics & Forecasting Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📂 Data Upload",
            "🔬 EDA Analysis",
            "📈 Sales Forecasting",
            "🔄 Churn Prediction",
            "👥 Customer Segmentation",
            "🚨 Anomaly Detection",
            "💡 Insights",
            "📋 Reports"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Built with Streamlit + Machine Learning")

# Routing
if page == "📂 Data Upload":
    from modules import data_upload
    data_upload.show()

elif page == "🔬 EDA Analysis":
    from modules import eda
    eda.show()

elif page == "📈 Sales Forecasting":
    from modules import sales_forecasting
    sales_forecasting.show()

elif page == "🔄 Churn Prediction":
    from modules import churn_prediction
    churn_prediction.show()

elif page == "👥 Customer Segmentation":
    from modules import customer_segmentation
    customer_segmentation.show()

elif page == "🚨 Anomaly Detection":
    from modules import anomaly_detection
    anomaly_detection.show()

elif page == "💡 Insights":
    from modules import insights
    insights.show()

elif page == "📋 Reports":
    from modules import reports
    reports.show()