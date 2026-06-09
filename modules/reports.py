import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from utils.theme import inject_css
import datetime

def generate_excel(df, metrics_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        if metrics_dict:
            pd.DataFrame([metrics_dict]).to_excel(writer, sheet_name='Model Metrics', index=False)
        if 'future_forecast' in st.session_state:
            st.session_state['future_forecast'].to_excel(writer, sheet_name='Forecast', index=False)
        if 'churn_preds' in st.session_state:
            churn_df = pd.DataFrame({
                'Predicted': st.session_state['churn_preds'],
                'Actual': st.session_state['churn_actual'],
            })
            churn_df.to_excel(writer, sheet_name='Churn Predictions', index=False)
    return output.getvalue()

def generate_pdf(df, metrics_dict):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(80, 80, 180)
    pdf.cell(0, 12, "Intelligent Sales AI - Analytics Report", ln=True, align='C')
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(5)

    # Dataset Summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Dataset Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Rows: {len(df):,}  |  Columns: {len(df.columns)}", ln=True)
    pdf.cell(0, 7, f"Numeric Columns: {len(df.select_dtypes(include=np.number).columns)}", ln=True)
    pdf.ln(5)

    # Model Metrics
    if metrics_dict:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Model Performance Metrics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in metrics_dict.items():
            pdf.cell(0, 7, f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}", ln=True)
        pdf.ln(5)

    # Descriptive stats
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Descriptive Statistics", ln=True)
    pdf.set_font("Helvetica", "", 8)
    stats = df.describe().round(2)
    col_widths = max(25, 190 // (len(stats.columns) + 1))

    # Header row
    pdf.set_fill_color(230, 225, 255)
    pdf.cell(30, 7, "Stat", border=1, fill=True)
    for col in stats.columns[:5]:
        pdf.cell(col_widths, 7, str(col)[:10], border=1, fill=True)
    pdf.ln()

    for idx, row in stats.iterrows():
        pdf.cell(30, 6, str(idx), border=1)
        for val in row.values[:5]:
            pdf.cell(col_widths, 6, str(round(val, 2)), border=1)
        pdf.ln()

    return pdf.output()

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>📋 Reports & Export</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df']
    metrics = st.session_state.get('forecast_metrics', {})

    st.markdown("### 📊 Report Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if metrics:
        st.markdown("### 📈 Model Metrics")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📗 Excel Report")
        if st.button("Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel..."):
                excel_data = generate_excel(df, metrics)
            st.download_button("⬇️ Download Excel Report", excel_data,
                               "analytics_report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        st.markdown("#### 📕 PDF Report")
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_data = generate_pdf(df, metrics)
            st.download_button("⬇️ Download PDF Report", bytes(pdf_data),
                               "analytics_report.pdf", "application/pdf")