import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.theme import inject_css, PASTEL_COLORS, metric_card

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>📈 Intelligent Sales Forecasting</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found.")
        return

    st.markdown("### ⚙️ Configuration")
    c1, c2, c3 = st.columns(3)
    target_col = c1.selectbox("🎯 Target (Sales Column)", numeric_cols)
    feature_cols = c2.multiselect("📥 Feature Columns", [c for c in numeric_cols if c != target_col],
                                   default=[c for c in numeric_cols if c != target_col][:3])
    model_choice = c3.selectbox("🤖 Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])

    test_size = st.slider("Test Split %", 10, 40, 20)

    if not feature_cols:
        st.info("Select at least one feature column.")
        return

    if st.button("🚀 Train Forecasting Model", use_container_width=True):
        with st.spinner("Training model..."):
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression()
            }
            model = models[model_choice]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

            st.session_state['forecast_model'] = model
            st.session_state['forecast_preds'] = preds
            st.session_state['forecast_actual'] = y_test.values
            st.session_state['forecast_metrics'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

        st.success(f"✅ {model_choice} trained!")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(metric_card("MAE", f"{mae:.2f}"), unsafe_allow_html=True)
        col2.markdown(metric_card("RMSE", f"{rmse:.2f}", color="#80cbc4"), unsafe_allow_html=True)
        col3.markdown(metric_card("R² Score", f"{r2:.4f}", color="#a5d6a7"), unsafe_allow_html=True)
        col4.markdown(metric_card("MAPE %", f"{mape:.2f}%", color="#ffe082"), unsafe_allow_html=True)

        # Actual vs Predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test.values[:100], mode='lines', name='Actual',
                                 line=dict(color=PASTEL_COLORS[0], width=2)))
        fig.add_trace(go.Scatter(y=preds[:100], mode='lines', name='Predicted',
                                 line=dict(color=PASTEL_COLORS[1], width=2, dash='dash')))
        fig.update_layout(title="Actual vs Predicted Sales", paper_bgcolor="#0e1117",
                          plot_bgcolor="#1a1d2e", font_color="#e0e0e0",
                          legend=dict(bgcolor="#1a1d2e"))
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        residuals = y_test.values - preds
        fig2 = px.histogram(x=residuals, nbins=40, title="Residuals Distribution",
                            color_discrete_sequence=[PASTEL_COLORS[3]])
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
        st.plotly_chart(fig2, use_container_width=True)

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
            fi = fi.sort_values('Importance', ascending=True)
            fig3 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                          title="Feature Importance",
                          color='Importance', color_continuous_scale='Purples')
            fig3.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig3, use_container_width=True)

        # Future Forecast
        st.markdown("### 🔮 Future Forecast")
        future_days = st.slider("Forecast days ahead", 7, 90, 30)
        last_vals = X.tail(1).values[0]
        future_X = np.tile(last_vals, (future_days, 1))
        future_preds = model.predict(future_X)
        future_df = pd.DataFrame({'Day': range(1, future_days+1), 'Forecast': future_preds})
        fig4 = px.line(future_df, x='Day', y='Forecast', title=f"Next {future_days} Day Forecast",
                       color_discrete_sequence=[PASTEL_COLORS[4]])
        fig4.update_traces(fill='tozeroy', fillcolor='rgba(165,214,167,0.1)')
        fig4.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
        st.plotly_chart(fig4, use_container_width=True)

        st.session_state['future_forecast'] = future_df