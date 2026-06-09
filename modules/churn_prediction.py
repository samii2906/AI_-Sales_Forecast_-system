import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from utils.theme import inject_css, PASTEL_COLORS, metric_card

def show():
    inject_css()
    st.markdown('<div class="section-header"><h2>🔄 Churn Prediction</h2></div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df'].copy()
    all_cols = df.columns.tolist()

    st.markdown("### ⚙️ Configuration")
    c1, c2, c3 = st.columns(3)
    target_col = c1.selectbox("🎯 Churn Target Column", all_cols,
                               index=all_cols.index('Churn') if 'Churn' in all_cols else 0)
    feature_cols = c2.multiselect("📥 Feature Columns", [c for c in all_cols if c != target_col],
                                   default=[c for c in all_cols if c != target_col][:6])
    model_choice = c3.selectbox("🤖 Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])

    use_smote = st.checkbox("⚖️ Balance classes with SMOTE", value=True)

    if not feature_cols:
        st.info("Select at least one feature column.")
        return

    if st.button("🚀 Train Churn Model", use_container_width=True):
        with st.spinner("Training churn model..."):
            # Encode categoricals
            X = df[feature_cols].copy()
            for col in X.select_dtypes(include='object').columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(0)

            y = df[target_col].copy()
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y.astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                  random_state=42, stratify=y)
            if use_smote:
                try:
                    sm = SMOTE(random_state=42)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                except:
                    pass

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500)
            }
            model = models[model_choice]
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else None

            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True)
            cm = confusion_matrix(y_test, preds)

            st.session_state['churn_model'] = model
            st.session_state['churn_scaler'] = scaler
            st.session_state['churn_features'] = feature_cols
            st.session_state['churn_preds'] = preds
            st.session_state['churn_proba'] = proba
            st.session_state['churn_actual'] = y_test

        st.success("✅ Churn model trained!")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(metric_card("Accuracy", f"{acc*100:.1f}%"), unsafe_allow_html=True)
        col2.markdown(metric_card("Precision", f"{report['1']['precision']*100:.1f}%", color="#80cbc4"), unsafe_allow_html=True)
        col3.markdown(metric_card("Recall", f"{report['1']['recall']*100:.1f}%", color="#a5d6a7"), unsafe_allow_html=True)
        col4.markdown(metric_card("F1-Score", f"{report['1']['f1-score']*100:.1f}%", color="#ffe082"), unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results Table", "🥧 Churn Breakdown", "📉 Confusion Matrix", "📈 ROC Curve"])

        with tab1:
            results_df = X_test.copy()
            results_df['Actual_Churn'] = y_test
            results_df['Predicted_Churn'] = preds
            if proba is not None:
                results_df['Churn_Probability'] = (proba * 100).round(2)
            results_df['Status'] = results_df['Predicted_Churn'].map({1: '🔴 High Risk', 0: '🟢 Retained'})
            st.dataframe(results_df.head(100), use_container_width=True)

            csv = results_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Churn Results", csv, "churn_results.csv")

        with tab2:
            churn_counts = pd.Series(preds).value_counts().rename({0: 'Retained', 1: 'Churned'})
            fig_pie = px.pie(values=churn_counts.values, names=churn_counts.index,
                             color_discrete_sequence=[PASTEL_COLORS[0], PASTEL_COLORS[2]],
                             hole=0.5, title="Predicted Churn Distribution")
            fig_pie.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0")
            st.plotly_chart(fig_pie, use_container_width=True)

            if proba is not None:
                fig_hist = px.histogram(x=proba, nbins=30, title="Churn Probability Distribution",
                                        color_discrete_sequence=[PASTEL_COLORS[1]])
                fig_hist.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab3:
            fig_cm = px.imshow(cm, text_auto=True,
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Retained', 'Churned'], y=['Retained', 'Churned'],
                               color_continuous_scale='Purples',
                               title="Confusion Matrix")
            fig_cm.update_layout(paper_bgcolor="#0e1117", font_color="#e0e0e0")
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab4:
            if proba is not None:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy',
                                              fillcolor='rgba(179,157,219,0.15)',
                                              line=dict(color=PASTEL_COLORS[0], width=2),
                                              name=f'AUC = {roc_auc:.3f}'))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Random'))
                fig_roc.update_layout(title="ROC Curve", paper_bgcolor="#0e1117",
                                       plot_bgcolor="#1a1d2e", font_color="#e0e0e0",
                                       xaxis_title="False Positive Rate",
                                       yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
            fi = fi.sort_values('Importance', ascending=True)
            fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                             title="Churn Driver Importance",
                             color='Importance', color_continuous_scale='Purples')
            fig_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", font_color="#e0e0e0")
            st.plotly_chart(fig_fi, use_container_width=True)