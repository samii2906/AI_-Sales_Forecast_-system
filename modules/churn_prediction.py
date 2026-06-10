import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.theme import inject_css, metric_card, section_header, plotly_dark_layout, PASTEL_COLORS

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def is_valid_target(series):
    """Check if a column is suitable as a binary/classification target."""
    unique_vals = series.dropna().unique()
    return len(unique_vals) <= 20  # at most 20 classes


def show():
    inject_css()
    section_header("Churn Prediction", "🔄")

    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state['df'].copy()
    all_cols = df.columns.tolist()

    # ── Only show columns suitable as classification targets (≤20 unique values)
    valid_target_cols = [c for c in all_cols if is_valid_target(df[c])]

    if not valid_target_cols:
        st.error("❌ No suitable target column found. Need a column with ≤ 20 unique values (e.g. Churn = 0/1).")
        return

    st.markdown("### ⚙️ Model Configuration")

    # ── Warn user what a valid target looks like
    st.info("💡 **Tip:** Select a binary column as target (e.g. Churn = 0 or 1). Columns with too many unique values are hidden.")

    c1, c2, c3 = st.columns(3)

    # Auto-select 'Churn' if available, else first valid target
    if 'Churn' in valid_target_cols:
        default_idx = valid_target_cols.index('Churn')
    else:
        default_idx = 0

    target_col = c1.selectbox(
        "🎯 Churn Target Column",
        valid_target_cols,
        index=default_idx,
        help="Pick a column with 2 classes (0/1 or Yes/No)"
    )

    # Show unique values of selected target
    unique_target = sorted(df[target_col].dropna().unique().tolist())
    st.markdown(f"**Target column unique values:** `{unique_target}` — {len(unique_target)} classes")

    if len(unique_target) < 2:
        st.error("❌ Target column has less than 2 unique values. Please choose a different column.")
        return

    feat_cols = c2.multiselect(
        "📥 Feature Columns",
        [c for c in all_cols if c != target_col],
        default=[c for c in all_cols if c != target_col and
                 df[c].dtype in ['int64', 'float64', 'object']][:8]
    )
    model_name = c3.selectbox(
        "🤖 Algorithm",
        ["Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression"]
    )

    adv1, adv2, adv3 = st.columns(3)
    use_smote = adv1.checkbox("⚖️ SMOTE Class Balancing", value=SMOTE_AVAILABLE)
    use_cv    = adv2.checkbox("5-Fold Cross Validation", value=True)
    threshold = adv3.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    if not feat_cols:
        st.info("Select at least one feature column.")
        return

    if st.button("🚀 Train Churn Model", use_container_width=True):
        with st.spinner("Training churn prediction model..."):

            # ── Encode features
            X = df[feat_cols].copy()
            for col in X.select_dtypes(include='object').columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(0)

            # ── Encode target → always produce 0, 1, 2... integers
            y_raw = df[target_col].copy().fillna('Unknown')
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))

            n_classes = len(np.unique(y))
            is_binary = n_classes == 2

            st.markdown(f"**Detected {n_classes} class(es):** `{le_target.classes_.tolist()}`")

            # ── Safe stratify
            class_counts = Counter(y)
            can_stratify = all(v >= 2 for v in class_counts.values())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y if can_stratify else None
            )

            # ── SMOTE (binary only)
            if use_smote and SMOTE_AVAILABLE and is_binary:
                try:
                    min_samples = min(Counter(y_train).values())
                    if min_samples >= 6:
                        sm = SMOTE(random_state=42)
                        X_train, y_train = sm.fit_resample(X_train, y_train)
                        st.success("✅ SMOTE applied successfully.")
                    else:
                        st.warning(f"SMOTE skipped: not enough samples (min class = {min_samples}).")
                except Exception as e:
                    st.warning(f"SMOTE skipped: {e}")

            # ── Scale
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            # ── Model
            model_map = {
                "XGBoost": XGBClassifier(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=5, random_state=42,
                    verbosity=0, eval_metric='logloss',
                    num_class=n_classes if n_classes > 2 else None
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=150, random_state=42, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=150, random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, C=1.0,
                    multi_class='auto'
                ),
            }

            # Clean XGBoost params for binary case
            if model_name == "XGBoost":
                xgb_params = dict(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=5, random_state=42,
                    verbosity=0, eval_metric='logloss'
                )
                model = XGBClassifier(**xgb_params)
            else:
                model = model_map[model_name]

            model.fit(X_tr_s, y_train)

            proba_all = model.predict_proba(X_te_s) if hasattr(model, 'predict_proba') else None
            preds = model.predict(X_te_s)

            if is_binary and proba_all is not None:
                proba = proba_all[:, 1]
                preds = (proba >= threshold).astype(int)
            else:
                proba = None

            acc    = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            cm     = confusion_matrix(y_test, preds)

            # ── Safe CV
            cv_scores = None
            if use_cv:
                min_class_count = min(Counter(y).values())
                cv_folds = min(5, min_class_count)
                if cv_folds >= 2:
                    try:
                        scoring = 'roc_auc' if is_binary else 'accuracy'
                        cv_scores = cross_val_score(
                            model, scaler.transform(X), y,
                            cv=cv_folds, scoring=scoring
                        )
                    except Exception as e:
                        st.warning(f"Cross-validation skipped: {e}")
                else:
                    st.warning("Not enough samples per class for cross-validation.")

            # Save to session
            st.session_state.update({
                'churn_model':    model,
                'churn_scaler':   scaler,
                'churn_features': feat_cols,
                'churn_preds':    preds,
                'churn_proba':    proba,
                'churn_actual':   y_test,
                'churn_X_test':   X_test,
                'churn_metrics':  {
                    'Accuracy':  acc,
                    'Precision': report.get('1', report.get('weighted avg', {})).get('precision', 0),
                    'Recall':    report.get('1', report.get('weighted avg', {})).get('recall', 0),
                    'F1-Score':  report.get('1', report.get('weighted avg', {})).get('f1-score', 0),
                }
            })

        st.success(f"✅ **{model_name}** trained on `{target_col}` — {n_classes} classes detected!")

        # ── Metrics
        c1, c2, c3, c4 = st.columns(4)
        prec = report.get('1', report.get('weighted avg', {})).get('precision', 0)
        rec  = report.get('1', report.get('weighted avg', {})).get('recall', 0)
        f1   = report.get('1', report.get('weighted avg', {})).get('f1-score', 0)

        c1.markdown(metric_card("Accuracy",  f"{acc*100:.1f}%",      icon="🎯"),            unsafe_allow_html=True)
        c2.markdown(metric_card("Precision", f"{prec*100:.1f}%",     color="#80cbc4", icon="🔍"), unsafe_allow_html=True)
        c3.markdown(metric_card("Recall",    f"{rec*100:.1f}%",      color="#a5d6a7", icon="📡"), unsafe_allow_html=True)
        c4.markdown(metric_card("F1-Score",  f"{f1*100:.1f}%",       color="#ffe082", icon="⚖️"), unsafe_allow_html=True)

        if cv_scores is not None:
            scoring_label = 'AUC-ROC' if is_binary else 'Accuracy'
            st.markdown(
                f"**{len(cv_scores)}-Fold CV {scoring_label}:** "
                f"Mean = `{cv_scores.mean():.4f}` | Std = `{cv_scores.std():.4f}`"
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Results Table", "🥧 Churn Breakdown",
            "📉 Confusion Matrix", "📈 ROC Curve", "🎯 PR Curve"
        ])

        # ── Tab 1: Results table
        with tab1:
            results_df = X_test.copy().reset_index(drop=True)
            results_df['Actual']    = le_target.inverse_transform(y_test if hasattr(y_test, '__iter__') else [y_test])
            results_df['Predicted'] = le_target.inverse_transform(preds)
            if proba is not None:
                results_df['Churn_Probability_%'] = (proba * 100).round(2)
            results_df['Risk_Level'] = results_df['Predicted'].astype(str).map(
                lambda v: '🔴 High Risk' if str(v) in ['1', 'Yes', 'True', '1.0'] else '🟢 Retained'
            )
            results_df['Correct'] = (results_df['Actual'] == results_df['Predicted']).map(
                {True: '✅', False: '❌'}
            )
            st.dataframe(results_df.head(200), use_container_width=True)
            csv = results_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Churn Results CSV", csv, "churn_predictions.csv", "text/csv")

        # ── Tab 2: Breakdown charts
        with tab2:
            pred_labels = le_target.inverse_transform(preds)
            churn_counts = pd.Series(pred_labels).value_counts().reset_index()
            churn_counts.columns = ['Class', 'Count']

            c_pie, c_bar = st.columns(2)
            with c_pie:
                fig_pie = px.pie(
                    churn_counts, values='Count', names='Class',
                    color_discrete_sequence=PASTEL_COLORS,
                    hole=0.5, title="Predicted Class Distribution"
                )
                fig_pie.update_traces(textinfo='percent+label+value')
                fig_pie.update_layout(**plotly_dark_layout())
                st.plotly_chart(fig_pie, use_container_width=True)

            with c_bar:
                if proba is not None:
                    risk_bins = pd.cut(
                        proba, bins=[0, 0.3, 0.6, 1.0],
                        labels=['Low Risk', 'Medium Risk', 'High Risk']
                    )
                    rb_counts = risk_bins.value_counts().reset_index()
                    rb_counts.columns = ['Risk Level', 'Count']
                    fig_risk = px.bar(
                        rb_counts, x='Risk Level', y='Count',
                        color='Risk Level',
                        color_discrete_sequence=[PASTEL_COLORS[4], PASTEL_COLORS[3], PASTEL_COLORS[2]],
                        title="Customer Risk Tiers", text='Count'
                    )
                    fig_risk.update_traces(textposition='outside')
                    fig_risk.update_layout(**plotly_dark_layout())
                    st.plotly_chart(fig_risk, use_container_width=True)

            if proba is not None:
                fig_hist = px.histogram(
                    x=proba, nbins=40,
                    title="Churn Probability Distribution",
                    color_discrete_sequence=[PASTEL_COLORS[0]],
                    labels={'x': 'Churn Probability'}
                )
                fig_hist.add_vline(x=threshold, line_dash="dash",
                                   line_color=PASTEL_COLORS[2],
                                   annotation_text=f"Threshold: {threshold}")
                fig_hist.update_layout(**plotly_dark_layout())
                st.plotly_chart(fig_hist, use_container_width=True)

        # ── Tab 3: Confusion matrix
        with tab3:
            class_labels = le_target.classes_.tolist()
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto",
                labels=dict(x="Predicted", y="Actual"),
                x=class_labels, y=class_labels,
                color_continuous_scale='Purples',
                title="Confusion Matrix"
            )
            fig_cm.update_layout(**plotly_dark_layout())
            st.plotly_chart(fig_cm, use_container_width=True)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.markdown(metric_card("True Positives",  tp, color="#a5d6a7", icon="✅"), unsafe_allow_html=True)
                cc2.markdown(metric_card("True Negatives",  tn, color="#90caf9", icon="✅"), unsafe_allow_html=True)
                cc3.markdown(metric_card("False Positives", fp, color="#ffe082", icon="⚠️"), unsafe_allow_html=True)
                cc4.markdown(metric_card("False Negatives", fn, color="#f48fb1", icon="❌"), unsafe_allow_html=True)

        # ── Tab 4: ROC Curve (binary only)
        with tab4:
            if proba is not None and is_binary:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'AUC = {roc_auc:.4f}',
                    line=dict(color=PASTEL_COLORS[0], width=2.5),
                    fill='tozeroy', fillcolor='rgba(179,157,219,0.08)'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    line=dict(dash='dash', color='gray', width=1.5),
                    name='Random Classifier'
                ))
                fig_roc.update_layout(
                    **plotly_dark_layout(),
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                quality = 'Excellent 🌟' if roc_auc > 0.9 else 'Good ✅' if roc_auc > 0.8 else 'Moderate ⚠️'
                st.markdown(f"**AUC-ROC Score: `{roc_auc:.4f}`** — {quality}")
            else:
                st.info("ROC Curve is available for binary classification only.")

        # ── Tab 5: PR Curve (binary only)
        with tab5:
            if proba is not None and is_binary:
                precision_vals, recall_vals, _ = precision_recall_curve(y_test, proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall_vals, y=precision_vals,
                    mode='lines', name='Precision-Recall',
                    line=dict(color=PASTEL_COLORS[1], width=2.5),
                    fill='tozeroy', fillcolor='rgba(128,203,196,0.08)'
                ))
                fig_pr.update_layout(
                    **plotly_dark_layout(),
                    title="Precision-Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision"
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("PR Curve is available for binary classification only.")

        # ── Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🔑 Churn Driver Analysis")
            fi = pd.DataFrame({
                'Feature':    feat_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance')
            fig_fi = px.bar(
                fi, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Purples',
                title="Top Churn Drivers"
            )
            fig_fi.update_layout(**plotly_dark_layout())
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── High-Risk Customer List (binary only)
        if proba is not None and is_binary:
            st.markdown("### 🚨 High-Risk Customer List")
            risk_threshold = st.slider(
                "Show customers with churn probability above (%)", 50, 95, 70
            )
            hr_df = X_test.copy().reset_index(drop=True)
            hr_df['Churn_Probability_%'] = (proba * 100).round(2)
            hr_df['Risk_Level'] = pd.cut(
                proba * 100,
                bins=[0, 30, 60, 100],
                labels=['🟢 Low', '🟡 Medium', '🔴 High']
            )
            hr_filtered = hr_df[
                hr_df['Churn_Probability_%'] >= risk_threshold
            ].sort_values('Churn_Probability_%', ascending=False)

            st.markdown(f"**{len(hr_filtered):,} customers** above {risk_threshold}% churn risk")
            st.dataframe(hr_filtered.head(100), use_container_width=True)
            csv_hr = hr_filtered.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download High-Risk List", csv_hr,
                "high_risk_customers.csv", "text/csv"
            )