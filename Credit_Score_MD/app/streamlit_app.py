"""
Streamlit web deployment for the credit score classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pipeline_local import clean_dataframe

st.set_page_config(
    page_title="Credit Score Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = ROOT / "models" / "model_best.pkl"
SCORE_COLOR = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
ENGINEERED_RATIO_FEATURES = {"Debt_to_Income", "EMI_to_Salary"}
TREE_BASED_KEYWORDS = (
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "DecisionTree",
    "XGB",
    "LGBM",
    "CatBoost",
)

CLIP_RANGES = {
    "Age": (14.0, 100.0),
    "Annual_Income": (0.0, 1_000_000.0),
    "Monthly_Inhand_Salary": (0.0, 100_000.0),
    "Num_Bank_Accounts": (0.0, 50.0),
    "Num_Credit_Card": (0.0, 50.0),
    "Interest_Rate": (0.0, 100.0),
    "Num_of_Loan": (0.0, 50.0),
    "Delay_from_due_date": (0.0, 90.0),
    "Num_of_Delayed_Payment": (0.0, 100.0),
    "Changed_Credit_Limit": (-50.0, 100.0),
    "Num_Credit_Inquiries": (0.0, 100.0),
    "Outstanding_Debt": (0.0, 1_000_000.0),
    "Credit_Utilization_Ratio": (0.0, 100.0),
    "Total_EMI_per_month": (0.0, 100_000.0),
    "Amount_invested_monthly": (0.0, 100_000.0),
    "Monthly_Balance": (-100_000.0, 100_000.0),
    "Credit_History_Age_Months": (0.0, 600.0),
    "Loan_Type_Count": (0.0, 50.0),
    "Debt_to_Income": (0.0, 100.0),
    "EMI_to_Salary": (0.0, 100.0),
}


@st.cache_resource(show_spinner="Loading model")
def load_model() -> dict[str, Any]:
    """Load the saved model artifact once per Streamlit session."""
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run the local pipeline first to create models/model_best.pkl.")
        st.stop()
    return joblib.load(MODEL_PATH)


artifact = load_model()
model = artifact["model"]
features = artifact["feature_columns"]
meta = artifact["metadata"]
classes = meta.get("classes", ["Good", "Poor", "Standard"])
numeric_features = meta.get("features", {}).get("numeric", [])
categorical_features = meta.get("features", {}).get("categorical", [])


def format_model_name(name: str) -> str:
    """Convert snake case model names to readable labels."""
    return str(name).replace("_", " ").title()


def align_features(X: pd.DataFrame) -> pd.DataFrame:
    """Align data to the feature schema used during training."""
    X = X.copy()
    for col in features:
        if col not in X.columns:
            X[col] = np.nan
    return X[features]


def predict_with_proba(X_clean: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
    """Return predictions and probabilities when the model supports them."""
    X_aligned = align_features(X_clean)
    predictions = model.predict(X_aligned)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_aligned)
        proba_df = pd.DataFrame(proba, columns=model.classes_, index=X_clean.index)
        return predictions, proba_df
    return predictions, None


def final_estimator() -> Any:
    """Return the model object inside the sklearn pipeline."""
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model")
    return model


def is_tree_based_model() -> bool:
    """Show importance only when the final estimator is tree-based and exposes importances."""
    estimator = final_estimator()
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return False
    return any(key in estimator.__class__.__name__ for key in TREE_BASED_KEYWORDS)


def readable_feature_name(name: str) -> str:
    """Clean transformed feature names for display."""
    for prefix in ["num__", "ordinal__", "onehot__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ")


def feature_importance_dataframe() -> pd.DataFrame:
    """Extract tree-based feature importance after preprocessing and feature selection."""
    if not is_tree_based_model() or not hasattr(model, "named_steps"):
        return pd.DataFrame(columns=["feature", "importance"])

    metadata_importance = meta.get("feature_importance", [])
    if metadata_importance:
        fi = pd.DataFrame(metadata_importance)
        if {"feature", "importance"}.issubset(fi.columns):
            fi["feature"] = fi["feature"].map(readable_feature_name)
            return fi.sort_values("importance", ascending=False)

    preprocessor = model.named_steps.get("preprocess")
    selector = model.named_steps.get("feature_selection")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None or not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        feature_names = [name for name, keep in zip(feature_names, mask) if keep]

    importances = np.asarray(estimator.feature_importances_)
    if len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    fi = pd.DataFrame(
        {
            "feature": [readable_feature_name(name) for name in feature_names],
            "importance": importances,
        }
    )
    return fi.sort_values("importance", ascending=False)


def score_badge(label: str) -> str:
    """Render a colored prediction label."""
    color = SCORE_COLOR.get(str(label), "#777777")
    return (
        f'<span style="background:{color};color:white;padding:5px 16px;'
        f'border-radius:12px;font-weight:700;font-size:1rem;">{label}</span>'
    )


def confidence_table(proba: pd.Series) -> pd.DataFrame:
    """Build a probability table for display."""
    return pd.DataFrame(
        {
            "Class": list(proba.index),
            "Probability": [round(float(value), 4) for value in proba.values],
        }
    )


def normalize_manual_row(row: dict[str, Any]) -> pd.DataFrame:
    """Normalize manual input before prediction."""
    df = pd.DataFrame([row])
    placeholders = {"", "_", "_______", "!@9#%8", "nan", "NaN", "None"}

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: "Unknown" if str(x).strip() in placeholders else x)

    numeric_like = [col for col in features if col not in categorical_features]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, (lower, upper) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    if {"Outstanding_Debt", "Annual_Income"}.issubset(df.columns) and "Debt_to_Income" in features:
        df["Debt_to_Income"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)

    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(df.columns) and "EMI_to_Salary" in features:
        df["EMI_to_Salary"] = df["Total_EMI_per_month"] / (df["Monthly_Inhand_Salary"] + 1)

    return align_features(df)


def validate_manual_row(row: dict[str, Any]) -> list[str]:
    """Return warnings for unrealistic manual values."""
    warnings = []
    for col, (lower, upper) in CLIP_RANGES.items():
        if col not in row or col in ENGINEERED_RATIO_FEATURES:
            continue
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.isna(value):
            warnings.append(f"{col} is not numeric and will be imputed by the pipeline.")
        elif value < lower or value > upper:
            warnings.append(f"{col} is outside the expected range and will be clipped before prediction.")
    return warnings


def prediction_summary(X: pd.DataFrame) -> dict[str, Any]:
    """Return prediction and probabilities for a single-row dataframe."""
    preds, proba_df = predict_with_proba(X)
    result = {"prediction": preds[0], "probabilities": None}
    if proba_df is not None:
        result["probabilities"] = proba_df.iloc[0]
    return result


def display_prediction_result(result: dict[str, Any]) -> None:
    """Display one prediction result."""
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Result")
        st.markdown(score_badge(result["prediction"]), unsafe_allow_html=True)
    if result["probabilities"] is not None:
        with right:
            st.markdown("### Confidence")
            proba_df = confidence_table(result["probabilities"])
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
            st.bar_chart(proba_df.set_index("Class"))


def display_model_comparison() -> None:
    """Show candidate model comparison in the sidebar."""
    experiments = meta.get("experiments", [])
    if not experiments:
        return

    st.markdown("### Model Comparison")
    st.caption(
        "This table shows the initial candidate model evaluation before final tuning. "
        "The metrics above summarize the final deployed model on the held-out test dataset."
    )
    exp_df = pd.DataFrame(experiments).rename(
        columns={
            "model": "Model",
            "cv_f1_weighted_mean": "CV Weighted F1 Mean",
            "cv_f1_weighted_std": "CV Weighted F1 Std",
            "test_accuracy": "Test Accuracy",
            "test_precision_weighted": "Test Precision",
            "test_recall_weighted": "Test Recall",
            "test_f1_weighted": "Test Weighted F1",
        }
    )
    exp_df["Model"] = exp_df["Model"].map(format_model_name)
    exp_df = exp_df.sort_values("Test Weighted F1", ascending=False)
    st.dataframe(exp_df.set_index("Model"), use_container_width=True)


DEFAULTS = {
    "Month": "June",
    "Age": 35,
    "Occupation": "Teacher",
    "Annual_Income": 60000.0,
    "Monthly_Inhand_Salary": 5000.0,
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 5,
    "Interest_Rate": 12,
    "Num_of_Loan": 2,
    "Delay_from_due_date": 15,
    "Num_of_Delayed_Payment": 5,
    "Changed_Credit_Limit": 7.5,
    "Num_Credit_Inquiries": 4,
    "Credit_Mix": "Standard",
    "Outstanding_Debt": 1200,
    "Credit_Utilization_Ratio": 30.5,
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 80.0,
    "Amount_invested_monthly": 200.0,
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Monthly_Balance": 400,
    "Credit_History_Age_Months": 180,
    "Loan_Type_Count": 2,
    "Debt_to_Income": 0.02,
    "EMI_to_Salary": 0.016,
}

CATEGORY_OPTIONS = {
    "Month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "Credit_Mix": ["Good", "Standard", "Bad", "Unknown"],
    "Payment_of_Min_Amount": ["Yes", "No", "NM", "Unknown"],
    "Payment_Behaviour": [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
        "Unknown",
    ],
}


def render_manual_form(prefix: str) -> dict[str, Any]:
    """Render the manual input form."""
    row: dict[str, Any] = {}
    columns = st.columns(3)
    for i, col in enumerate(features):
        container = columns[i % 3]
        default_value = DEFAULTS.get(col, 0)
        if col in CATEGORY_OPTIONS:
            options = CATEGORY_OPTIONS[col]
            index = options.index(str(default_value)) if str(default_value) in options else 0
            row[col] = container.selectbox(col, options, index=index, key=f"{prefix}_{col}")
        elif col in categorical_features:
            row[col] = container.text_input(col, str(default_value), key=f"{prefix}_{col}")
        else:
            label = col
            help_text = None
            if col in ENGINEERED_RATIO_FEATURES:
                label = f"{col} (auto recalculated)"
                help_text = "This value is recalculated before prediction."
            lower, upper = CLIP_RANGES.get(col, (None, None))
            kwargs = {
                "label": label,
                "value": float(default_value),
                "format": "%.4f",
                "key": f"{prefix}_{col}",
                "help": help_text,
            }
            if lower is not None:
                kwargs["min_value"] = float(lower)
            if upper is not None:
                kwargs["max_value"] = float(upper)
            row[col] = container.number_input(**kwargs)
    return row


with st.sidebar:
    st.title("Final Model and Model Card")
    st.caption(
        "Model card metrics show the final deployed model on the held-out test dataset. "
        "Model comparison shows the initial evaluation before final tuning."
    )
    st.metric("Final Deployed Model", format_model_name(meta.get("best_model", "-")))
    st.metric("Final Test Accuracy", f"{meta.get('test_accuracy', 0):.2%}")
    st.metric("Final Test Weighted F1", f"{meta.get('test_f1_weighted', 0):.4f}")

    st.markdown("### Per-Class Test Metrics")
    report = meta.get("classification_report", {})
    rows = []
    for cls in classes:
        class_metrics = report.get(cls, {})
        rows.append(
            {
                "Class": cls,
                "Precision": round(float(class_metrics.get("precision", 0)), 4),
                "Recall": round(float(class_metrics.get("recall", 0)), 4),
                "F1": round(float(class_metrics.get("f1-score", 0)), 4),
                "Support": int(class_metrics.get("support", 0)),
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    st.markdown("### Final Test Confusion Matrix")
    cm = meta.get("confusion_matrix", [])
    if cm:
        st.dataframe(pd.DataFrame(cm, index=classes, columns=classes), use_container_width=True)

    display_model_comparison()

st.title("Credit Score Classifier")
st.write(
    "This application predicts customer credit score labels using the trained machine learning model. "
    "The labels are Good, Poor, and Standard."
)

with st.expander("Model notes and limitations"):
    st.write(
        "The prediction is intended as decision support for credit performance assessment. "
        "Manual input values are validated, normalized, and aligned with the training feature schema."
    )
    st.write(
        "For production use, the model should be monitored for data drift, performance changes, "
        "and potential bias across customer segments."
    )

tab_names = ["CSV Upload", "Manual Entry", "What-If Analysis"]
show_importance = is_tree_based_model()
if show_importance:
    tab_names.append("Feature Importance")

tabs = st.tabs(tab_names)
tab_csv, tab_manual, tab_what_if = tabs[:3]
tab_importance = tabs[3] if show_importance else None

with tab_csv:
    st.subheader("Upload a CSV file")
    st.caption("The file should use the same schema as the training data. The target column is ignored if it exists.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.info(f"{len(raw):,} rows and {raw.shape[1]} columns detected.")
        X_clean = clean_dataframe(raw)
        predictions, proba_df = predict_with_proba(X_clean)
        output = raw.copy()
        output.insert(0, "Predicted_Credit_Score", predictions)

        if proba_df is not None:
            insert_position = 1
            for cls in proba_df.columns:
                output.insert(insert_position, f"Prob_{cls}", proba_df[cls].round(4).values)
                insert_position += 1

        counts = pd.Series(predictions).value_counts()
        metric_columns = st.columns(len(classes))
        for i, cls in enumerate(classes):
            n = int(counts.get(cls, 0))
            metric_columns[i].metric(cls, f"{n:,}", f"{n / len(predictions):.1%} of total")

        st.dataframe(output.head(200), use_container_width=True)
        st.download_button(
            label="Download prediction results",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="credit_score_predictions.csv",
            mime="text/csv",
        )

with tab_manual:
    st.subheader("Enter a single customer profile")
    st.caption("Manual inputs are validated and normalized before prediction.")
    manual_row = render_manual_form("manual")

    if st.button("Predict", type="primary"):
        warnings = validate_manual_row(manual_row)
        for warning in warnings:
            st.warning(warning)
        X_input = normalize_manual_row(manual_row)
        display_prediction_result(prediction_summary(X_input))

with tab_what_if:
    st.subheader("What-If Analysis")
    st.caption("Change one direct numeric feature and compare the prediction against the base profile.")
    base_row = render_manual_form("what_if")
    numeric_choices = [col for col in features if col not in categorical_features and col not in ENGINEERED_RATIO_FEATURES]
    left, right = st.columns(2)
    feature_to_change = left.selectbox("Feature to change", numeric_choices)
    change_percent = right.slider("Percentage change", min_value=-80, max_value=200, value=10, step=5)

    if st.button("Run what-if comparison", type="primary"):
        scenario_row = dict(base_row)
        base_value = pd.to_numeric(pd.Series([scenario_row.get(feature_to_change, 0)]), errors="coerce").iloc[0]
        if pd.isna(base_value):
            base_value = 0.0
        scenario_row[feature_to_change] = float(base_value) * (1 + change_percent / 100)

        base_X = normalize_manual_row(base_row)
        scenario_X = normalize_manual_row(scenario_row)
        base_result = prediction_summary(base_X)
        scenario_result = prediction_summary(scenario_X)

        comparison = pd.DataFrame(
            [
                {
                    "Scenario": "Base",
                    "Changed Feature": feature_to_change,
                    "Feature Value": base_X.iloc[0][feature_to_change],
                    "Prediction": base_result["prediction"],
                },
                {
                    "Scenario": f"What-if ({change_percent:+d}%)",
                    "Changed Feature": feature_to_change,
                    "Feature Value": scenario_X.iloc[0][feature_to_change],
                    "Prediction": scenario_result["prediction"],
                },
            ]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        if base_result["probabilities"] is not None and scenario_result["probabilities"] is not None:
            proba_compare = pd.DataFrame(
                {
                    "Base": base_result["probabilities"],
                    f"What-if ({change_percent:+d}%)": scenario_result["probabilities"],
                }
            )
            st.subheader("Probability comparison")
            st.dataframe(proba_compare, use_container_width=True)
            st.bar_chart(proba_compare)

if show_importance and tab_importance is not None:
    with tab_importance:
        st.subheader("Top Feature Importances")
        st.caption("This tab appears only when the final deployed model is a tree-based model with feature importances.")
        fi_df = feature_importance_dataframe()
        if fi_df.empty:
            st.info("Feature importance could not be extracted from the final tree-based model.")
        else:
            top_n = st.slider("Number of features to show", min_value=5, max_value=30, value=15, step=5)
            top_fi = fi_df.head(top_n).sort_values("importance", ascending=True)
            st.bar_chart(top_fi.set_index("feature")["importance"])
            st.dataframe(fi_df.head(top_n), use_container_width=True, hide_index=True)
