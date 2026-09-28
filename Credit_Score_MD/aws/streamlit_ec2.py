"""
streamlit_ec2.py - Versi Streamlit untuk deployment di EC2.

Perbedaan dari app/streamlit_app.py:
- Path model diambil dari MODEL_PATH env variable atau default ke model_best.pkl
  di folder yang sama dengan script ini (bukan relatif ke root project)
- Import clean_dataframe disalin langsung (tidak import dari src/)
  agar bisa jalan di EC2 tanpa harus setup sys.path

Cara pakai di EC2 (dari root project yang sudah di-clone):
    streamlit run aws/streamlit_ec2.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

def _find_model_path() -> Path:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return Path(env_path)

    # Root project = dua level di atas file ini (aws/ -> root)
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "aws" / "model_artifact" / "model_best.pkl",
        root / "models" / "model_best.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback ke path pertama (akan trigger error yang jelas)
    return candidates[0]


MODEL_PATH = _find_model_path()


# clean_dataframe (disalin agar tidak import dari src/)

TARGET_COLUMN = "Credit_Score"
PLACEHOLDER_COLS = ["Occupation", "Credit_Mix", "Payment_Behaviour",
                    "Payment_of_Min_Amount", "Month"]
PLACEHOLDER_VALUES = {"_______", "_", "!@9#%8", "nan", "NaN", "", "None"}
DIRTY_NUMERIC_COLS = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                      "Changed_Credit_Limit", "Outstanding_Debt",
                      "Amount_invested_monthly", "Monthly_Balance"]
DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN",
             "Type_of_Loan", "Credit_History_Age", TARGET_COLUMN]
INFERENCE_CLIP_RANGES = {
    "Age": (14.0, 100.0), "Annual_Income": (0.0, 1_000_000.0),
    "Monthly_Inhand_Salary": (0.0, 100_000.0), "Num_Bank_Accounts": (0.0, 50.0),
    "Num_Credit_Card": (0.0, 50.0), "Interest_Rate": (0.0, 100.0),
    "Num_of_Loan": (0.0, 50.0), "Delay_from_due_date": (0.0, 90.0),
    "Num_of_Delayed_Payment": (0.0, 100.0), "Changed_Credit_Limit": (-50.0, 100.0),
    "Num_Credit_Inquiries": (0.0, 100.0), "Outstanding_Debt": (0.0, 1_000_000.0),
    "Credit_Utilization_Ratio": (0.0, 100.0), "Total_EMI_per_month": (0.0, 100_000.0),
    "Amount_invested_monthly": (0.0, 100_000.0), "Monthly_Balance": (-100_000.0, 100_000.0),
    "Debt_to_Income": (0.0, 100.0), "EMI_to_Salary": (0.0, 100.0),
}


def _to_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if re.fullmatch(r"_+[^_]*_+", text):
        return np.nan
    text = text.rstrip("_")
    if text in {"", "nan", "NaN", "None"}:
        return np.nan
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        return float(text)
    except ValueError:
        return np.nan


def _credit_history_to_months(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(value))
    if not match:
        return np.nan
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def _loan_type_count(value) -> int:
    if pd.isna(value):
        return 0
    return len(re.findall(r"Loan", str(value)))


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PLACEHOLDER_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].apply(lambda x: "Unknown" if x in PLACEHOLDER_VALUES else x)
    for col in DIRTY_NUMERIC_COLS:
        if col in out.columns:
            out[col] = out[col].apply(_to_number)
    if "Credit_History_Age" in out.columns:
        out["Credit_History_Age_Months"] = out["Credit_History_Age"].apply(_credit_history_to_months)
    if "Type_of_Loan" in out.columns:
        out["Loan_Type_Count"] = out["Type_of_Loan"].apply(_loan_type_count)
    if {"Outstanding_Debt", "Annual_Income"}.issubset(out.columns):
        out["Debt_to_Income"] = out["Outstanding_Debt"] / (out["Annual_Income"] + 1)
    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(out.columns):
        out["EMI_to_Salary"] = out["Total_EMI_per_month"] / (out["Monthly_Inhand_Salary"] + 1)
    for col, (lower, upper) in INFERENCE_CLIP_RANGES.items():
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].clip(lower=lower, upper=upper)
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    return out


# Streamlit app - identik dengan app/streamlit_app.py

st.set_page_config(
    page_title="Credit Score Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCORE_COLOR = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
ENGINEERED_RATIO_FEATURES = {"Debt_to_Income", "EMI_to_Salary"}
TREE_BASED_KEYWORDS = (
    "RandomForest", "ExtraTrees", "GradientBoosting",
    "DecisionTree", "XGB", "LGBM", "CatBoost",
)
CLIP_RANGES = INFERENCE_CLIP_RANGES


@st.cache_resource(show_spinner="Loading model")
def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        st.error(
            f"Model tidak ditemukan di {MODEL_PATH}. "
            "Jalankan pipeline_sagemaker.py dulu."
        )
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
    return str(name).replace("_", " ").title()


def align_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in features:
        if col not in X.columns:
            X[col] = np.nan
    return X[features]


def predict_with_proba(X_clean: pd.DataFrame):
    X_aligned = align_features(X_clean)
    predictions = model.predict(X_aligned)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_aligned)
        proba_df = pd.DataFrame(proba, columns=model.classes_, index=X_clean.index)
        return predictions, proba_df
    return predictions, None


def final_estimator() -> Any:
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model")
    return model


def is_tree_based_model() -> bool:
    estimator = final_estimator()
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return False
    return any(key in estimator.__class__.__name__ for key in TREE_BASED_KEYWORDS)


def readable_feature_name(name: str) -> str:
    for prefix in ["num__", "ordinal__", "onehot__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ")


def feature_importance_dataframe() -> pd.DataFrame:
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
    if preprocessor is None or estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])
    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        feature_names = [n for n, keep in zip(feature_names, mask) if keep]
    importances = np.asarray(estimator.feature_importances_)
    if len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])
    fi = pd.DataFrame({
        "feature": [readable_feature_name(n) for n in feature_names],
        "importance": importances,
    })
    return fi.sort_values("importance", ascending=False)


def score_badge(label: str) -> str:
    color = SCORE_COLOR.get(str(label), "#777777")
    return (
        f'<span style="background:{color};color:white;padding:5px 16px;'
        f'border-radius:12px;font-weight:700;font-size:1rem;">{label}</span>'
    )


def confidence_table(proba: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Class": list(proba.index),
        "Probability": [round(float(v), 4) for v in proba.values],
    })


def prediction_summary(X_input: pd.DataFrame) -> dict[str, Any]:
    preds, proba_df = predict_with_proba(X_input)
    prediction = str(preds[0])
    probabilities = None
    if proba_df is not None:
        probabilities = proba_df.iloc[0]
    return {"prediction": prediction, "probabilities": probabilities}


def display_prediction_result(result: dict[str, Any]) -> None:
    st.markdown(f"### Prediction: {score_badge(result['prediction'])}", unsafe_allow_html=True)
    if result["probabilities"] is not None:
        st.markdown("**Class Probabilities:**")
        st.dataframe(confidence_table(result["probabilities"]), use_container_width=True, hide_index=True)


def validate_manual_row(row: dict[str, Any]) -> list[str]:
    warnings_list = []
    if row.get("Annual_Income", 1) == 0:
        warnings_list.append("Annual Income bernilai 0. Pastikan ini benar.")
    return warnings_list


def normalize_manual_row(row: dict[str, Any]) -> pd.DataFrame:
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
    X_clean = clean_dataframe(df)
    return X_clean


def display_model_comparison() -> None:
    experiments = meta.get("experiments", [])
    if not experiments:
        return
    st.markdown("### Model Comparison (pre-tuning)")
    exp_df = pd.DataFrame(experiments)
    display_cols = [c for c in ["model", "cv_f1_weighted_mean", "test_f1_weighted", "test_accuracy"] if c in exp_df.columns]
    st.dataframe(exp_df[display_cols], use_container_width=True, hide_index=True)


DEFAULTS: dict[str, Any] = {
    "Age": 30, "Annual_Income": 50000.0, "Monthly_Inhand_Salary": 4000.0,
    "Num_Bank_Accounts": 2, "Num_Credit_Card": 3, "Interest_Rate": 12.0,
    "Num_of_Loan": 1, "Delay_from_due_date": 5, "Num_of_Delayed_Payment": 2,
    "Changed_Credit_Limit": 10.0, "Num_Credit_Inquiries": 1,
    "Credit_Mix": "Standard", "Outstanding_Debt": 1500.0,
    "Credit_Utilization_Ratio": 30.0, "Credit_History_Age_Months": 268.0,
    "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 300.0,
    "Amount_invested_monthly": 100.0,
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Monthly_Balance": 500.0, "Loan_Type_Count": 2,
    "Occupation": "Scientist", "Month": "January",
    "Debt_to_Income": 0.03, "EMI_to_Salary": 0.07,
}

CATEGORY_OPTIONS: dict[str, list[str]] = {
    "Credit_Mix": ["Bad", "Standard", "Good", "Unknown"],
    "Payment_of_Min_Amount": ["No", "NM", "Yes", "Unknown"],
    "Occupation": [
        "Accountant", "Architect", "Developer", "Doctor", "Engineer",
        "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
        "Media_Manager", "Musician", "Nurse", "Scientist", "Teacher",
        "Writer", "Unknown",
    ],
    "Month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "Payment_Behaviour": [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
        "Unknown",
    ],
}


def render_manual_form(prefix: str) -> dict[str, Any]:
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
                help_text = "Nilai ini dihitung ulang otomatis sebelum prediksi."
            lower, upper = CLIP_RANGES.get(col, (None, None))
            kwargs = {
                "label": label, "value": float(default_value),
                "format": "%.4f", "key": f"{prefix}_{col}", "help": help_text,
            }
            if lower is not None:
                kwargs["min_value"] = float(lower)
            if upper is not None:
                kwargs["max_value"] = float(upper)
            row[col] = container.number_input(**kwargs)
    return row

# Sidebar

with st.sidebar:
    st.title("Final Model and Model Card")
    st.caption("Metrics dari model final pada held-out test dataset.")
    st.metric("Final Deployed Model", format_model_name(meta.get("best_model", "-")))
    st.metric("Final Test Accuracy", f"{meta.get('test_accuracy', 0):.2%}")
    st.metric("Final Test Weighted F1", f"{meta.get('test_f1_weighted', 0):.4f}")

    st.markdown("### Per-Class Test Metrics")
    report = meta.get("classification_report", {})
    rows = []
    for cls in classes:
        cm = report.get(cls, {})
        rows.append({
            "Class": cls,
            "Precision": round(float(cm.get("precision", 0)), 4),
            "Recall": round(float(cm.get("recall", 0)), 4),
            "F1": round(float(cm.get("f1-score", 0)), 4),
            "Support": int(cm.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    st.markdown("### Final Test Confusion Matrix")
    cm_data = meta.get("confusion_matrix", [])
    if cm_data:
        st.dataframe(
            pd.DataFrame(cm_data, index=classes, columns=classes),
            use_container_width=True,
        )

    display_model_comparison()

# Main content

st.title("Credit Score Classifier")
st.write(
    "Aplikasi prediksi credit score nasabah menggunakan model machine learning. "
    "Label prediksi: Good, Standard, Poor."
)

with st.expander("Model notes and limitations"):
    st.write(
        "Prediksi ini adalah decision support untuk penilaian kredit. "
        "Input manual divalidasi dan dinormalisasi sebelum prediksi."
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
    st.caption("Schema harus sama dengan data training. Kolom target diabaikan kalau ada.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.info(f"{len(raw):,} baris, {raw.shape[1]} kolom.")
        X_clean = clean_dataframe(raw)
        predictions, proba_df = predict_with_proba(X_clean)
        output = raw.copy()
        output.insert(0, "Predicted_Credit_Score", predictions)
        if proba_df is not None:
            pos = 1
            for cls in proba_df.columns:
                output.insert(pos, f"Prob_{cls}", proba_df[cls].round(4).values)
                pos += 1
        counts = pd.Series(predictions).value_counts()
        metric_cols = st.columns(len(classes))
        for i, cls in enumerate(classes):
            n = int(counts.get(cls, 0))
            metric_cols[i].metric(cls, f"{n:,}", f"{n / len(predictions):.1%}")
        st.dataframe(output.head(200), use_container_width=True)
        st.download_button(
            "Download prediction results",
            output.to_csv(index=False).encode("utf-8"),
            "credit_score_predictions.csv",
            "text/csv",
        )

with tab_manual:
    st.subheader("Enter a single customer profile")
    st.caption("Input divalidasi dan dinormalisasi sebelum prediksi.")
    manual_row = render_manual_form("manual")
    if st.button("Predict", type="primary"):
        warnings_list = validate_manual_row(manual_row)
        for w in warnings_list:
            st.warning(w)
        X_input = normalize_manual_row(manual_row)
        display_prediction_result(prediction_summary(X_input))

with tab_what_if:
    st.subheader("What-If Analysis")
    st.caption("Ubah satu fitur numerik dan bandingkan prediksinya.")
    base_row = render_manual_form("what_if")
    numeric_choices = [c for c in features if c not in categorical_features and c not in ENGINEERED_RATIO_FEATURES]
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
        comparison = pd.DataFrame([
            {"Scenario": "Base", "Changed Feature": feature_to_change,
             "Feature Value": base_X.iloc[0][feature_to_change],
             "Prediction": base_result["prediction"]},
            {"Scenario": f"What-if ({change_percent:+d}%)", "Changed Feature": feature_to_change,
             "Feature Value": scenario_X.iloc[0][feature_to_change],
             "Prediction": scenario_result["prediction"]},
        ])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if base_result["probabilities"] is not None and scenario_result["probabilities"] is not None:
            proba_compare = pd.DataFrame({
                "Base": base_result["probabilities"],
                f"What-if ({change_percent:+d}%)": scenario_result["probabilities"],
            })
            st.subheader("Probability comparison")
            st.dataframe(proba_compare, use_container_width=True)
            st.bar_chart(proba_compare)

if show_importance and tab_importance is not None:
    with tab_importance:
        st.subheader("Top Feature Importances")
        fi_df = feature_importance_dataframe()
        if fi_df.empty:
            st.info("Feature importance tidak tersedia untuk model ini.")
        else:
            top_n = st.slider("Jumlah fitur", min_value=5, max_value=30, value=15, step=5)
            top_fi = fi_df.head(top_n).sort_values("importance", ascending=True)
            st.bar_chart(top_fi.set_index("feature")["importance"])
            st.dataframe(fi_df.head(top_n), use_container_width=True, hide_index=True)
