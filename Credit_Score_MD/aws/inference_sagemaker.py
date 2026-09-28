from __future__ import annotations

import json
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_COLUMN = "Credit_Score"

PLACEHOLDER_COLS = [
    "Occupation", "Credit_Mix", "Payment_Behaviour",
    "Payment_of_Min_Amount", "Month",
]
PLACEHOLDER_VALUES = {"_______", "_", "!@9#%8", "nan", "NaN", "", "None"}

DIRTY_NUMERIC_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt",
    "Amount_invested_monthly", "Monthly_Balance",
]

OUTLIER_CLIP_COLS = [
    "Age", "Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Outstanding_Debt",
    "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
    "Monthly_Inhand_Salary", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
    "Debt_to_Income", "EMI_to_Salary",
]

DROP_COLS = [
    "Unnamed: 0", "ID", "Customer_ID", "Name", "SSN",
    "Type_of_Loan", "Credit_History_Age", TARGET_COLUMN,
]

# Batas clip yang dipakai saat inference single-row
INFERENCE_CLIP_RANGES = {
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
    "Debt_to_Income": (0.0, 100.0),
    "EMI_to_Salary": (0.0, 100.0),
}


# Helper functions

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


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versi self-contained dari CreditDataPreprocessor.clean().
    Logika identik dengan pipeline_local.py.
    Satu-satunya perbedaan: outlier clip pakai INFERENCE_CLIP_RANGES
    karena tidak bisa hitung percentile dari 1 baris data.
    """
    out = df.copy()

    # 1. Bersihkan placeholder di kolom kategorik
    for col in PLACEHOLDER_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].apply(
                lambda x: "Unknown" if x in PLACEHOLDER_VALUES else x
            )

    # 2. Bersihkan nilai numerik yang kotor
    for col in DIRTY_NUMERIC_COLS:
        if col in out.columns:
            out[col] = out[col].apply(_to_number)

    # 3. Konversi Credit_History_Age ke total bulan
    if "Credit_History_Age" in out.columns:
        out["Credit_History_Age_Months"] = out["Credit_History_Age"].apply(
            _credit_history_to_months
        )

    # 4. Hitung jumlah tipe pinjaman dari teks bebas
    if "Type_of_Loan" in out.columns:
        out["Loan_Type_Count"] = out["Type_of_Loan"].apply(_loan_type_count)

    # 5. Feature engineering: rasio utang dan EMI
    if {"Outstanding_Debt", "Annual_Income"}.issubset(out.columns):
        out["Debt_to_Income"] = out["Outstanding_Debt"] / (out["Annual_Income"] + 1)

    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(out.columns):
        out["EMI_to_Salary"] = out["Total_EMI_per_month"] / (
            out["Monthly_Inhand_Salary"] + 1
        )

    # 6. Clip outlier pakai batas statis
    for col, (lower, upper) in INFERENCE_CLIP_RANGES.items():
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].clip(lower=lower, upper=upper)

    # 7. Hapus kolom identifier dan target
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")

    return out



# SageMaker handler functions


def model_fn(model_dir: str):
    """
    SageMaker memanggil ini sekali saat endpoint pertama kali start.
    Harus return artifact yang akan diteruskan ke predict_fn.
    """
    model_path = os.path.join(model_dir, "model_best.pkl")
    artifact = joblib.load(model_path)
    print(f"[ENDPOINT LOG] Model loaded: {artifact['metadata'].get('best_model')}")
    print(f"[ENDPOINT LOG] Feature count: {len(artifact['feature_columns'])}")
    return artifact


def input_fn(request_body: str, content_type: str = "application/json"):
    """
    SageMaker memanggil ini untuk parse request yang masuk.
    Menerima JSON berisi satu dict (single row) atau list of dicts (batch).
    """
    if content_type != "application/json":
        raise ValueError(
            f"Content type tidak didukung: {content_type}. Gunakan application/json."
        )

    data = json.loads(request_body)

    if isinstance(data, dict):
        return pd.DataFrame([data])
    elif isinstance(data, list):
        return pd.DataFrame(data)
    else:
        raise ValueError("Input harus berupa JSON object atau JSON array of objects.")


def predict_fn(input_data: pd.DataFrame, artifact: dict):
    """
    SageMaker memanggil ini untuk menjalankan prediksi.
    input_data adalah output dari input_fn.
    artifact adalah output dari model_fn.
    """
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # Bersihkan dan engineer features
    X_clean = _clean_dataframe(input_data)

    # Align ke schema training
    for col in feature_columns:
        if col not in X_clean.columns:
            X_clean[col] = np.nan
    X_aligned = X_clean[feature_columns]

    predictions = model.predict(X_aligned).tolist()

    result = {"predictions": predictions}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_aligned)
        result["probabilities"] = proba.tolist()
        result["labels"] = list(model.classes_)

    print(f"[ENDPOINT LOG] Predicted {len(predictions)} row(s): {predictions}")
    return result


def output_fn(prediction: dict, accept: str = "application/json"):
    """
    SageMaker memanggil ini untuk format response yang keluar ke client.
    """
    return json.dumps(prediction), "application/json"
