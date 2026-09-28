# Credit Score Classification — Model Deployment

**Problem:** Multi-class classification — predict customer Credit Score (Good / Standard / Poor)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Train the Model (Local Pipeline)

```bash
# From project root
py src/models/pipeline_local.py --data data/data_C.csv --model-out models/model_best.pkl --metrics-out models/metrics.json
```

This will:
1. Clean and preprocess `data_C.csv`
2. Run cross-validation on 4 candidate models (Logistic Regression, Random Forest, Extra Trees, Gradient Boosting)
3. Select and optionally tune the best model
4. Save `models/model_best.pkl` and `models/metrics.json`
5. Write a training log to `logs/pipeline_local.log`

---

## Step 2 — Run Inference (CLI)

```bash
# Predict on a CSV file
py app/inference.py --input data/data_C.csv --output results/predictions.csv

# Show model info + first 5 predictions
py app/inference.py --input data/data_C.csv
```

---

## Step 3 — Web Deployment (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

Features:
- **CSV Upload** — batch prediction with downloadable results
- **Manual Entry** — single customer prediction with confidence scores
- **What-If Analysis** — compare prediction when one feature changes by a percentage
- **Feature Importance** — top contributing features (tree-based models only)
- **Sidebar** — model card, per-class metrics, confusion matrix, model comparison table

---

## Preprocessing Summary

| Step | Detail |
|---|---|
| Drop identifiers | `ID`, `Customer_ID`, `Name`, `SSN`, `Unnamed: 0` |
| Clean dirty categories | `Credit_Mix='_'` → Unknown; `Payment_Behaviour='!@9#%8'` → Unknown; `Occupation='_______'` → Unknown |
| Clean dirty numerics | Strip trailing `_` suffix (e.g. `'35_'` → 35); treat `'__value__'` sentinel as NaN |
| Parse Credit_History_Age | `"X Years and Y Months"` → total months (int) |
| Feature engineering | `Debt_to_Income = Outstanding_Debt / (Annual_Income + 1)` |
| Feature engineering | `EMI_to_Salary = Total_EMI_per_month / (Monthly_Inhand_Salary + 1)` |
| Loan type count | Count occurrences of "Loan" in `Type_of_Loan` free text |
| Outlier clipping | [1st, 99th] percentile per numeric column (including engineered features) |
| Imputation | Median for numeric; most-frequent for categorical |
| Encoding | OrdinalEncoder for `Credit_Mix`, `Payment_of_Min_Amount`; OHE for the rest |
| Scaling | StandardScaler on numeric features |
| Feature selection | SelectPercentile (f_classif, top 85%) |

---

## Model Evaluation Metric

**Primary metric: Weighted F1-score**  
Chosen because the dataset is imbalanced (~53% Standard, ~29% Poor, ~18% Good). Accuracy alone would be misleading.

---

## MLflow Tracking

If `mlflow` is installed, each pipeline run logs parameters, metrics, and the model artifact to `mlruns/`.

```bash
mlflow ui
```
