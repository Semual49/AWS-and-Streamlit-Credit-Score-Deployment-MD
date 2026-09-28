"""
Local training pipeline for credit score classification.

Dataset: data_C.csv — 25 000 records, target Credit_Score (Good / Standard / Poor).

This module is written with an object-oriented structure to satisfy the
project requirement for separate preprocessing, training, and evaluation
classes. It also logs experiments to MLflow when MLflow is installed.

Key design decisions:
- Stratified 80/20 train-test split to preserve class proportions.
- Weighted F1 as primary metric due to class imbalance (~53% Standard, ~29% Poor, ~18% Good).
- Outlier clipping at [1st, 99th] percentile before scaling.
- SelectPercentile (f_classif, top 85%) for feature selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COLUMN = "Credit_Score"

CREDIT_MIX_ORDER = [["Bad", "Standard", "Good", "Unknown"]]
PAY_MIN_ORDER = [["No", "NM", "Yes", "Unknown"]]


def setup_logger(log_path: str = "logs/pipeline_local.log") -> logging.Logger:
    """Create a logger that writes to console and file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("credit_score_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class nullcontext:
    """Minimal context manager used when MLflow is not installed."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def try_import_mlflow():
    """Import MLflow only when it is available in the environment."""
    try:
        import mlflow
        import mlflow.sklearn
        return mlflow
    except Exception:
        return None


def to_number(value: Any) -> float:
    """Convert messy numeric values into floats.

    Handles cases seen in data_C:
    - Trailing underscore suffix  : '35_'      -> 35.0
    - Leading+trailing underscores: '__10000__' -> NaN  (sentinel value, not a real amount)
    - Pure numeric strings        : '20364.57'  -> 20364.57
    - Already-float input         : 303.92      -> 303.92
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    # Sentinel pattern: value wrapped in underscores (e.g. '__10000__') is dirty
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


def credit_history_to_months(value: Any) -> float:
    """Convert values such as '2 Years and 3 Months' into total months."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(value))
    if not match:
        return np.nan
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def loan_type_count(value: Any) -> int:
    """Count loan types from the free-text Type_of_Loan field."""
    if pd.isna(value):
        return 0
    return len(re.findall(r"Loan", str(value)))


class CreditDataPreprocessor:
    """Clean raw credit score data and build the sklearn preprocessing step."""

    placeholder_cols = [
        "Occupation",
        "Credit_Mix",
        "Payment_Behaviour",
        "Payment_of_Min_Amount",
        "Month",
    ]
    placeholder_values = {"_______", "_", "!@9#%8", "nan", "NaN", "", "None"}
    dirty_numeric_cols = [
        "Age",
        "Annual_Income",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    outlier_clip_cols = [
        "Age",
        "Annual_Income",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Outstanding_Debt",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
        "Monthly_Inhand_Salary",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        # Engineered features — also need clipping before they reach the scaler
        "Debt_to_Income",
        "EMI_to_Salary",
    ]
    drop_cols = [
        "Unnamed: 0",
        "ID",
        "Customer_ID",
        "Name",
        "SSN",
        "Type_of_Loan",
        "Credit_History_Age",
        TARGET_COLUMN,
    ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data and create model features.

        Processing order (order matters — do not resequence):
          1. Replace placeholder/dirty category values with 'Unknown'
          2. Strip trailing-underscore suffix from dirty numeric strings;
             treat '__value__' sentinels as NaN
          3. Parse Credit_History_Age -> Credit_History_Age_Months (int)
          4. Count loan types from Type_of_Loan free text
          5. Engineer Debt_to_Income and EMI_to_Salary ratios
          6. Clip outliers at [1st, 99th] percentile (includes engineered features)
          7. Drop identifier / raw source columns
        """
        out = df.copy()

        for col in self.placeholder_cols:
            if col in out.columns:
                out[col] = out[col].astype(str).str.strip()
                out[col] = out[col].apply(lambda x: "Unknown" if x in self.placeholder_values else x)

        for col in self.dirty_numeric_cols:
            if col in out.columns:
                out[col] = out[col].apply(to_number)

        if "Credit_History_Age" in out.columns:
            out["Credit_History_Age_Months"] = out["Credit_History_Age"].apply(credit_history_to_months)

        if "Type_of_Loan" in out.columns:
            out["Loan_Type_Count"] = out["Type_of_Loan"].apply(loan_type_count)

        if {"Outstanding_Debt", "Annual_Income"}.issubset(out.columns):
            out["Debt_to_Income"] = out["Outstanding_Debt"] / (out["Annual_Income"] + 1)

        if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(out.columns):
            out["EMI_to_Salary"] = out["Total_EMI_per_month"] / (out["Monthly_Inhand_Salary"] + 1)

        for col in self.outlier_clip_cols:
            if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
                lower, upper = out[col].quantile([0.01, 0.99])
                out[col] = out[col].clip(lower=lower, upper=upper)

        return out.drop(columns=[c for c in self.drop_cols if c in out.columns], errors="ignore")

    def build_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build a preprocessing transformer for numeric and categorical columns."""
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        ordinal_cols = [c for c in ["Credit_Mix", "Payment_of_Min_Amount"] if c in X.columns]
        onehot_cols = [c for c in X.select_dtypes(exclude="number").columns if c not in ordinal_cols]

        transformers = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )

        if ordinal_cols:
            orders = []
            for col in ordinal_cols:
                orders.append(CREDIT_MIX_ORDER[0] if col == "Credit_Mix" else PAY_MIN_ORDER[0])
            transformers.append(
                (
                    "ordinal",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    categories=orders,
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    ordinal_cols,
                )
            )

        if onehot_cols:
            transformers.append(
                (
                    "onehot",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
                        ]
                    ),
                    onehot_cols,
                )
            )

        return ColumnTransformer(transformers=transformers, remainder="drop")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function used by Streamlit and cloud inference."""
    return CreditDataPreprocessor().clean(df)


@dataclass
class ExperimentResult:
    model: str
    cv_f1_weighted_mean: float
    cv_f1_weighted_std: float
    test_accuracy: float
    test_precision_weighted: float
    test_recall_weighted: float
    test_f1_weighted: float


class ModelTrainer:
    """Train candidate models and tune the best candidate."""

    def __init__(self, random_state: int = RANDOM_STATE, logger: logging.Logger | None = None) -> None:
        self.random_state = random_state
        self.logger = logger or logging.getLogger("credit_score_pipeline")

    def candidates(self) -> dict[str, Any]:
        """Return candidate machine learning models."""
        return {
            "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=1),
            "random_forest": RandomForestClassifier(
                n_estimators=80,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=80,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=3,
                subsample=0.8,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=self.random_state,
            ),
        }

    def param_distributions(self) -> dict[str, dict[str, list[Any]]]:
        """Return hyperparameter spaces for tuning."""
        return {
            "random_forest": {
                "model__n_estimators": [160, 220],
                "model__max_depth": [18, 22, None],
                "model__min_samples_leaf": [1, 2, 3],
            },
            "extra_trees": {
                "model__n_estimators": [160, 220],
                "model__max_depth": [18, 22, None],
                "model__min_samples_leaf": [1, 2, 3],
            },
            "gradient_boosting": {
                "model__n_estimators": [60, 100, 150],
                "model__learning_rate": [0.05, 0.08, 0.1],
                "model__max_depth": [3, 4],
                "model__subsample": [0.8, 1.0],
            },
            "logistic_regression": {
                "model__C": [0.3, 1.0, 3.0],
                "model__solver": ["lbfgs"],
            },
        }

    def build_pipeline(self, preprocessor: ColumnTransformer, model: Any) -> Pipeline:
        """Create a full sklearn pipeline."""
        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("feature_selection", SelectPercentile(score_func=f_classif, percentile=85)),
                ("model", model),
            ]
        )

    def compare_models(self, X_train, y_train, X_test, y_test, preprocessor, mlflow=None) -> tuple[list[dict], dict, str]:
        """Run cross-validation and test evaluation for every candidate."""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        experiments: list[dict] = []
        fitted: dict[str, Pipeline] = {}
        best_name = ""
        best_score = -1.0

        # Use up to 8 000 rows for cross-validation so CV scores are representative
        # of the full training set.  1 500 rows (the previous value) is only 7.5 %
        # of the 20 000-row train split and produces unreliable fold scores.
        cv_sample_size = min(8000, len(X_train))
        if cv_sample_size < len(X_train):
            X_cv, _, y_cv, _ = train_test_split(
                X_train,
                y_train,
                train_size=cv_sample_size,
                random_state=self.random_state,
                stratify=y_train,
            )
        else:
            X_cv, y_cv = X_train, y_train
        self.logger.info("Cross-validation sample size: %d", len(X_cv))

        for name, estimator in self.candidates().items():
            self.logger.info("Training candidate: %s", name)
            pipeline = self.build_pipeline(preprocessor, estimator)
            scores = cross_validate(
                pipeline,
                X_cv,
                y_cv,
                cv=cv,
                scoring={"f1_weighted": "f1_weighted"},
                n_jobs=1,
                error_score="raise",
            )
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)

            result = ExperimentResult(
                model=name,
                cv_f1_weighted_mean=round(float(scores["test_f1_weighted"].mean()), 4),
                cv_f1_weighted_std=round(float(scores["test_f1_weighted"].std()), 4),
                test_accuracy=round(float(accuracy_score(y_test, pred)), 4),
                test_precision_weighted=round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4),
                test_recall_weighted=round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4),
                test_f1_weighted=round(float(f1_score(y_test, pred, average="weighted")), 4),
            )
            result_dict = result.__dict__
            experiments.append(result_dict)
            fitted[name] = pipeline
            self.logger.info("%s result: %s", name, result_dict)

            if mlflow is not None:
                with mlflow.start_run(run_name=f"candidate_{name}", nested=True):
                    mlflow.log_param("model", name)
                    mlflow.log_metrics({k: v for k, v in result_dict.items() if isinstance(v, (int, float))})

            if result.test_f1_weighted > best_score:
                best_score = result.test_f1_weighted
                best_name = name

        return experiments, fitted, best_name

    def tune_best(self, best_name: str, preprocessor: ColumnTransformer, X_train, y_train) -> Pipeline:
        """Tune the selected candidate using RandomizedSearchCV."""
        candidates = self.candidates()
        param_grid = self.param_distributions().get(best_name, {})
        pipeline = self.build_pipeline(preprocessor, candidates[best_name])
        if not param_grid:
            return pipeline.fit(X_train, y_train)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(8, sum(len(v) for v in param_grid.values()) + 2),
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)
        self.logger.info("Best tuning parameters: %s", search.best_params_)
        self.logger.info("Best tuning CV F1 weighted: %.4f", search.best_score_)
        return search.best_estimator_


class ModelEvaluator:
    """Evaluate the final model and build deployment metadata."""

    def evaluate(self, model: Pipeline, X_test, y_test, classes: list[str]) -> dict[str, Any]:
        """Return final evaluation metrics."""
        pred = model.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        return {
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_precision_weighted": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4),
            "test_recall_weighted": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4),
            "test_f1_weighted": round(float(f1_score(y_test, pred, average="weighted")), 4),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
            "classification_report": report,
        }

    def feature_importance(self, model: Pipeline) -> list[dict[str, Any]]:
        """Extract feature importance for tree-based final models only."""
        if not hasattr(model, "named_steps"):
            return []
        estimator = model.named_steps.get("model")
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return []
        preprocessor = model.named_steps.get("preprocess")
        selector = model.named_steps.get("feature_selection")
        if preprocessor is None:
            return []
        try:
            names = list(preprocessor.get_feature_names_out())
        except Exception:
            return []
        if selector is not None and hasattr(selector, "get_support"):
            mask = selector.get_support()
            names = [name for name, keep in zip(names, mask) if keep]
        importances = np.asarray(estimator.feature_importances_)
        if len(importances) != len(names):
            return []
        rows = [
            {"feature": name, "importance": round(float(value), 8)}
            for name, value in zip(names, importances)
        ]
        return sorted(rows, key=lambda x: x["importance"], reverse=True)


class CreditScorePipeline:
    """Main orchestration class for local build and retraining."""

    def __init__(self, random_state: int = RANDOM_STATE, log_path: str = "logs/pipeline_local.log") -> None:
        self.random_state = random_state
        self.logger = setup_logger(log_path)
        self.preprocessor_class = CreditDataPreprocessor()
        self.trainer = ModelTrainer(random_state=random_state, logger=self.logger)
        self.evaluator = ModelEvaluator()

    def run(self, data_path: str, model_out: str, metrics_out: str, tune: bool = True) -> dict[str, Any]:
        """Execute the full local training pipeline."""
        self.logger.info("Reading dataset from %s", data_path)
        raw = pd.read_csv(data_path)
        if TARGET_COLUMN not in raw.columns:
            raise ValueError(f"Dataset must contain target column: {TARGET_COLUMN}")

        X = self.preprocessor_class.clean(raw)
        y = raw[TARGET_COLUMN].astype(str)
        classes = sorted(y.unique().tolist())
        self.logger.info("Raw shape: %s", raw.shape)
        self.logger.info("Clean feature shape: %s", X.shape)
        self.logger.info("Class distribution: %s", y.value_counts().to_dict())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        preprocessor = self.preprocessor_class.build_transformer(X_train)
        mlflow = try_import_mlflow()
        if mlflow is not None:
            mlflow.set_tracking_uri("file:mlruns")
            mlflow.set_experiment("credit_score_classification")

        with (mlflow.start_run(run_name="local_training_pipeline") if mlflow is not None else nullcontext()):
            experiments, fitted, best_name = self.trainer.compare_models(
                X_train,
                y_train,
                X_test,
                y_test,
                preprocessor,
                mlflow=mlflow,
            )

            if tune:
                self.logger.info("Tuning best candidate: %s", best_name)
                best_model = self.trainer.tune_best(best_name, preprocessor, X_train, y_train)
            else:
                self.logger.info("Skipping tuning. Using best candidate from comparison: %s", best_name)
                best_model = fitted[best_name]

            final_metrics = self.evaluator.evaluate(best_model, X_test, y_test, classes)
            feature_importance = self.evaluator.feature_importance(best_model)
            metadata = {
                "problem": "Credit score classification",
                "target": TARGET_COLUMN,
                "classes": classes,
                "best_model": best_name,
                "random_state": self.random_state,
                "feature_selection": {
                    "method": "SelectPercentile with f_classif",
                    "percentile": 85,
                },
                "features": {
                    "numeric": X.select_dtypes(include="number").columns.tolist(),
                    "categorical": X.select_dtypes(exclude="number").columns.tolist(),
                },
                "experiments": experiments,
                "feature_importance": feature_importance,
                **final_metrics,
            }

            if mlflow is not None:
                mlflow.log_params({"best_model": best_name, "random_state": self.random_state, "feature_selection_percentile": 85})
                mlflow.log_metrics({k: v for k, v in final_metrics.items() if isinstance(v, (int, float))})
                mlflow.sklearn.log_model(best_model, artifact_path="model")

        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": best_model,
            "metadata": metadata,
            "feature_columns": X.columns.tolist(),
        }
        joblib.dump(artifact, model_out, compress=3)
        Path(metrics_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.logger.info("Model saved to %s", model_out)
        self.logger.info("Metrics saved to %s", metrics_out)
        self.logger.info("Final metrics: %s", {k: metadata[k] for k in ["best_model", "test_accuracy", "test_f1_weighted"]})
        print(json.dumps({k: metadata[k] for k in ["best_model", "test_accuracy", "test_f1_weighted"]}, indent=2))
        return metadata




def str_to_bool(value: Any) -> bool:
    """Parse boolean values coming from CLI or SageMaker hyperparameters."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit score classifier")
    parser.add_argument("--data", default="data/data_C.csv", help="Path to CSV dataset")
    parser.add_argument("--model-out", default="models/model_best.pkl", help="Output path for model pickle")
    parser.add_argument("--metrics-out", default="models/metrics.json", help="Output path for metrics JSON")
    parser.add_argument("--no-tune", nargs="?", const=True, default=False, type=str_to_bool, help="Skip tuning for faster execution")
    parser.add_argument("--log-path", default="logs/pipeline_local.log", help="Path for pipeline log file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = CreditScorePipeline(log_path=args.log_path)
    pipeline.run(
        data_path=args.data,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        tune=not args.no_tune,
    )


if __name__ == "__main__":
    main()
