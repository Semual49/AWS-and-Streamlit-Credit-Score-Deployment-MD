"""
inference.py — Inference entry point for the Credit Score classifier.

Usage (CLI):
    python inference.py --input data/data_C.csv --output results/predictions.csv
    python inference.py --input data/single_row.csv

Usage (programmatic):
    from app.inference import CreditScoreInference
    inf = CreditScoreInference()
    result = inf.predict_single({...})
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.pipeline_local import clean_dataframe

DEFAULT_MODEL_PATH = ROOT / "models" / "model_best.pkl"


class CreditScoreInference:
    """Load a saved model artifact and run predictions."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run the local pipeline first:\n"
                "  python -m src.models.pipeline_local --data data/data_C.csv"
            )
        artifact = joblib.load(model_path)
        self.model: Any          = artifact["model"]
        self.feature_columns: list[str] = artifact["feature_columns"]
        self.metadata: dict[str, Any]   = artifact["metadata"]
        self.classes: list[str]         = self.metadata.get("classes", [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align a cleaned DataFrame to the training feature schema."""
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_columns]

    def _predict_df(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        """Return predictions (and probabilities when available) as a DataFrame."""
        X_aligned = self._align(X_clean)
        preds = self.model.predict(X_aligned)
        result = pd.DataFrame({"Predicted_Credit_Score": preds}, index=X_clean.index)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_aligned)
            for i, cls in enumerate(self.model.classes_):
                result[f"Prob_{cls}"] = proba[:, i].round(4)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_csv(self, input_path: str | Path) -> pd.DataFrame:
        """Run inference on a CSV file.  Returns original rows + predictions."""
        raw = pd.read_csv(input_path)
        X_clean = clean_dataframe(raw)
        pred_df = self._predict_df(X_clean)
        return pd.concat([pred_df, raw.reset_index(drop=True)], axis=1)

    def predict_single(self, row: dict[str, Any]) -> dict[str, Any]:
        """Run inference on a single customer dict.

        Args:
            row: Dict mapping feature names to raw (pre-clean) values.

        Returns:
            Dict with keys: prediction, probabilities, classes.
        """
        raw = pd.DataFrame([row])
        X_clean = clean_dataframe(raw)
        pred_df = self._predict_df(X_clean)

        prediction = str(pred_df["Predicted_Credit_Score"].iloc[0])
        proba_cols = [c for c in pred_df.columns if c.startswith("Prob_")]
        probabilities = (
            {c.replace("Prob_", ""): float(pred_df[c].iloc[0]) for c in proba_cols}
            if proba_cols else {}
        )
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "classes": self.classes,
        }

    def model_info(self) -> dict[str, Any]:
        """Return a summary of the deployed model."""
        return {
            "best_model":        self.metadata.get("best_model"),
            "test_accuracy":     self.metadata.get("test_accuracy"),
            "test_f1_weighted":  self.metadata.get("test_f1_weighted"),
            "classes":           self.classes,
            "n_features":        len(self.feature_columns),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained Credit Score classifier"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file (same schema as training data)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write prediction CSV (default: prints sample to stdout)"
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH),
        help="Path to model artifact .pkl (default: models/model_best.pkl)"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inf = CreditScoreInference(model_path=args.model)

    print("=== Model Info ===")
    print(json.dumps(inf.model_info(), indent=2))
    print()

    output_df = inf.predict_csv(args.input)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(out_path, index=False)
        print(f"Predictions saved to: {out_path}")
    else:
        print("=== Sample Predictions (first 5 rows) ===")
        print(output_df.head())

    counts = output_df["Predicted_Credit_Score"].value_counts()
    print("\n=== Prediction Distribution ===")
    print(counts.to_string())


if __name__ == "__main__":
    main()
