"""Shared helpers used by both the training script and the Streamlit app."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
)

TARGET = "Revenue"

# Skewed count / duration columns -> log1p + scaling
SKEWED_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "PageValues",
]
# Already-bounded ratio columns -> scaling only
RATIO_COLS = ["BounceRates", "ExitRates", "SpecialDay"]
# Integer codes / labels that are really categories -> one-hot
CATEGORICAL_COLS = [
    "Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType",
]
BINARY_COLS = ["Weekend"]

FEATURES = SKEWED_COLS + RATIO_COLS + CATEGORICAL_COLS + BINARY_COLS

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def _to_binary(series: pd.Series) -> pd.Series:
    """Turn TRUE/FALSE, True/False, 1/0 or yes/no into 0/1 integers."""
    if series.dtype == bool:
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    )
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Column '{series.name}' has unexpected values: {list(bad)}")
    return mapped.astype(int)


def prepare_frame(df: pd.DataFrame):
    """Validate columns and split a raw dataframe into X (features) and y (target)."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].copy()
    X["Weekend"] = _to_binary(X["Weekend"])
    for col in ["Month", "VisitorType"]:
        X[col] = X[col].astype(str)
    y = _to_binary(df[TARGET])
    return X, y


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """The six metrics required by the assignment."""
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
