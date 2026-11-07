"""
Train 5 classifiers on the UCI Online Shoppers Purchasing Intention dataset,
evaluate them on a held-out test split and save everything the app needs.

Run:  python train_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.utils import (
    BINARY_COLS, CATEGORICAL_COLS, MODEL_FILES, RATIO_COLS, SKEWED_COLS,
    TARGET, compute_metrics, prepare_frame,
)

SEED = 42
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def build_preprocessor() -> ColumnTransformer:
    # log1p tames the long right tail of page counts / durations before scaling
    skewed = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("skewed", skewed, SKEWED_COLS),
        ("ratio", StandardScaler(), RATIO_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])


# Each model + a small hyper-parameter grid (tuned with 5-fold CV on the train split only)
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [10, 30]},
    ),
    "KNN": (
        KNeighborsClassifier(weights="distance"),
        {"clf__n_neighbors": [7, 15, 25]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
    ),
    "Support Vector Machine": (
        # Platt scaling on top of the RBF SVM gives probabilities for the AUC score
        CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
            method="sigmoid", ensemble=False,
        ),
        {"clf__estimator__C": [0.5, 2.0]},
    ),
}


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Stratified 80/20 split keeps the ~15.5% buyer ratio in both parts
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=SEED
    )
    test_df.to_csv(ROOT / "test_data.csv", index=False)  # uploaded in the app
    X_train, y_train = prepare_frame(train_df)
    X_test, y_test = prepare_frame(test_df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, best_params = {}, {}

    for name, (clf, grid) in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_prob)
        best_params[name] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}

        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        print(f"{name:<24} best={best_params[name]}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    table = pd.DataFrame(results).T.round(4)
    table.index.name = "ML Model Name"
    table.to_csv(MODEL_DIR / "metrics.csv")
    with open(MODEL_DIR / "training_info.json", "w") as f:
        json.dump({
            "dataset": "UCI Online Shoppers Purchasing Intention",
            "rows": int(df.shape[0]), "features": int(df.shape[1] - 1),
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "positive_rate": round(float(df[TARGET].mean()), 4),
            "best_params": best_params,
        }, f, indent=2)
    print("\n", table.to_markdown())


if __name__ == "__main__":
    main()
