"""Leakage-safe educational workflow for the public Kaggle cancer CSV.

This module is not a screening or diagnostic tool. It expects the Kaggle CSV
with a ``diagnosis`` column containing ``B`` and ``M`` labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

LABELS = {"B": 0, "M": 1}
DISPLAY_LABELS = ["Benign (B)", "Malignant (M)"]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the Kaggle CSV; exclude its identifier from features."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the CSV from "
            "https://www.kaggle.com/datasets/erdemtaha/cancer-data and set "
            "CANCER_DATA_PATH to its location; the dataset is not distributed here."
        )

    frame = pd.read_csv(path)
    if "diagnosis" not in frame.columns:
        raise ValueError("CSV must contain a 'diagnosis' target column.")

    labels = frame["diagnosis"].astype("string").str.strip().str.upper()
    unknown = sorted(set(labels.dropna().unique()) - set(LABELS))
    if labels.isna().any() or unknown:
        raise ValueError("'diagnosis' must contain only non-missing B or M labels.")

    excluded = [column for column in frame.columns if column.lower() in {"id", "diagnosis"}]
    features = frame.drop(columns=excluded).copy()
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("CSV must contain at least one non-empty numeric feature column.")

    for column in features.columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Feature column {column!r} contains non-numeric values.")
        features[column] = converted
    if features.isna().all(axis=0).any():
        raise ValueError("Feature columns cannot be entirely missing.")

    target = labels.map(LABELS).astype("int64").rename("diagnosis")
    if target.value_counts().reindex([0, 1], fill_value=0).min() < 2:
        raise ValueError("At least two rows of each diagnosis class are required.")
    return features, target


def _pipeline(estimator: Any) -> Pipeline:
    """Keep imputation/scaling inside estimator fitting (including CV folds)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("model", estimator),
        ]
    )


def evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 44,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame, pd.Series]:
    """Fit four classifiers on a stratified split and report held-out metrics.

    KNN hyperparameters are selected using stratified cross-validation on the
    training partition only. Test metrics are descriptive for this split, not a
    clinical validation or an estimate of real-world screening performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    min_train_class_count = int(y_train.value_counts().min())
    n_splits = min(cv_folds, min_train_class_count)
    if n_splits < 2:
        raise ValueError("At least two training examples per class are required for CV.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models: dict[str, Pipeline] = {
        "Logistic Regression": _pipeline(
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=random_state)
        ),
        "K-Nearest Neighbors": _pipeline(KNeighborsClassifier()),
        "Gaussian Naive Bayes": _pipeline(GaussianNB()),
        "Decision Tree": _pipeline(DecisionTreeClassifier(random_state=random_state)),
    }
    min_cv_training_size = min(
        len(train_indices) for train_indices, _ in cv.split(X_train, y_train)
    )
    neighbor_options = [k for k in (3, 5, 7) if k <= min_cv_training_size] or [1]
    search = GridSearchCV(
        models["K-Nearest Neighbors"],
        param_grid={"model__n_neighbors": neighbor_options},
        scoring="recall",
        cv=cv,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    models["K-Nearest Neighbors"] = search.best_estimator_

    for name, estimator in models.items():
        if name != "K-Nearest Neighbors":
            estimator.fit(X_train, y_train)

    rows = []
    for name, estimator in models.items():
        predicted = estimator.predict(X_test)
        malignant_probability = estimator.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "malignant_recall": recall_score(y_test, predicted, pos_label=1),
                "benign_recall": recall_score(y_test, predicted, pos_label=0),
                "malignant_precision": precision_score(
                    y_test, predicted, pos_label=1, zero_division=0
                ),
                "malignant_f1": f1_score(y_test, predicted, pos_label=1, zero_division=0),
                "roc_auc": roc_auc_score(y_test, malignant_probability),
            }
        )
    return pd.DataFrame(rows), models, X_test, y_test
