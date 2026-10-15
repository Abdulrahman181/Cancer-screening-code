"""Leakage-safe benchmark utilities for the Wisconsin diagnostic breast cancer data.

The project intentionally does not ship patient data.  These functions validate the
expected CSV contract, create a deterministic stratified split, and evaluate only
estimators that are fitted on the training data supplied by the caller.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 44
TEST_SIZE = 0.20
TARGET_COLUMN = "diagnosis"
ID_COLUMNS = {"id", "unnamed: 32"}
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_data_path(
    explicit_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Find the real input CSV without silently generating or downloading data.

    Relative paths supplied explicitly or through ``CANCER_DATA_PATH`` are resolved
    from the caller's current working directory.  Repository defaults are resolved
    from this module's directory, so opening the notebook outside the repository is
    still predictable.
    """
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    environment_path = os.getenv("CANCER_DATA_PATH")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    root = Path(project_root).expanduser() if project_root is not None else PROJECT_ROOT
    candidates.extend((root / "data" / "Cancer_Data.csv", root / "Cancer_Data.csv"))

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.absolute()
        if normalized not in seen:
            unique_candidates.append(candidate)
            seen.add(normalized)
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        "Cancer screening data is required but was not found. "
        f"Set CANCER_DATA_PATH or place Cancer_Data.csv in data/. Searched: {searched}"
    )


def load_dataset(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load and validate the Wisconsin diagnostic CSV.

    Missing numeric measurements are allowed because the model pipelines impute
    them on training folds.  Invalid labels, duplicate columns, all-missing
    features, non-numeric features, and non-finite values are rejected early.
    """
    data_path = resolve_data_path(path)
    try:
        frame = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset {data_path} is empty") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Dataset {data_path} has duplicate column names: {duplicates}")
    frame = frame.drop(
        columns=[column for column in frame.columns if column.lower() == "unnamed: 32"],
        errors="ignore",
    )
    if frame.empty:
        raise ValueError(f"Dataset {data_path} contains no rows")

    required = {TARGET_COLUMN, "id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset {data_path} is missing required columns: {sorted(missing)}")

    labels = frame[TARGET_COLUMN].astype("string").str.strip().str.upper()
    invalid_mask = ~labels.isin(["B", "M"])
    if invalid_mask.any():
        invalid_labels = sorted(
            {"<NA>" if pd.isna(value) else str(value) for value in labels[invalid_mask]}
        )
        raise ValueError(f"{TARGET_COLUMN!r} must contain only B/M labels; found {invalid_labels}")

    feature_columns = [
        column
        for column in frame.columns
        if column.lower() not in ID_COLUMNS and column != TARGET_COLUMN
    ]
    if not feature_columns:
        raise ValueError("No feature columns remain after removing id and diagnosis")
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature columns must be numeric; non-numeric columns: {non_numeric}")

    X = frame[feature_columns].copy()
    all_missing = [column for column in feature_columns if X[column].isna().all()]
    if all_missing:
        raise ValueError(f"Feature columns contain no usable values: {all_missing}")
    numeric_values = X.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric_values).any():
        raise ValueError("Feature columns must not contain positive or negative infinity")

    y = labels.map({"B": 0, "M": 1}).astype("int8")
    if y.nunique() != 2:
        raise ValueError("Both benign (B) and malignant (M) classes are required")
    return X, y, feature_columns


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the fixed stratified split and fail clearly when CV is impossible."""
    if len(X) == 0 or len(y) == 0 or len(X) != len(y):
        raise ValueError("X and y must be non-empty and have the same number of rows")
    if y.nunique() != 2:
        raise ValueError("Both classes are required for a stratified split")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    if len(test_counts) != 2:
        raise ValueError("The held-out split must contain both classes")
    if train_counts.min() < cv_splits:
        raise ValueError(
            f"Each training class needs at least {cv_splits} rows for {cv_splits}-fold CV; "
            f"got {train_counts.to_dict()}"
        )
    return X_train, X_test, y_train, y_test


def make_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Wrap an estimator so imputation and scaling are fitted inside each fold."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def build_models(*, random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Build deterministic baselines plus a training-only KNN grid search."""
    models: dict[str, Any] = {
        "logistic_regression": make_pipeline(
            LogisticRegression(solver="liblinear", max_iter=2_000, random_state=random_state)
        ),
        "knn": make_pipeline(KNeighborsClassifier(n_neighbors=5)),
        "gaussian_nb": make_pipeline(GaussianNB()),
        "decision_tree": make_pipeline(
            DecisionTreeClassifier(criterion="entropy", random_state=random_state)
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models["knn_grid_search"] = GridSearchCV(
        models["knn"],
        {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
    return models


def evaluate_model(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Fit once on training data and calculate held-out classification metrics."""
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(f"Estimator {name!r} must expose predict_proba for ROC AUC")
    probabilities = np.asarray(estimator.predict_proba(X_test))
    classes = np.asarray(getattr(estimator, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes) or 1 not in classes:
        raise ValueError(f"Estimator {name!r} returned an invalid probability schema")
    positive_probability = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, positive_probability),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }


def evaluate_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate each supplied model and return rows sorted by held-out ROC AUC."""
    results = pd.DataFrame(
        [
            evaluate_model(name, estimator, X_train, y_train, X_test, y_test)
            for name, estimator in models.items()
        ]
    )
    return results.sort_values("roc_auc", ascending=False).reset_index(drop=True)
