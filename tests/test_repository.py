from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from cancer_screening import load_dataset, resolve_data_path, split_dataset

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "Cancer screening code.ipynb"
MODULE = REPO / "cancer_screening.py"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_source() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_valid_and_has_no_stale_outputs() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            ast.parse("".join(cell.get("source", [])))


def test_notebook_and_module_have_no_machine_specific_paths() -> None:
    source = notebook_source() + "\n" + MODULE.read_text(encoding="utf-8")
    for forbidden in ("E:\\programming_languages", "/home/", "C:\\"):
        assert forbidden not in source


def test_module_contains_leakage_safe_training_contract() -> None:
    source = MODULE.read_text(encoding="utf-8")
    for required in (
        "train_test_split",
        "stratify=y",
        "random_state=",
        "SimpleImputer",
        "StandardScaler",
        "Pipeline",
        "FileNotFoundError",
        "StratifiedKFold",
        "error_score=\"raise\"",
    ):
        assert required in source
    assert "cancer_screening" in notebook_source()


def test_missing_data_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cancer screening data is required"):
        resolve_data_path(tmp_path / "does-not-exist.csv", project_root=tmp_path)


def test_load_dataset_validates_schema_and_preserves_missing_measurements(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": range(10),
            "diagnosis": ["B", "M"] * 5,
            "radius_mean": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "texture_mean": range(10, 20),
            "Unnamed: 32": [None] * 10,
        }
    )
    path = tmp_path / "Cancer_Data.csv"
    frame.to_csv(path, index=False)
    X, y, feature_columns = load_dataset(path)
    assert feature_columns == ["radius_mean", "texture_mean"]
    assert list(X.columns) == feature_columns
    assert X["radius_mean"].isna().sum() == 1
    assert y.tolist() == [0, 1] * 5


def test_load_dataset_rejects_bad_features(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "diagnosis": ["B", "M"],
            "measurement": [float("inf"), 2.0],
        }
    )
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="infinity"):
        load_dataset(path)


def test_split_dataset_rejects_insufficient_training_rows() -> None:
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series([0, 1] * 5)
    with pytest.raises(ValueError, match="5-fold CV"):
        split_dataset(X, y)
