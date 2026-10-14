from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "Cancer screening code.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_valid_and_has_no_stale_outputs() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))


def test_notebook_has_no_machine_specific_paths() -> None:
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )
    assert "E:\\programming_languages" not in source
    assert "/home/" not in source
    assert "C:\\" not in source


def test_notebook_contains_leakage_safe_training_contract() -> None:
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )
    assert "train_test_split" in source
    assert "stratify=y" in source
    assert "random_state=RANDOM_STATE" in source
    assert "SimpleImputer" in source and "StandardScaler" in source
    assert "Pipeline" in source
    assert "FileNotFoundError" in source
