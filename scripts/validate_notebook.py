"""Static validation for the repository notebook and modeling module."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REQUIRED_MODULE_CONTRACT = (
    "FileNotFoundError",
    "train_test_split",
    "stratify=y",
    "Pipeline",
    "SimpleImputer",
    "StandardScaler",
)


def validate(path: Path) -> list[str]:
    """Return human-readable validation errors without requiring the dataset."""
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read notebook: {exc}"]

    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("notebook must contain a non-empty cells list")
        cells = []

    code_sources: list[str] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            errors.append(f"cell {index} has an invalid cell_type")
        if cell.get("cell_type") == "code":
            if cell.get("outputs", []) != []:
                errors.append(f"cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                errors.append(f"cell {index} has a stale execution count")
            source = "".join(cell.get("source", []))
            code_sources.append(source)
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f"cell {index} has syntax error: {exc}")

    combined_source = "\n".join(code_sources)
    try:
        compile(combined_source, str(path), "exec")
    except SyntaxError as exc:
        errors.append(f"combined notebook source has syntax error: {exc}")
    for forbidden in ("E:\\programming_languages", "/home/", "C:\\"):
        if forbidden in combined_source:
            errors.append(f"machine-specific path found: {forbidden}")
    if "cancer_screening" not in combined_source:
        errors.append("notebook must use the reusable cancer_screening module")

    module_path = path.parent / "cancer_screening.py"
    try:
        module_source = module_path.read_text(encoding="utf-8")
        ast.parse(module_source)
    except (OSError, SyntaxError) as exc:
        errors.append(f"model module is missing or invalid: {exc}")
    else:
        for required in REQUIRED_MODULE_CONTRACT:
            if required not in module_source:
                errors.append(f"required module contract missing: {required}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("Cancer screening code.ipynb"))
    args = parser.parse_args()
    errors = validate(args.path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
