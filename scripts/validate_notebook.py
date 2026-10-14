"""Static validation for the repository notebook; does not require the dataset."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read notebook: {exc}"]
    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("outputs"):
            errors.append(f"cell {index} contains stale outputs")
        if cell.get("cell_type") == "code":
            try:
                ast.parse("".join(cell.get("source", [])))
            except SyntaxError as exc:
                errors.append(f"cell {index} has syntax error: {exc}")
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for forbidden in ("E:\\programming_languages", "/home/", "C:\\"):
        if forbidden in source:
            errors.append(f"machine-specific path found: {forbidden}")
    for required in ("FileNotFoundError", "train_test_split", "stratify=y", "Pipeline"):
        if required not in source:
            errors.append(f"required contract missing: {required}")
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
