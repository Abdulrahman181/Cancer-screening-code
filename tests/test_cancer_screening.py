"""Tests for data validation and leakage-safe training behavior."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cancer_screening import evaluate_models, load_dataset


class DatasetLoadingTests(unittest.TestCase):
    def _load_frame(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            frame.to_csv(path, index=False)
            return load_dataset(path)

    def test_maps_labels_and_excludes_identifier_and_empty_column(self):
        features, target = self._load_frame(
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "diagnosis": ["B", "M", "B", "M"],
                    "radius": [1.0, 2.0, np.nan, 4.0],
                    "empty": [np.nan] * 4,
                }
            )
        )
        self.assertEqual(list(features.columns), ["radius"])
        self.assertEqual(target.tolist(), [0, 1, 0, 1])

    def test_rejects_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "only non-missing B or M"):
            self._load_frame(pd.DataFrame({"diagnosis": ["B", "X"], "feature": [1, 2]}))

    def test_rejects_non_numeric_features(self):
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            self._load_frame(
                pd.DataFrame(
                    {"diagnosis": ["B", "M", "B", "M"], "feature": [1, "bad", 3, 4]}
                )
            )

    def test_rejects_infinite_feature_values(self):
        with self.assertRaisesRegex(ValueError, "infinite"):
            self._load_frame(
                pd.DataFrame(
                    {"diagnosis": ["B", "M", "B", "M"], "feature": [1, 2, np.inf, 4]}
                )
            )

    def test_rejects_empty_or_unparseable_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parse dataset CSV"):
                load_dataset(path)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def _sample_data():
        rng = np.random.default_rng(21)
        labels = pd.Series(np.array([0, 1] * 30), name="diagnosis")
        features = pd.DataFrame(
            {"feature_a": rng.normal(size=len(labels)), "feature_b": rng.normal(size=len(labels))}
        )
        return features, labels

    def test_evaluates_models_without_fitting_preprocessor_on_test_rows(self):
        rng = np.random.default_rng(21)
        labels = np.array([0, 1] * 60)
        features = pd.DataFrame(
            {
                "feature_a": rng.normal(size=len(labels)),
                "feature_b": rng.normal(size=len(labels)),
            }
        )
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=44, stratify=labels
        )
        features.loc[test_indices, "feature_a"] = 1_000_000

        results, models, X_test, y_test = evaluate_models(
            features, pd.Series(labels), random_state=44
        )
        self.assertEqual(len(results), 4)
        self.assertEqual(len(X_test), len(test_indices))
        self.assertEqual(len(y_test), len(test_indices))
        self.assertTrue({"malignant_recall", "benign_recall", "roc_auc"}.issubset(results.columns))
        self.assertEqual(set(models), set(results["model"]))
        scaler = models["Logistic Regression"].named_steps["scaler"]
        self.assertLess(scaler.data_max_[0], 1_000_000)
        self.assertLess(scaler.data_max_[0], features["feature_a"].max())

    def test_rejects_target_and_identifier_features(self):
        features, target = self._sample_data()
        for column in ("diagnosis", " ID "):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "must not"):
                evaluate_models(features.assign(**{column.strip(): 1}), target)

    def test_rejects_misaligned_or_invalid_targets(self):
        features, target = self._sample_data()
        with self.assertRaisesRegex(ValueError, "identical row indices"):
            evaluate_models(features, target.set_axis(np.arange(100, 160)))
        invalid = target.copy()
        invalid.iloc[0] = 2
        with self.assertRaisesRegex(ValueError, "binary labels"):
            evaluate_models(features, invalid)

    def test_rejects_nonfinite_direct_inputs_and_invalid_configuration(self):
        features, target = self._sample_data()
        features.loc[0, "feature_a"] = -np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            evaluate_models(features, target)
        features.loc[0, "feature_a"] = 0
        with self.assertRaisesRegex(ValueError, "test_size"):
            evaluate_models(features, target, test_size=1.0)
        with self.assertRaisesRegex(ValueError, "cv_folds"):
            evaluate_models(features, target, cv_folds=1)


class NotebookSafetyTests(unittest.TestCase):
    def test_notebook_is_valid_and_contains_no_saved_outputs(self):
        notebook_path = Path(__file__).resolve().parents[1] / "Cancer screening code.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
