"""Tests for data validation and leakage-safe training behavior."""
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


class EvaluationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
