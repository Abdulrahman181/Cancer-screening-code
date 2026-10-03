# Educational Breast Cancer Classification Example

This repository is a small **educational machine-learning exercise** using the public Kaggle [Cancer Data](https://www.kaggle.com/datasets/erdemtaha/cancer-data) dataset. It is **not a cancer screening, diagnostic, or clinical decision-support tool**. No clinical validation is provided, and results on one public dataset do not establish real-world performance.

## Dataset

The dataset is not included. Obtain the CSV from Kaggle yourself and follow its current license and terms. The expected file has a `diagnosis` column (`B` = benign, `M` = malignant), numeric feature columns, and may have an `id` column. The loader excludes the identifier, drops wholly empty feature columns (such as the dataset's empty placeholder), and imputes partial missing values within each training fold. It rejects unknown labels, non-numeric or infinite features, and malformed input. The evaluation API also checks row alignment, binary labels, and forbids `diagnosis`/`id` columns from being passed as features.

Place the CSV at `data/Cancer_Data.csv`, or set `CANCER_DATA_PATH` to its location. The module resolves the default relative to the repository (not the current working directory). The data directory is ignored by Git to help prevent accidental dataset commits.

## Setup and run

Use Python 3.10–3.12, then install the pinned environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Open `Cancer screening code.ipynb` in Jupyter and run its cells from top to bottom. Alternatively, use the tested Python API:

```python
from src.cancer_screening import load_dataset, evaluate_models

X, y = load_dataset()  # honors CANCER_DATA_PATH; otherwise uses repository data/Cancer_Data.csv
metrics, fitted_models, X_test, y_test = evaluate_models(X, y)
print(metrics)
```

The workflow uses a stratified **training/test** holdout split. Median imputation and min-max scaling are fitted inside each estimator pipeline, so they do not learn from the held-out test partition; KNN hyperparameter selection uses stratified cross-validation (the validation procedure) confined to the training partition. There is no additional independent validation dataset. Class encoding is explicit (`B` = 0, `M` = 1), and metrics include recall for both classes, malignant precision/F1, and ROC AUC. A single split is illustrative, not a robust estimate of performance; in particular, malignant recall is not a clinical sensitivity guarantee. No dataset-specific metric or external/clinical validation result is claimed here.

Fitted pipelines are returned in memory for this educational example; the workflow does not write model files or patient-level predictions. Persisted Python model files can execute code when loaded and are not produced by default. Do not commit downloaded datasets, model artifacts, secrets, or private data.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use small synthetic samples only; they do not validate results on the Kaggle dataset. The GitHub Actions workflow runs these checks on Python 3.10 and 3.12, verifies dependency consistency, compiles the Python modules, and checks that the notebook contains valid code with no saved outputs.

## Limitations and reuse

The dataset must be obtained separately, and its license/terms may differ from the code's. This repository does not declare a code license; no permission to reuse the code is implied. The notebook contains no saved model outputs or patient-level example rows. Do not put private or patient-identifiable data in this public repository.
