# Cancer screening benchmark

A reproducible scikit-learn benchmark for the Wisconsin Diagnostic Breast Cancer dataset. The repository contains a cleaned notebook and static repository checks; it does **not** include patient data.

## Data

The notebook requires the original `Cancer_Data.csv` (commonly distributed with the Wisconsin Diagnostic Breast Cancer dataset). Do not commit the dataset or derived artifacts. Either set an explicit path:

```bash
export CANCER_DATA_PATH=/absolute/path/to/Cancer_Data.csv
```

or place the file at `data/Cancer_Data.csv`. If it is missing, execution stops with a descriptive `FileNotFoundError`; no synthetic or placeholder data is generated.

Expected columns include `id`, `diagnosis` (`B`/`M`), and numeric measurement columns. An optional `Unnamed: 32` export artifact is ignored.

## Environment and usage

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m pytest
ruff check .
jupyter notebook 'Cancer screening code.ipynb'
```

The notebook uses a deterministic stratified 80/20 split (`random_state=44`). Imputation and standardization are inside each model pipeline, so they are fitted only on training data. KNN selection uses stratified cross-validation on the training split; the held-out test set is reserved for final accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and confusion-matrix reporting.

This is an engineering/research benchmark, not a clinical diagnostic tool. See [`ENGINEERING_REVIEW.md`](ENGINEERING_REVIEW.md) for the audit findings and validation limits.
