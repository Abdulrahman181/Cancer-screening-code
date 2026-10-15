# Engineering review — Cancer-screening-code

## Scope

Audited the complete repository at commit `a2739d5`. The baseline contained one file: `Cancer screening code.ipynb` (95 cells, 52 code cells). No Python modules, README, dependency manifest, tests, or data were tracked.

The repository does not contain the dataset and no data or metrics were invented during this review.

## Baseline findings

### Critical / high impact

1. **Non-portable data path:** the notebook read `E:\programming_languages\Cancer_Data.csv`, so it could not run outside the original author's machine.
2. **Preprocessing leakage:** `MinMaxScaler.fit_transform` and mean imputation were fitted on the full dataset before the train/test split. Test information could therefore influence training transforms and reported metrics.
3. **Unstable/under-specified split:** `train_test_split` was not stratified. The split seed was present, but the class balance was not protected and model/CV randomness was not consistently specified.
4. **Fragile positional schema:** features and target were selected with `iloc` positions; there was no schema validation, no explicit B/M label validation, and `dropna(axis=1)` silently removed any column containing a missing value.
5. **Notebook state bugs:** variables were reused across models; the Naive Bayes zero-one-loss cell referenced `y_pred` from a prior model rather than its own predictions. This makes results dependent on execution order and potentially wrong without raising an error.
6. **Stale evidence:** the baseline notebook retained 42 outputs and 51 execution counts, including historical accuracy values and rendered figures. Those values were not reproducible from the repository because the data was absent.
7. **Insufficient evaluation:** the baseline primarily reported accuracy and confusion matrices, with no balanced accuracy, precision, recall, F1, ROC AUC, or clearly reserved final test protocol. Grid search also used default CV behavior rather than an explicit shuffled stratified splitter.

### Medium / maintainability

- There was no documented setup or data contract.
- No dependency versions or development checks were declared.
- Generated PNGs were written to the working directory with no artifact policy.
- The baseline notebook had repeated imports, inconsistent names, and an invalid escape-sequence warning from the Windows path.

## Changes made

- Replaced the old notebook with a clean, eight-cell, output-free notebook.
- Added explicit `CANCER_DATA_PATH`/`data/Cancer_Data.csv` resolution and a descriptive `FileNotFoundError` when data is absent. No fallback or synthetic data exists.
- Added dataset schema checks for `id`, `diagnosis`, numeric features, B/M labels, and both classes; safely ignores only the common `Unnamed: 32` export artifact.
- Added deterministic stratified 80/20 splitting (`random_state=44`).
- Moved median imputation and standardization inside each model `Pipeline`, fitting transforms only on training folds.
- Added deterministic Logistic Regression, KNN, Gaussian Naive Bayes, and Decision Tree pipelines. KNN grid search uses an explicit shuffled `StratifiedKFold` and ROC AUC scoring on training data only.
- Added held-out metrics: accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and confusion matrix. Metrics are generated only when the real CSV is supplied.
- Added `README.md`, `requirements.txt`, `.gitignore`, `pyproject.toml`, `data/.gitkeep`, `scripts/validate_notebook.py`, and `tests/test_repository.py`.
- Removed all historical notebook outputs and execution counts.

## Validation

Completed successfully:

- `python3 scripts/validate_notebook.py` — passed.
- `pytest -q` — **3 passed**.
- `ruff check .` — passed with no errors.
- AST parsing of every notebook code cell and the combined code — passed.
- Notebook JSON validation — passed; 8 cells, 5 code cells, 0 outputs, all execution counts `None`.
- Explicit missing-data guard test — passed; it stops with: `Cancer screening data is required but was not found...`.

A full model run was **not claimed**: the repository intentionally has no `Cancer_Data.csv`, and `nbformat`/`nbconvert` are not installed in this sandbox. Runtime dependencies available in the sandbox were sufficient for import-level checks, but without the real dataset there are no valid metrics to report.

## Reproduction

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export CANCER_DATA_PATH=/absolute/path/to/Cancer_Data.csv
python -m pytest
ruff check .
jupyter notebook 'Cancer screening code.ipynb'
```

This is an engineering/research benchmark, not a clinical diagnostic device. Do not commit patient data, generated artifacts, or unverified metrics.

## Second Senior Review

This second review covered every tracked file, the full notebook source and metadata, prior history (`git log`/`git diff`), repository checks, and runtime dependency availability.

### Findings

- The notebook's modeling logic was not reusable or directly runtime-tested; only static notebook checks existed.
- Dataset validation accepted some dangerous inputs: empty files/frames, all-missing features, and positive/negative infinity. Small datasets could reach an opaque five-fold-CV failure after the split.
- ROC AUC probability selection relied on column position rather than the estimator's class schema, and grid-search failures could be silently converted to `NaN` via the default `error_score`.
- Repository-default data paths were relative to the notebook process's current directory rather than the repository module, and notebook validation did not reject stale execution counts or validate the modeling module.

### Fixes made

- Added `cancer_screening.py` as the tested source of truth for path resolution, schema validation, stratified splitting, leakage-safe pipelines, model construction, and held-out evaluation.
- Added explicit checks for empty data, duplicate columns, invalid B/M labels, all-missing features, non-finite measurements, both-class requirements, held-out class presence, and adequate training rows for five-fold CV.
- Made ROC AUC use the estimator's `classes_` schema and changed KNN grid search to `error_score="raise"`; retained deterministic seeds and shuffled stratified CV.
- Rebuilt the notebook as output-free, execution-count-free orchestration over the module. Extended the validator and tests to cover notebook state, module contracts, missing-data behavior, schema checks, and small-data failures.
- Updated README usage/limitations and added explicit pytest discovery configuration. No data, metrics, generated artifacts, or secrets were added.

### Validation and remaining limits

The follow-up checks are recorded in the final review result. The repository still intentionally lacks `Cancer_Data.csv`, so no end-to-end fit or clinical/research metric result is claimed. Notebook execution also depends on installing the declared Jupyter/runtime dependencies; static checks and module/unit checks do not substitute for a real-data run.
