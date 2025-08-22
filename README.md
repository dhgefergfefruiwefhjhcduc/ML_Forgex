# mlforgex
[![PyPI Downloads](https://static.pepy.tech/badge/mlforgex)](https://pepy.tech/projects/mlforgex)  [![PyPI Version](https://img.shields.io/pypi/v/mlforgex.svg)](https://pypi.org/project/mlforgex/)  [![License](https://img.shields.io/pypi/l/mlforgex.svg)](https://github.com/yourusername/mlforgex/blob/main/LICENSE)

**mlforgex** is an **end-to-end machine learning automation package** for Python.
It allows you to **train, evaluate, and make predictions** with minimal effort — handling **data preprocessing**, **model selection**, **hyperparameter tuning**, and **artifact generation** automatically.
It supports **both classification and regression** problems and ships with sensible defaults to get you started quickly while providing advanced options for production workflows.

---
## Table of contents
- Key features
- Installation
- Requirements
- Quickstart (train → predict)
  - CLI quickstart
  - Python API quickstart
- Detailed features & explanations
- CLI reference (flags explained)
- Artifacts & outputs (what is saved)
- How it works (high-level pipeline)
- Advanced options & integrations
- Examples
- Testing
- License & author
---

# Key features
- **Automatic data preprocessing**: missing value handling, outlier & duplicate removal, encoding, scaling, and multicollinearity handling.
- **Automatic problem detection**: classification vs regression; binary vs multiclass detection.
- **Imbalanced data handling**: SMOTE (oversampling), under-sampling, auto detection and application.
- **Model training & evaluation**: trains a candidate model pool and selects the best model using task-appropriate metrics and cross-validation.

- **Artifact saving**: trained model, preprocessing pipeline, encoder, metrics, plots, and feature importances are saved to disk.
- **Visualizations**: correlation heatmap, confusion matrix, ROC, learning/residual curves, feature importance.
- **Progress bars & parallel training**: uses `tqdm` for progress and `n_jobs` for parallelism.

---

# Installation

Install the package from PyPI:

```bash
pip install mlforgex
```

---

# Requirements
Minimum tested environment:

- Python >= 3.8
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- imbalanced-learn
- tqdm
- scipy
- requests

See the full list in `requirements.txt`.

---

# Quickstart (train → predict)

You can train using the **CLI** or the **Python API**. The library auto-detects task type (classification vs regression) from the target column and runs an appropriate pipeline.

## CLI quickstart

```bash
# Train (example)
mlforge-train \
  --data_path path/to/data.csv \
  --dependent_feature TargetColumn \
  --rmse_prob 0.3 \
  --f1_prob 0.7 \
  --n_jobs -1 \
  --n_iter 100 \
  --cv 3 \
  --artifacts_dir artifacts
# add --fast to speed up the run
```

After training, run prediction on new rows:

```bash
mlforge-predict \
  --model_path artifacts/model.pkl \
  --preprocessor_path artifacts/preprocessor.pkl \
  --input_data path/to/new_data.csv \
  --encoder_path artifacts/encoder.pkl  # only for classification
# add --no-predicted_data to disable saving predicted data 
```

## Python API quickstart

```python
from mlforgex import train_model, predict

train_model(
    data_path="data.csv",
    dependent_feature="TargetColumn",
    rmse_prob=0.3,   # weight used to rank regression models
    f1_prob=0.7,     # weight used to rank classification models
    n_jobs=-1,
    n_iter=100,
    cv=3,
    artifacts_dir="artifacts",
    fast=False       # set True to skip tuning and go faster
)

preds = predict(
    model_path="artifacts/model.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    input_data_path="new_data.csv",
    encoder_path="artifacts/encoder.pkl"  # optional
)
print(preds[:10])
```

---

# Detailed features & explanations

This section explains each major feature and what it does, so users understand what to expect and how to customize behavior.

## Automatic Data Preprocessing
- **Missing value handling**: numeric columns get imputed with mean or median (auto chosen); categorical columns use mode or a constant label depending on frequency and cardinality. 
- **Outlier removal**: optional z-score or IQR-based outlier removal; configurable via API/CLI. Defaults are conservative to avoid dropping useful data.
- **Duplicate removal**: exact duplicate rows are removed before training.
- **Encoding**: low-cardinality categoricals → One-Hot Encoding; high-cardinality → Ordinal/Target encoding (configurable). Encoders are saved to `encoder.pkl` for reproducible inference.
- **Scaling**: StandardScaler by default for many models.
- **Feature dropping & multicollinearity**: constant/near-constant features dropped; highly collinear features identified (via VIF) and handled to reduce redundancy.

## Automatic Problem Detection
- Inspects `dependent_feature` values to decide:
  - **Regression** if target dtype is numeric and has many unique values.
  - **Classification** if target is categorical / few unique values.
- For classification, detects **binary** vs **multiclass** and adjusts metric selection accordingly.

## Imbalanced Data Handling
- Performs imbalance check (class distribution threshold configurable).
- If imbalance is detected, the pipeline can apply:
  - **SMOTE** (Synthetic Minority Oversampling Technique)
  - **Random under-sampling** (or combinations like SMOTE + Tomek links)
- Resampling is applied only to the training fold inside cross-validation to avoid data leakage.

## Model Training & Evaluation
- Trains a set of candidate models appropriate for the task (linear models, tree ensembles, boosting machines, etc.).
- Uses cross-validation to estimate per-model performance.
- Selects the best model using a composite scoring policy:
  - For classification: F1 / ROC-AUC prioritized (configurable via `--f1_prob` weight).
  - For regression: RMSE / R² prioritized (configurable via `--rmse_prob` weight).

## Hyperparameter Tuning
- Tuning via `RandomizedSearchCV`.
- Controlled via `--n_iter` and `--cv` for `RandomizedSearchCV`, and `--n_jobs` for parallelism.
- **Fast mode (`--fast`)** bypasses tuning and uses robust default hyperparameters for each model—this drastically reduces runtime at the cost of potentially suboptimal model hyperparameters. Use `--fast` for quick iteration or when compute is limited.

## Artifact Saving & Reproducibility
- Saves these artifacts to `artifacts_dir`:
  - `model.pkl` — best performing, serialized model
  - `preprocessor.pkl` — fitted preprocessing pipeline (encoders, scalers)
  - `encoder.pkl` — label/target encoder (classification only)
  - `metrics.txt` — train/test metrics
  - `Plots/` — saved PNGs of the generated visualizations

## Visualizations & Reporting
- Automatically generates and saves:
  - Correlation heatmap (features)
  - Confusion matrix 
  - ROC curve 
  - Precision-Recall curve 
  - Learning curve (train vs validation)
  - Feature importance bar chart
  - Residual plots 


---

# CLI reference (flags explained)

### Train command
```bash
mlforge-train \
  --data_path <path> \
  --dependent_feature <column> \
  --rmse_prob <float> \
  --f1_prob <float> \
  [--n_jobs <int>] \
  [--n_iter <int>] \
  [--cv <int>] \
  [--artifacts_dir <path>] \
  [--artifacts_name <name>] \
  [--fast]
```

| Flag | Type | Default | Explanation |
|---|---:|---:|---|
| `--data_path` | str | — | CSV file path to the dataset. Must include header row and the target column. |
| `--dependent_feature` | str | — | Name of the target column to predict. |
| `--rmse_prob` | float | 0.3 | Ranking weight for regression models (higher means RMSE is prioritized). |
| `--f1_prob` | float | 0.7 | Ranking weight for classification models (higher means F1 is prioritized). |
| `--n_jobs` | int | -1 | Number of CPU cores used for parallelism (`-1` uses all available cores). |
| `--n_iter` | int | 100 | Number of parameter settings sampled when `RandomizedSearchCV` is used. |
| `--cv` | int | 3 | Number of cross-validation folds. |
| `--artifacts_dir` | str | None | Directory where artifacts, metrics, and plots will be saved. |
| `--artifacts_name` | str | artifacts | Name of the artifacts directory. |
| `--fast` | flag | False | **Enable fast mode**. This is a boolean flag — include it to enable. When enabled: skips hyperparameter tuning and uses strong defaults for models to produce results much faster. Example usage: `--fast`. |

**Important notes**:
- `--fast` is a flag; do not pass `True`/`False` as value. Use `--fast` to enable fast mode, omit it to run in full mode.
- `rmse_prob` and `f1_prob` act as relative weights. Only the appropriate one is used for the detected task type (the other is ignored).

### Predict command
```bash
mlforge-predict \
  --model_path <model.pkl> \
  --preprocessor_path <preprocessor.pkl> \
  --input_data <input.csv> \
  --encoder_path <encoder.pkl> 
```

| Flag | Type | Default | Explanation |
|---|---:|---:|---|
| `--model_path` | str | — | Path to the trained model pickle. |
| `--preprocessor_path` | str | — | Path to the preprocessing pipeline pickle. |
| `--input_data` | str | — | CSV file with rows to predict (same feature columns except target). |
| `--encoder_path` | str | — | Path to the encoder pickle (classification only). If not provided for classification, predictions will be returned as encoded values. |
| `--predicted_data` | flag | True | Saves the input data with prediction column. |

---

**Important notes**:
- `--predicted_data` is a flag; do not pass `True`/`False` as value. Use `--no-predicted_data` to disable saving predicted data.



# Artifacts & outputs (what is saved)

After a training run, the `artifacts_dir` contains:

```
artifacts/
├─ model.pkl                 # Serialized best model
├─ preprocessor.pkl          # Fitted preprocessing pipeline
├─ encoder.pkl               # Label encoder (classification)
├─ metrics.txt             # Text file with train/test metrics & CV results
└─ Plots/
   ├─ correlation_heatmap.png
   ├─ confusion_matrix.png
   ├─ roc_curve.png
   ├─ precision_recall.png
   ├─ learning_curve.png
   ├─ feature_importance.png
   └─ residuals.png
```

The `metrics.txt` contains entries such as:

```
Message: Training completed successfully
Problem type: Regression
Model: RandomForestRegressor
Output feature: ...
Categorical features: [...]
Numerical features: [...]
Train R2: ...
Train RMSE: ...
Test R2: ...
Test RMSE: ...
Hyper tuned: False
Dropped Columns: [....]



Arguments used :- 
data_path: ...
dependent_feature: ...
rmse_prob: 0.5
f1_prob: 0.5
n_jobs: -1
n_iter: 100
n_splits: 5
fast: False
artifacts_dir: None
artifacts_name: ...
corr_threshold: 0.85
skew_threshold: 1
z_threshold: 3
overfit_threshold: 0.15

```

---

# How it works (high-level pipeline)

1. **Load & validate data**: Reads CSV, checks for target column, basic schema validation.
2. **Problem detection**: Infers whether we have regression or classification.
3. **Preprocessing**: Missing value imputation, encoding, scaling, duplicate/outlier removal.
4. **Imbalance handling**: If classification and imbalance detected, apply resampling on training folds.
5. **Candidate model training**: Train a curated set of models appropriate for the detected task.
6. **(Optional) tuning**: Use randomized/grid search to tune hyperparameters (skipped in `--fast`). Tuning runs inside CV to avoid leak.
7. **Model selection**: Rank models by composite score derived from `f1_prob`/`rmse_prob` and pick the best.
8. **Save artifacts & report**: Store model, pipeline, metrics, plots, and run config for reproducibility.



# Examples

## Minimal CLI example (regression)
```bash
mlforge-train --data_path housing.csv --dependent_feature SalePrice --cv 5 --n_iter 50 --artifacts_dir housing_artifacts
```

## Predicting from Python
```python
from mlforgex import predict
preds = predict("artifacts/model.pkl", "artifacts/preprocessor.pkl", "new_rows.csv", encoder_path=None)
print(preds.head())
```

---


# Testing

Run tests with:

```bash
pytest test/
```

Include unit tests that check:
- Preprocessing pipeline idempotence
- Correct problem detection behavior
- Model training produces expected keys in `metrics.txt`
- Predict pipeline loads and transforms inputs without error

---

# License & author

This project is licensed under the [MIT License](LICENSE).

**Author**: Priyanshu Mathur  
📧 mathurpriyanshu2006@gmail.com  
Portfolio: https://my-portfolio-phi-two-53.vercel.app/  
PyPI: https://pypi.org/project/mlforgex/

---


