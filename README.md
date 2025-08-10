<!-- # mlforgex [![PyPI Downloads](https://static.pepy.tech/badge/mlforgex)](https://pepy.tech/projects/mlforgex)

**mlforgex** is a Python package that enables easy training, evaluation, and prediction for machine learning models on cleaned dataset. It supports both classification and regression problems, automates preprocessing, model selection, hyperparameter tuning, and generates useful artifacts and plots for analysis.

## Features

- Automatic data preprocessing (missing value handling, encoding, scaling)
- Imbalance handling (under-sampling, over-sampling)
- Model selection and evaluation (classification & regression)
- Hyperparameter tuning with RandomizedSearchCV
- Artifact saving (model, preprocessor, encoder)
- Visualization of metrics and learning curves
- Simple CLI for training and prediction

## Installation

Install mlforge using pip:

```sh
pip install mlforgex
```
stall .
```

## Requirements

- Python >= 3.8
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- xgboost
- imbalanced-learn

See [requirements.txt](requirements.txt) for details.

## Usage

### Train a Model

You can train a model using the CLI:

```sh
mlforge-train --data_path path/to/your/data.csv --dependent_feature TargetColumn --rmse_prob 0.3 --f1_prob 0.7 --n_jobs -1 --n_iter 100 --cv 3
```

Or programmatically:

```python
from mlforge import train_model

train_model(
    data_path=<data_path>,
    dependent_feature=<dependent_feature>,
    rmse_prob=<rmse_probability>,
    f1_prob=<f1_probability>,
    n_jobs=<n_jobs>
    n_iter=<n_iter>,
    n_splits=<n_splits>,
    artifacts_dir=<artifacts_folder_path>,
    fast=<train_fast>
)
```

### Predict

Use the CLI:

```sh
mlforge-predict --model_path path/to/model.pkl --preprocessor_path path/to/preprocessor.pkl --input_data path/to/input.csv --encoder_path path/to/encoder.pkl
```

Or programmatically:

```python
from mlforge import predict

result = predict(
    <model.pkl>,
    <preprocessor.pkl>,
    <input_data.csv>,
    <encoder.pkl>
)
print(result)
```

## Artifacts

After training, the following files are saved :

- `model.pkl`: Trained model
- `preprocessor.pkl`: Preprocessing pipeline
- `encoder.pkl`: Label encoder (for classification)
- `Plots/`: Visualizations (correlation heatmap, confusion matrix, ROC curve, etc.)

## Testing

Run tests using pytest:

```sh
pytest test/
```
## Author

Priyanshu Mathur  
[Portfolio](https://my-portfolio-phi-two-53.vercel.app/)  
Email: mathurpriyanshu2006@gmail.com

## Project Links

- [PyPI](https://pypi.org/project/mlforgex/) -->

# mlforgex  
[![PyPI Downloads](https://static.pepy.tech/badge/mlforgex)](https://pepy.tech/projects/mlforgex)  
[![PyPI Version](https://img.shields.io/pypi/v/mlforgex.svg)](https://pypi.org/project/mlforgex/)  
[![License](https://img.shields.io/pypi/l/mlforgex.svg)](https://github.com/yourusername/mlforgex/blob/main/LICENSE)  

**mlforgex** is an **end-to-end machine learning automation package** for Python.  
It allows you to **train, evaluate, and make predictions** with minimal effort — handling **data preprocessing**, **model selection**, **hyperparameter tuning**, and **artifact generation** automatically.  
It supports **both classification and regression** problems.

---

## 🚀 Features

- **Automatic Data Preprocessing**
  - Handles missing values
  - Encodes categorical variables
  - Scales numeric features
- **Automatic Problem Detection**
  - Detects whether task is **classification** or **regression**
- **Imbalanced Data Handling**
  - Over-sampling (SMOTE)
  - Under-sampling
- **Model Training & Evaluation**
  - Multiple algorithms tested
  - Best model selected automatically
- **Hyperparameter Tuning**
  - Optional tuning via `RandomizedSearchCV`
- **Artifact Saving**
  - Trained model (`model.pkl`)
  - Preprocessing pipeline (`preprocessor.pkl`)
  - Encoder (for classification)
- **Visualizations**
  - Correlation heatmap
  - Confusion matrix
  - ROC curves
  - Learning curves
- **Command Line Interface (CLI)**
  - Train and predict directly from the terminal

---

## 📦 Installation

Install via pip:

```bash
pip install mlforgex
```

---

## 📋 Requirements

- Python >= 3.8  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- xgboost  
- imbalanced-learn 
- tqdm 

Full list in [requirements.txt](requirements.txt).

---

## 📖 Usage

### 1️⃣ Train a Model

You can **train a model** using either **CLI** or **Python code**.

#### **CLI Usage**

```bash
mlforge-train \
    --data_path path/to/data.csv \
    --dependent_feature TargetColumn \
    --rmse_prob 0.3 \
    --f1_prob 0.7 \
    --n_jobs -1 \
    --n_iter 100 \
    --cv 3
```

#### **Python API Usage**

```python
from mlforge import train_model

train_model(
    data_path="data.csv",
    dependent_feature="TargetColumn",
    rmse_prob=0.3,
    f1_prob=0.7,
    n_jobs=-1,
    n_iter=100,
    cv=3,
    artifacts_dir="artifacts",
    fast=False
)

```

**Example Output:**
```python
Message: Training completed successfully
Problem_type: Classification
Model: AdaBoostClassifier
Output feature: Outcome
Categorical features: []
Numerical features: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
Train accuracy: 0.8235
Train F1: 0.8245
Train precision: 0.8201
Train recall: 0.8289
Train rocauc: 0.9052
Test accuracy: 0.7576
Test F1: 0.6889
Test precision: 0.6263
Test recall: 0.7654
Test rocauc: 0.8284
Hyper tuned: False
Dropped Columns: []
```

---

### 2️⃣ Predict with a Trained Model

#### **CLI Usage**

```bash
mlforge-predict \
    --model_path path/to/model.pkl \
    --preprocessor_path path/to/preprocessor.pkl \
    --input_data path/to/input.csv \
    --encoder_path path/to/encoder.pkl
```

#### **Python API Usage**

```python
from mlforge import predict

predictions = predict(
    model_path="model.pkl",
    preprocessor_path="preprocessor.pkl",
    input_data_path="input.csv",
    encoder_path="encoder.pkl"
)

print(predictions)
```

---

## 📂 Artifacts

After training, `mlforgex` generates the following files:

| File | Description |
|------|-------------|
| `model.pkl` | Trained ML model |
| `preprocessor.pkl` | Preprocessing pipeline (scaling, encoding, etc.) |
| `encoder.pkl` | Label encoder (classification only) |
| `Plots/` | Visualization folder containing heatmaps, ROC curves, etc. |

---

## 🛠 CLI Command Reference

### **Train Command**
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
    [--fast <bool>]
```

### **Predict Command**
```bash
mlforge-predict \
    --model_path <model.pkl> \
    --preprocessor_path <preprocessor.pkl> \
    --input_data <input.csv> \
    [--encoder_path <encoder.pkl>]
```

---

## ⚡ Example Workflow

```bash
# Step 1: Train the model
mlforge-train --data_path housing.csv --dependent_feature Price --rmse_prob 0.3 --f1_prob 0.7

# Step 2: Use the trained model for predictions
mlforge-predict --model_path artifacts/model.pkl --preprocessor_path artifacts/preprocessor.pkl --input_data new_data.csv
```

---

## 🧪 Testing

Run all tests with:

```bash
pytest test/
```

---

## ❗ Troubleshooting & Common Errors

- **"Target is multiclass but average='binary'"**  
  This happens when using binary metrics on a multiclass dataset.  
  ✅ Fix: Use `average='macro'` or `average='weighted'` in metrics computation.

- **"FileNotFoundError"**  
  Ensure all file paths are correct and accessible.

- **"ModuleNotFoundError"**  
  Install missing dependencies with:  
  ```bash
  pip install -r requirements.txt
  ```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Priyanshu Mathur**  
📧 Email: mathurpriyanshu2006@gmail.com  
🌐 [Portfolio](https://my-portfolio-phi-two-53.vercel.app/)  
📦 [PyPI Package](https://pypi.org/project/mlforgex/)  
