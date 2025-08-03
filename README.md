# mlforgex [![PyPI Downloads](https://static.pepy.tech/badge/mlforgex)](https://pepy.tech/projects/mlforgex)

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

- [PyPI](https://pypi.org/project/mlforgex/)
