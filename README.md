# MLForge

**MLForge** is a Python package that enables easy training, evaluation, and prediction for machine learning models. It supports both classification and regression problems, automates preprocessing, model selection, hyperparameter tuning, and generates useful artifacts and plots for analysis.

## Features

- Automatic data preprocessing (missing value handling, encoding, scaling)
- Imbalance handling (under-sampling, over-sampling)
- Model selection and evaluation (classification & regression)
- Hyperparameter tuning with RandomizedSearchCV
- Artifact saving (model, preprocessor, encoder)
- Visualization of metrics and learning curves
- Simple CLI for training and prediction

## Installation

Install MLForge using pip:

```sh
pip install mlforge
```

Or clone the repository and install locally:

```sh
git clone https://github.com/yourusername/mlforge.git
cd mlforge
pip install .
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
mlforge-train --data mlforge/diabetes_cleaned.csv --target Outcome --rmse 0.3 --f1 0.7
```

Or programmatically:

```python
from mlforge import train_model

result = train_model(
    "mlforge/diabetes_cleaned.csv",
    "Outcome",
    rmse_prob=0.3,
    f1_prob=0.7,
    n_jobs=-1
)
print(result)
```

### Predict

Use the CLI:

```sh
mlforge-predict --model mlforge/artifacts/model.pkl --preprocessor mlforge/artifacts/preprocessor.pkl --input mlforge/input.csv --encoder mlforge/artifacts/encoder.pkl
```

Or programmatically:

```python
from mlforge import predict

result = predict(
    "mlforge/artifacts/model.pkl",
    "mlforge/artifacts/preprocessor.pkl",
    "mlforge/input.csv",
    "mlforge/artifacts/encoder.pkl"
)
print(result)
```

## Artifacts

After training, the following files are saved in `mlforge/artifacts/`:

- `model.pkl`: Trained model
- `preprocessor.pkl`: Preprocessing pipeline
- `encoder.pkl`: Label encoder (for classification)
- `Plots/`: Visualizations (correlation heatmap, confusion matrix, ROC curve, etc.)

## Testing

Run tests using pytest:

```sh
pytest test/
```

## License

[MIT License](https://github.com/dhgefergfefruiwefhjhcduc/ML_Forge?tab=MIT-1-ov-file)

## Author

Priyanshu Mathur  
[Portfolio](https://my-portfolio-phi-two-53.vercel.app/)  
Email: mathurpriyanshu2006@gmail.com

## Project Links

- [PyPI](https://pypi.org/project/mlforgex/1.0.0/)
