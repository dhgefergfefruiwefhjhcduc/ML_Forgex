<div align="center">
  <!-- TODO: Replace with the provided logo URL -->
  <img src="https://res.cloudinary.com/datrajdgv/image/upload/v1759823370/logo_vmruwp.png" alt="mlforgex Logo" width="200" />

  <br />

  <strong>Automated Machine Learning utility for streamlined training, evaluation, and prediction.</strong>
  
  <br />
  <br />

  <!-- Badges -->
  <a href="https://pypi.org/project/mlforgex/"><img src="https://img.shields.io/pypi/v/mlforgex?color=blue&logo=pypi&logoColor=white" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/mlforgex/"><img src="https://img.shields.io/pypi/pyversions/mlforgex?color=blue&logo=python&logoColor=white" alt="Python versions" /></a>
  <a href="https://github.com/dhgefergfefruiwefhjhcduc/ML_Forgex/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dhgefergfefruiwefhjhcduc/ML_Forgex?color=green" alt="License" /></a>
  <a href="https://github.com/dhgefergfefruiwefhjhcduc/ML_Forgex/stargazers"><img src="https://img.shields.io/github/stars/dhgefergfefruiwefhjhcduc/ML_Forgex?style=social" alt="Stars" /></a>
  <a href="https://dhgefergfefruiwefhjhcduc.github.io/mlforgex_documentation/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat" alt="Documentation" /></a>
<a href="https://github.com/dhgefergfefruiwefhjhcduc/ML_Forgex">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github&logoColor=white" alt="GitHub Repository" />
</a>
 <a href="https://pepy.tech/projects/mlforgex"><img src="https://static.pepy.tech/personalized-badge/mlforgex?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads" alt="PyPI Downloads"></a>
          <a href="https://github.com/josephmisiti/awesome-machine-learning" target="_blank"><img src="https://awesome.re/badge.svg" alt="Listed in Awesome Machine Learning"></img></a>

  <br />
  <br />
  
  <i>Effortlessly scale your ML workflows with built-in preprocessing, hyperparameter tuning, ensembling, and an interactive dashboard.</i>
</div>

---

## 🚀 What is mlforgex?

**mlforgex** is a lightweight, high-performance automated machine learning (AutoML) utility designed to drastically reduce boilerplate code. Whether you're working on Regression, Classification, or Natural Language Processing (NLP) tasks, mlforgex handles everything from data cleaning and scaling to advanced hyperparameter tuning and model ensembling.

With just a few lines of code, you get a fully trained pipeline and a beautiful interactive HTML dashboard of your model's performance.

## ✨ Key Features

- 🧠 **End-to-End AutoML**: Automatically detects the problem type (Regression or Classification) and trains the best-in-class algorithms (XGBoost, LightGBM, CatBoost, Random Forest, etc.).
- 🧹 **Intelligent Preprocessing**: Built-in outlier removal, skewness handling, highly-correlated feature dropping, and one-hot/ordinal encoding.
- ⚖️ **Imbalance Handling**: Automatically detects and handles mild to severe class imbalances using Random Under-sampling or SMOTETomek.
- 📝 **Native NLP Support**: Processes text data natively—tokenization, stopword removal, lemmatization, and Word2Vec embeddings out-of-the-box.
- 🚀 **Advanced Ensembles**: Automatically constructs Stacking and Voting ensembles from the top-performing tuned models to maximize accuracy and minimize overfitting.
- 📊 **Interactive Dashboards**: Generates comprehensive Plotly-based HTML dashboards with feature importance, correlation heatmaps, and metric curves (ROC, Precision-Recall).

## 📦 Installation

mlforgex is available on PyPI. Install it using pip:

```bash
pip install mlforgex
```

*(Requires Python 3.8 or higher)*

## ⚡ Quick Start

mlforgex makes training highly optimized models incredibly simple. Just provide the path to your dataset and the name of the target column.

### Training a Model

```python
from mlforgex.train import train_model

# Train a model for Classification or Regression automatically
train_model(
    data_path="data/dataset.csv",
    dependent_feature="target_column",
    n_iter=100,               # Number of randomized search iterations
    fast=False,               # Set to True for a quick baseline
    artifacts_dir="output",   # Directory to save models and dashboard
    dashboard_title="My Project Dashboard"
)
```

### Training an NLP Model

```python
from mlforgex.train import train_model

# Enable NLP mode for text classification
train_model(
    data_path="data/reviews.csv",
    dependent_feature="sentiment",
    nlp=True,                 # Turn on NLP processing
    artifacts_dir="nlp_output"
)
```

### Making Predictions

Once trained, mlforgex saves a `model.pkl` and `preprocessor.pkl` inside your artifacts directory. 

```python
import pandas as pd
from mlforgex.predict import predict_data

# Load new unseen data
new_data = pd.read_csv("data/new_data.csv")

# Generate predictions using the saved artifacts
predictions = predict_data(
    data=new_data,
    artifacts_path="output/artifacts"
)

print(predictions)
```

## 📈 Interactive Dashboard

After every training run, mlforgex automatically generates a `Dashboard.html` in your artifacts folder. Open it in any web browser to explore:
- Data insights and distribution shapes.
- Correlation heatmaps.
- Feature importances.
- Interactive evaluation metric curves (e.g., ROC-AUC, Learning Curves).

## 🛠 Command Line Interface (CLI)

You can also use mlforgex directly from the terminal!

```bash
# Train a model
mlforgex-train --data my_data.csv --target price --out_dir my_models

# Make predictions
mlforgex-predict --data unseen.csv --artifacts my_models/artifacts
```

## 📘 Documentation

For advanced usage, custom thresholds, and API references, check out the full documentation:

👉 **[mlforgex Official Documentation](https://dhgefergfefruiwefhjhcduc.github.io/mlforgex_documentation/)**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. If you find a bug or have a feature request, please open an issue.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  Made with ❤️ by <a href="https://github.com/dhgefergfefruiwefhjhcduc">Priyanshu Mathur</a>
</div>
