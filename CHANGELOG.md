# Changelog

All notable changes to this project will be documented in this file.

## [version - 1.2.0] - 2026-07-15

### Added
- **LightGBM integration:** Full support for `LGBMClassifier` and `LGBMRegressor` models, optimized search grids in `params.py`, and updated dependencies in `requirements.txt`.
- **Smart data cleaning pipeline:** Implemented intelligent missing value handling (using a `"Missing"` category for categorical columns with >10% missing values) and target variable protection during cleaning.
- **Advanced outlier handling:** Replaced row deletion with Winsorization (Outlier Capping) supporting IQR and Z-score-based clipping.
- **Smart skewness normalization:** Integrated Yeo-Johnson Power Transformation into the preprocessing pipeline for skewed numerical features.
- **Automated inference consistency:** Embedded all transformations into the serialized `preprocessor.pkl` pipeline to eliminate extra logic in `predict.py`.
- **Multi-model saving:** Saved the top 3 ranked models individually by algorithm name (e.g., `xgbclassifier.pkl`) for easier benchmarking and ensembling.
- **Centralized logging system:** Introduced a unified `mlforgex` logger across the project, routing stdout logs to `artifacts/training_logs.log`.
- **Deep execution tracking:** Added detailed pipeline stage execution logs covering dataset dimensions, feature engineering, and Imbalance Ratios.
- **Performance profiling:** Tracked execution times across major pipeline stages, including data loading, training, and dashboard generation.

### Changed
- **Pipeline evaluation refactoring:** Removed over 150 lines of duplicated evaluation logic by introducing reusable `evaluate_regression_model()` and `evaluate_classification_model()` helpers.
- **Structured console output:** Replaced all legacy `print()` statements with structured logging across `train.py`, `predict.py`, `cleaning.py`, and `dashboard.py`.
- **Dashboard UI enhancements:** Redesigned the HTML dashboard table using premium glassmorphism styling, alternating row colors, and interactive hover effects.

### Fixed
- **Dashboard model visibility:** Resolved an issue where tree-based models exceeding the overfitting threshold were erroneously omitted from the dashboard.
- **Temporary artifact cleanup:** Fixed an intermittent operating system `WinError 3` occurring during cleanup of the temporary artifacts directory.
