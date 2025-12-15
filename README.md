# Wine Quality Classification with Deep Learning

## Overview

Deep learning and machine learning project focused on binary classification of wine quality (high vs. low) using physicochemical features from the Wine Quality (Red Wine) dataset.

## Objective

To evaluate whether Deep Neural Networks (DNN) can outperform classical machine learning models for wine quality prediction, and to analyze the trade-off between model complexity, performance, and operational efficiency.

## Dataset

* Wine Quality Dataset (Red Wine)
* Fully numerical physicochemical features
* Target variable transformed into two classes: **High Quality** and **Low Quality**

Key features include acidity, alcohol content, sulphates, pH, density, and sulfur dioxide levels.

## Methodology

The project follows a structured machine learning workflow:

* Exploratory Data Analysis (EDA)
* Feature engineering and target transformation
* Outlier analysis
* Feature scaling (StandardScaler)
* Model training and comparison

Both classical machine learning models and deep learning architectures were evaluated under the same preprocessing pipeline.

## Models Evaluated

### Classical Machine Learning

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest
* Bagging
* Gradient Boosting (XGBoost)

### Deep Learning

* Deep Neural Networks (DNN)

  * Multiple architectures with varying depth
  * Dropout regularization
  * L1 and L2 regularization
  * Adam optimizer and binary cross-entropy loss

## Evaluation Metrics

* ROC AUC (primary metric)
* Accuracy
* F1-score

## Results

* **Best overall model:** XGBoost
* **Accuracy:** 0.82
* **ROC AUC:** 0.88
* **F1-score:** 0.83

Among DNN architectures, the deepest model with combined regularization achieved the best performance:

* **Best DNN accuracy:** 76.5%

Tree-based ensemble models consistently outperformed deep neural networks, demonstrating superior performance with lower computational complexity.

## Key Insights

* Increasing model complexity does not necessarily lead to better performance.
* Ensemble tree-based models are highly effective for structured tabular data.
* Deep learning models require careful regularization and tuning but did not surpass classical ensembles in this use case.
* XGBoost provides the best balance between predictive performance and operational efficiency.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Google Colab
* Matplotlib / Seaborn

## Author

Nicolas Salinas

---

This project is part of a Data Science training program and is intended for educational and portfolio purposes.
