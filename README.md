# Loan Approval Prediction System

An end-to-end Machine Learning web application built with **Streamlit** that predicts loan approval and estimates default risk. The system processes applicant financial profiles through an optimized **Random Forest** classification pipeline tracked and logged using **MLflow**.

---

## Project Overview

Financial institutions face a dual challenge: approving loans to qualified candidates while minimizing default risks. This project delivers an automated credit scoring solution designed to:
* **Predict Loan Approval Status:** Classify applications into approved or high-risk categories.
* **Assess Risk Probability:** Calculate the exact probability of default for informed decision-making.
* **Explain Decision Criteria:** Provide complete transparency into feature selection, model hyperparameters, and information entropy calculations within an interactive multi-page dashboard.

---

## Stack
![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

## Model Architecture
#### 1. Algorithm Selection: Random Forest
The decision engine uses a **Random Forest Classifier** ensemble. By combining predictions across multiple decision trees, the model achieves high stability, resists overfitting, and captures complex non-linear financial interactions.

#### 2. Preprocessing & Feature Engineering Pipeline
To prevent training-serving skew, all data preprocessing steps are encapsulated into a single Scikit-Learn `Pipeline`:
* **Feature Selection:** Preliminarily filtered features based on cumulative Gini importance threshold (>= 96%).
* **Categorical Encoding:** `OrdinalEncoder` mapping categorical attributes.
* **Discretization:** `DecisionTreeDiscretiser` grouping continuous numerical attributes into optimal bins.

#### 3. Hyperparameter Optimization & Entropy Criterion
Hyperparameters were optimized using **GridSearchCV** with 3-fold cross-validation evaluated on **ROC-AUC**:
* `n_estimators`: `700`
* `min_samples_leaf`: `50`
* `criterion`: `'entropy'`

### For more details and see the model for yourself, access: https://loan-approval-project-wxprkxfaret4jssp9cghm4.streamlit.app/
#### Dataset: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
