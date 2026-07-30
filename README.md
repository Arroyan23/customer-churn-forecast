# 📉 Customer Churn Forecast

**Predicting whether a telecom customer is about to churn — powered by XGBoost, scikit-learn pipelines, and a healthy dose of EDA-driven feature engineering.**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Preprocessing%20Pipeline-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-brightgreen?logo=xgboost)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability%20(WIP)-9cf)
![Accuracy](https://img.shields.io/badge/Holdout%20Accuracy-81.9%25-success)

</div>

---

## 🧭 Overview

Customer churn is one of the most expensive problems in subscription-based businesses — losing a customer is far costlier than retaining one. This project tackles the classic **telecom churn prediction** problem: given a customer's demographics, subscribed services, contract type, and billing history, predict whether they will **churn (`Yes`) or stay (`No`)**.

The workflow follows a deliberately linear path: **understand the data → visualize the patterns → engineer smarter features → train & evaluate a model.** Every insight from the EDA phase directly informed a feature engineering decision down the line — nothing here is guesswork.

## 📁 Repository Structure

```
customer-churn-forecast/
├── data-viz.ipynb     # Exploratory Data Analysis — distributions, correlations, risk segments
├── model.ipynb        # Preprocessing pipeline, XGBoost training, evaluation & submission
├── latihan.ipynb       # Scratch/sandbox notebook for quick experiments
├── train.csv           # Training data (594,194 rows)
├── test.csv             # Test data (for final predictions)
├── submission.csv       # Final prediction output
└── README.md
```

## 🗃️ Dataset

A telecom-style customer dataset with **594,194 rows** and **21 columns**, covering demographics, subscribed services, contract details, and billing — no missing values, which kept preprocessing refreshingly clean.

| Category | Features |
|---|---|
| 👤 Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| 📡 Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| 📄 Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| 💵 Billing | `MonthlyCharges`, `TotalCharges` |
| 🎯 Target | `Churn` (`Yes` / `No`) |

## 🔍 Exploratory Data Analysis

The full analysis lives in [`data-viz.ipynb`](./data-viz.ipynb). Here's the story it tells:

### 1. The dataset is imbalanced

```
No   ██████████████████████████████████████  77.5%
Yes  ████████████████                         22.5%
```

Roughly 3 out of 4 customers stay. This imbalance is exactly why `scale_pos_weight` was baked into the XGBoost model later — without it, the model would happily predict "No" for everyone and still look "accurate."

### 2. Billing behavior vs. churn

A `histplot` of `MonthlyCharges` split by `Churn` showed that customers who churn tend to cluster at **higher monthly charge brackets**, while loyal customers skew toward lower, more predictable bills. A `pairplot` between `MonthlyCharges` and `TotalCharges` reinforced the same signal from a different angle.

### 3. Raw `TotalCharges` is a biased number

Since `TotalCharges ≈ tenure × MonthlyCharges`, it naturally rewards long-tenured customers with a bigger number — even if they're actually risky payers. That observation led to a much better metric:

$$\text{Avg\_Monthly\_Paid} = \frac{\text{TotalCharges}}{\text{tenure}}$$

This normalizes billing by subscription length, revealing "true" spending intensity instead of just tenure accumulation.

### 4. The killer insight — risk segmentation

Bucketing customers by `Avg_Monthly_Paid` into **Low / Medium / High** risk groups and cross-tabulating against churn rate uncovered a clean, monotonic trend:

```
Low Risk      ██                                    2.9%  churn rate
Medium Risk   ████████████████                     26.6%  churn rate
High Risk     ████████████████████                 33.2%  churn rate
```

**The higher a customer's average monthly spend relative to their tenure, the more likely they are to churn.** This single derived feature turned out to be one of the strongest behavioral signals in the whole dataset.

## 🛠️ Feature Engineering

Straight from the EDA findings, the following features were engineered on top of the raw data:

| Feature | Formula / Logic | Why it matters |
|---|---|---|
| `Total_Charges` | `tenure × MonthlyCharges` | Sanity-check / reconstructed billing total |
| `Avg_Monthly_Paid` | `TotalCharges / tenure` | Normalized spend, independent of subscription length |
| `Avg_Monthly_Paid_Group` | Binned into `low_risk` / `medium_risk` / `high_risk` | Directly encodes the churn-risk pattern discovered in EDA |
| `Charge_Group` | `MonthlyCharges` binned into `low` / `medium` / `high` | Captures pricing-tier sensitivity |

All categorical engineered groups are label-encoded (`0`, `1`, `2`) for direct model consumption.

## ⚙️ Preprocessing Pipeline

Built with scikit-learn's `ColumnTransformer` inside [`model.ipynb`](./model.ipynb):

```python
preprocessing = ColumnTransformer(transformers=[
    ('numerical_cols', SimpleImputer(strategy='mean'), numerical_cols),
    ('categorical_cols', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols)
])
```

- **Numerical columns** → mean imputation (safety net, even though the raw data has no nulls)
- **Categorical columns** (15 of them, all low-cardinality: 2–4 unique values) → most-frequent imputation + one-hot encoding
- **Target (`Churn`)** → `LabelEncoder` (`No → 0`, `Yes → 1`)

## 🤖 Model

The classifier of choice is **`XGBClassifier`**, tuned to respect the class imbalance uncovered during EDA:

```python
ratio = (y_train == 0).sum() / (y_train == 1).sum()

model_xgb = XGBClassifier(
    n_estimators=200,
    scale_pos_weight=ratio,
    learning_rate=0.05,
)

my_pipeline = Pipeline(steps=[
    ('preprocess', preprocessing),
    ('model', model_xgb)
])
```

### 📈 Result

| Metric | Score |
|---|---|
| Holdout Accuracy (80/20 split) | **81.9%** |

### 🔮 What's next

- 🧩 Wiring the newly engineered features (`Avg_Monthly_Paid_Group`, `Charge_Group`, etc.) into the production pipeline — currently validated in EDA but not yet merged into `model.ipynb`'s final training run
- 🧠 SHAP-based feature importance / explainability (import already scaffolded in the notebook, explainer not yet wired up)
- 🎯 Precision/Recall/F1 and ROC-AUC evaluation beyond raw accuracy, given the class imbalance

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Arroyan23/customer-churn-forecast.git
cd customer-churn-forecast

# Install dependencies
pip install pandas scikit-learn xgboost seaborn matplotlib shap

# Explore the data
jupyter notebook data-viz.ipynb

# Train the model
jupyter notebook model.ipynb
```

## 🧰 Tech Stack

- **Python** — core language
- **Pandas** — data wrangling & feature engineering
- **Seaborn / Matplotlib** — exploratory visualization
- **scikit-learn** — preprocessing pipelines, encoding, imputation
- **XGBoost** — gradient-boosted classification model
- **SHAP** *(planned)* — model explainability

## 👤 Author

**Arroyan23**
Electrical Engineering Student | ML & Data Enthusiast

---

⭐️ If this project's approach to EDA-driven feature engineering was useful to you, consider dropping a star!
