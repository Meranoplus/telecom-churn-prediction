# Telecom Customer Churn Prediction

A machine learning pipeline that predicts customer churn for a telecom company using structured customer data stored in a MySQL database.

## Results

| Metric | Score |
|--------|-------|
| Precision | 0.6327 |
| Recall | 0.7460 |
| F1 | 0.6847 |
| ROC-AUC | 0.8950 |
| PR-AUC | 0.7606 |

## Pipeline Overview

**Step 1 — Base Model Comparison**
Five models are evaluated using 10-fold stratified cross-validation scored on recall: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.

**Step 2 — Hyperparameter Tuning**
The top two models (XGBoost and CatBoost) are tuned using `RandomizedSearchCV`. XGBoost uses `scale_pos_weight` set to the inverse class ratio (~2.77) to handle class imbalance.

**Step 3 — Threshold Tuning**
Instead of using the default 0.5 threshold, out-of-fold probabilities from the winning model are used to find the threshold that maximizes precision while keeping recall at or above 0.75.

**Step 4 — Final Evaluation**
The tuned model is retrained on the full training set and evaluated on the held-out test set.

## Preprocessing

- Dropped identifiers and high-cardinality location columns
- Binary encoded gender, marital status, and yes/no service flags
- One-hot encoded contract type, payment method, internet type, and offer
- Three-value encoded service columns: `1` = using, `0` = not using, `-1` = service not available
- Features scaled with `StandardScaler`

## Setup

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm sqlalchemy pymysql python-dotenv
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=your_host
DB_NAME=your_database
DB_TABLE=your_table
```

### 3. Run

```bash
python Churn_detection.py
```

## Dependencies

- Python 3.8+
- scikit-learn
- XGBoost
- CatBoost
- LightGBM
- SQLAlchemy + PyMySQL
- pandas, numpy
- python-dotenv

## Dataset

The dataset contains telecom customer records with features including demographics, account info, service subscriptions, and contract details. Target variable is binary: `1` for churned customers, `0` for stayed/joined.

Source: [Kaggle Telecom Customer Churn](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics)

I cleaned the raw dataset and loaded it into MySQL for this pipeline. Main changes:
- Dropped nulls and standardized column names
- Created binary target from `Customer Status`
- Loaded into `telecom_churn2` table for SQL-based workflow

Target: 1 = churned, 0 = stayed/joined. Class distribution: ~73.5% / ~26.5%.
