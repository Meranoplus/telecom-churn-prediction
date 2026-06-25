import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, cross_val_predict)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ── Helpers ───────────────────────────────────────────────

def evaluate(model, X, y, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return {
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall":    recall_score(y, y_pred),
        "F1":        f1_score(y, y_pred),
        "ROC-AUC":   roc_auc_score(y, y_proba),
        "PR-AUC":    average_precision_score(y, y_proba)
    }


def print_metrics(results: dict):
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


def find_threshold(y_true, y_proba, target_recall=0.75):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    mask = recalls[:-1] >= target_recall
    if mask.any():
        return thresholds[mask][np.argmax(precisions[:-1][mask])]
    return 0.5


# ── Config ────────────────────────────────────────────────

config = {
    'random_state':  42,
    'test_size':     0.2,
    'n_splits':      10,
    'target_recall': 0.75
}

# ── Base model params ─────────────────────────────────────

lr_params = {
    'max_iter':              1000,
    'cv':                    5,
    'random_state':          42,
    'n_jobs':                -1,
    'l1_ratios':             (0.0,),
    'use_legacy_attributes': False
}

rf_params = {
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs':       -1
}

xgb_params = {
    'n_estimators':  500,
    'learning_rate': 0.1,
    'max_depth':     6,
    'random_state':  42,
    'n_jobs':        -1
}

lgbm_params = {
    'n_estimators':  500,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  42,
    'n_jobs':        -1,
    'verbose':       -1,
    'objective':     'binary'
}

cat_params = {
    'iterations':    500,
    'learning_rate': 0.1,
    'depth':         6,
    'random_state':  42,
    'verbose':       0
}

# ── Tuning grids ──────────────────────────────────────────

xgb_param_grid = {
    'n_estimators':     [300, 500, 700],
    'learning_rate':    [0.01, 0.1, 0.2],
    'max_depth':        [3, 6, 9],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma':            [0, 0.1, 0.3],
}

cat_param_grid = {
    'iterations':          [300, 500, 700],
    'learning_rate':       [0.01, 0.1, 0.2],
    'depth':               [4, 6, 8],
    'l2_leaf_reg':         [1, 3, 5, 10],
    'bagging_temperature': [0, 0.5, 1.0]
}

# ── Database connection ───────────────────────────────────

load_dotenv()
user     = os.getenv("DB_USER")
password = quote_plus(os.getenv("DB_PASSWORD"))
host     = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
table    = os.getenv("DB_TABLE")

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
df     = pd.read_sql(f"SELECT * FROM {table}", engine)

# ── Preprocessing ─────────────────────────────────────────

# Drop identifiers and high-cardinality location cols
df.drop(columns=["Customer ID", "Churn Reason", "Zip Code",
                  "Latitude", "Longitude", "City",
                  "Churn Category"], inplace=True)

# Binary encode
binary_map  = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
binary_cols = ["Gender", "Married", "Phone Service", "Paperless Billing"]
for col in binary_cols:
    df[col] = df[col].map(binary_map)

# Target: Churned=1, Stayed/Joined=0
df["Churn"] = (df["Customer Status"] == "Churned").astype(int)
df.drop(columns=["Customer Status"], inplace=True)

# One-hot with drop_first to avoid collinearity
df = pd.get_dummies(df, columns=["Contract", "Payment Method",
                                  "Internet Type", "Offer"], drop_first=True)

# Three-value cols: -1 = no service available, 0 = not using, 1 = using
three_val_map  = {"Yes": 1, "No": 0,
                  "No Internet Service": -1,
                  "No Phone Service":    -1}
three_val_cols = ["Multiple Lines", "Internet Service", "Online Security",
                  "Online Backup", "Device Protection Plan", "Premium Tech Support",
                  "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data"]
for col in three_val_cols:
    df[col] = df[col].map(three_val_map)

# Sanity check
remaining = df.select_dtypes(include='object').columns.tolist()
if remaining:
    print(f"Still has text columns: {remaining}")
else:
    print("All columns numeric")

# ── Features & target ─────────────────────────────────────

X = df.drop(columns=["Churn"])
y = df["Churn"]

# ── Train/test split ──────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["test_size"],
    random_state=config["random_state"],
    stratify=y
)

# ── Scaling ───────────────────────────────────────────────

scaler         = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

# ── Step 1: Compare base models ───────────────────────────

print("\n" + "="*50)
print("STEP 1: BASE MODEL COMPARISON")
print("="*50)

skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True,
                      random_state=config["random_state"])

models = {
    "logistic": LogisticRegressionCV(**lr_params),
    "rf":       RandomForestClassifier(**rf_params),
    "xgb":      XGBClassifier(**xgb_params),
    "lgbm":     LGBMClassifier(**lgbm_params),
    "catboost": CatBoostClassifier(**cat_params)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=skf, scoring="recall")
    print(f"  {name}: Recall = {scores.mean():.4f} (+/- {scores.std():.4f})")

# ── Step 2: Tune top 2 models ─────────────────────────────

print("\n" + "="*50)
print("STEP 2: HYPERPARAMETER TUNING")
print("="*50)

# inverse class ratio for imbalanced target
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_grid = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight),
    param_distributions=xgb_param_grid,
    n_iter=20, cv=skf,
    scoring="recall",
    n_jobs=-1, refit=True, verbose=0,
    random_state=config["random_state"]
)
xgb_grid.fit(X_train_scaled, y_train)
print(f"  XGBoost  best recall (CV): {xgb_grid.best_score_:.4f}")

cat_grid = RandomizedSearchCV(
    estimator=CatBoostClassifier(random_state=42, verbose=0),
    param_distributions=cat_param_grid,
    n_iter=20, cv=skf,
    scoring="recall",
    n_jobs=-1, refit=True, verbose=0,
    random_state=config["random_state"]
)
cat_grid.fit(X_train_scaled, y_train)
print(f"  CatBoost best recall (CV): {cat_grid.best_score_:.4f}")

if xgb_grid.best_score_ >= cat_grid.best_score_:
    winner, final_model = "XGBoost",  xgb_grid.best_estimator_
else:
    winner, final_model = "CatBoost", cat_grid.best_estimator_

print(f"\nWinner: {winner}")

# ── Step 3: Threshold tuning ──────────────────────────────

print("\n" + "="*50)
print("STEP 3: THRESHOLD TUNING")
print("="*50)

# OOF probs to avoid test leakage
y_proba_train = cross_val_predict(
    final_model, X_train_scaled, y_train,
    cv=skf, method="predict_proba"
)[:, 1]

best_threshold = find_threshold(
    y_train, y_proba_train,
    target_recall=config["target_recall"]
)
print(f"  Best threshold: {best_threshold:.4f}")

# ── Step 4: Final evaluation ──────────────────────────────

print("\n" + "="*50)
print("STEP 4: FINAL TEST RESULTS")
print("="*50)

final_model.fit(X_train_scaled, y_train)
results = evaluate(final_model, X_test_scaled, y_test, threshold=best_threshold)
print_metrics(results)

if results["Recall"] < 0.5:
    print("Warning: Low recall — consider lowering threshold in config")