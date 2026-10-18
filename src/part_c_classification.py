
import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.shipment_prep import FEATURES, NOISE_COLUMNS

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "modelling_table.csv"))

X = df[FEATURES].copy()
y = df["is_delayed"].astype(int)

# Stratified 70/15/15 split, using the assignment's random seed.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Split sizes:", len(X_train), len(X_val), len(X_test))
print("Delayed counts:", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

def score_model(y_true, pred, prob=None):
    result = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if prob is not None:
        result["roc_auc"] = roc_auc_score(y_true, prob)
    return result

# Baseline: predict every shipment as on time.
baseline_pred = np.zeros(len(y_val), dtype=int)
baseline_metrics = score_model(y_val, baseline_pred)

# Logistic Regression with scaling.
logistic = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])
logistic.fit(X_train, y_train)
logistic_prob = logistic.predict_proba(X_val)[:, 1]
logistic_pred = (logistic_prob >= 0.5).astype(int)

# Select Decision Tree depth using validation F1; ties go to the smaller depth.
tree_rows = []
for depth in [2, 3, 4, 5, 6, 8, 10]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    tree_rows.append({
        "max_depth": depth,
        **score_model(y_val, pred),
    })

tree_depth_df = pd.DataFrame(tree_rows)
tree_depth_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_c_tree_depths.csv"), index=False
)
best_depth = int(
    tree_depth_df.sort_values(
        ["f1", "max_depth"], ascending=[False, True]
    ).iloc[0]["max_depth"]
)

tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
tree.fit(X_train, y_train)
tree_prob = tree.predict_proba(X_val)[:, 1]
tree_pred = (tree_prob >= 0.5).astype(int)

# Random Forest baseline model.
rf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_val)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)

# Univariate F-scores and Random Forest feature importance, training data only.
f_scores, _ = f_classif(X_train, y_train)
importance = rf.feature_importances_

feature_report = pd.DataFrame({
    "feature": FEATURES,
    "f_score": np.nan_to_num(f_scores, nan=0.0),
    "rf_importance": importance,
})
feature_report = feature_report.sort_values(
    "rf_importance", ascending=False
)
feature_report["importance_rank"] = np.arange(1, len(feature_report) + 1)
feature_report.to_csv(
    os.path.join(OUTPUT_DIR, "part_c_feature_importance.csv"), index=False
)

# Evaluate several SelectKBest values on validation data.
selection_rows = []
selection_models = {}

for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
    ])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    selection_models[k] = pipe
    selection_rows.append({
        "k": k,
        **score_model(y_val, pred, prob),
    })

selection_df = pd.DataFrame(selection_rows)
selection_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_c_feature_selection.csv"), index=False
)

# Required final selector size.
K_FINAL = 7

# Grid search on training data only.
grid_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_classif, k=K_FINAL)),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
grid = GridSearchCV(
    grid_pipe,
    {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [4, 6, 8],
        "model__min_samples_leaf": [1, 5, 10],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

# Randomized search with the same parameter choices.
random_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_classif, k=K_FINAL)),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
random_search = RandomizedSearchCV(
    random_pipe,
    {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [4, 6, 8],
        "model__min_samples_leaf": [1, 5, 10],
    },
    n_iter=9,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)

# Compare default, grid-search, and randomized-search models on validation data.
tuning_candidates = [
    ("default", selection_models[K_FINAL], 0),
    ("grid", grid.best_estimator_, 1),
    ("randomized", random_search.best_estimator_, 2),
]

tuning_rows = []
candidate_probabilities = {}

for name, model, priority in tuning_candidates:
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = score_model(y_val, pred, prob)
    candidate_probabilities[name] = prob
    tuning_rows.append({
        "method": name,
        **metrics,
        "tie_priority": priority,
    })

tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_c_tuning.csv"), index=False
)

# Highest validation F1 wins; ties prefer default, then grid, then randomized.
winner_row = tuning_df.sort_values(
    ["f1", "tie_priority"], ascending=[False, True]
).iloc[0]
winner_name = winner_row["method"]
winner_model = {name: model for name, model, _ in tuning_candidates}[winner_name]

# Choose the highest threshold whose validation recall is at least 0.80.
chosen_threshold = 0.1
threshold_rows = []

for threshold in np.round(np.arange(0.1, 1.0, 0.1), 1):
    pred = (candidate_probabilities[winner_name] >= threshold).astype(int)
    m = score_model(y_val, pred)
    threshold_rows.append({"threshold": float(threshold), **m})
    if m["recall"] >= 0.80:
        chosen_threshold = float(threshold)

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_c_thresholds.csv"), index=False
)

# Fit final selected pipeline on training rows only and score test once.
winner_model.fit(X_train, y_train)
test_prob = winner_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= chosen_threshold).astype(int)
test_metrics = score_model(y_test, test_pred, test_prob)
tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

final_config = {
    "winner": winner_name,
    "threshold": chosen_threshold,
    "k_features": K_FINAL,
    "best_tree_depth": best_depth,
    "grid_best_params": grid.best_params_,
    "randomized_best_params": random_search.best_params_,
    "test_metrics": test_metrics,
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}

with open(os.path.join(OUTPUT_DIR, "part_c_final_config.json"), "w") as f:
    json.dump(final_config, f, indent=2)

pd.DataFrame([{
    "winner": winner_name,
    "threshold": chosen_threshold,
    **test_metrics,
    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
}]).to_csv(
    os.path.join(OUTPUT_DIR, "part_c_final_test_metrics.csv"), index=False
)

print("\nBaseline:", baseline_metrics)
print("\nLogistic Regression:", score_model(y_val, logistic_pred, logistic_prob))
print("\nBest tree depth:", best_depth)
print("Decision Tree:", score_model(y_val, tree_pred, tree_prob))
print("\nRandom Forest:", score_model(y_val, rf_pred, rf_prob))
print("\nFeature importance:")
print(feature_report.to_string(index=False))
print("\nFeature selection comparison:")
print(selection_df.round(4).to_string(index=False))
print("\nTuning comparison:")
print(tuning_df.round(4).to_string(index=False))
print("\nChosen threshold:", chosen_threshold)
print("\nFinal test metrics:", test_metrics)
print("Confusion matrix:", {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
print("\nPart C reports saved in outputs/")
