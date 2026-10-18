import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/content/AI-ML-Capstone-Project"
sys.path.insert(0, ROOT)

from src.shipment_prep import FEATURES

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "modelling_table.csv"))
X = df[FEATURES].copy()
y = df["is_delayed"].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

config_path = os.path.join(OUTPUT_DIR, "part_c_final_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

print("Config keys:", list(config.keys()))
print("Config:", config)

# Rebuild the final model from the saved configuration.
# Expected configuration contains the selected feature count and RF parameters.
k = int(config.get("k", config.get("selected_k", 7)))
params = config.get("best_params", config.get("model_params", {}))

clean_params = {}
for key, value in params.items():
    key = key.replace("classifier__", "").replace("model__", "")
    if key in {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"}:
        clean_params[key] = value

clean_params.setdefault("n_estimators", 100)
clean_params.setdefault("random_state", 42)
clean_params.setdefault("n_jobs", -1)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_classif, k=k)),
    ("classifier", RandomForestClassifier(**clean_params)),
])

model.fit(X_train, y_train)

# Load the selected threshold from the saved Part C threshold report.
threshold = float(config.get("threshold", 0.5))
threshold_path = os.path.join(OUTPUT_DIR, "part_c_thresholds.csv")

if os.path.exists(threshold_path):
    thresholds = pd.read_csv(threshold_path)
    if {"threshold", "recall"}.issubset(thresholds.columns):
        eligible = thresholds[thresholds["recall"] >= 0.80]
        if not eligible.empty:
            threshold = float(eligible["threshold"].max())

model_path = os.path.join(MODEL_DIR, "delay_classifier.joblib")
joblib.dump(model, model_path)

pd.DataFrame({"feature": FEATURES}).to_csv(
    os.path.join(MODEL_DIR, "feature_order.csv"), index=False
)

# Save training medians for reproducible preprocessing.
pd.DataFrame({
    "feature": FEATURES,
    "median": [
        float(X_train[col].median()) if X_train[col].notna().any() else 0.0
        for col in FEATURES
    ],
}).to_csv(os.path.join(MODEL_DIR, "feature_medians.csv"), index=False)

with open(os.path.join(MODEL_DIR, "decision_threshold.json"), "w") as f:
    json.dump({"threshold": threshold, "selected_k": k, "model_params": clean_params}, f, indent=2)

# Reload and verify identical probabilities.
reloaded = joblib.load(model_path)
before = model.predict_proba(X_test)[:, 1]
after = reloaded.predict_proba(X_test)[:, 1]

print("\nModel saved:", model_path)
print("Selected features:", k)
print("Decision threshold:", threshold)
print("Test ROC-AUC:", round(roc_auc_score(y_test, after), 4))
print("Maximum probability difference after reload:", float(np.max(np.abs(before - after))))
print("MODEL PACKAGING COMPLETED")
